"""
Phase 11 Feature 3 — Human Feedback Loop (persistence layer).

Stores human feedback in a SQLite database.

IMPORTANT: This system LOGS feedback for future use. It does NOT
retrain or update any model. UI and docs say this explicitly.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Optional

from src.config.settings import settings

DB_PATH = settings.data_dir / "feedback.db"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                customer_message TEXT,
                predicted_intent TEXT,
                corrected_intent TEXT,
                predicted_decision TEXT,
                overridden_decision TEXT,
                reply_accepted INTEGER,  -- 1=accept, 0=reject
                edited_reply TEXT,
                notes TEXT,
                thread_id INTEGER,
                pipeline_result_json TEXT
            )
        """)
        conn.commit()


def save_feedback(
    customer_message: str,
    predicted_intent: str,
    corrected_intent: Optional[str],
    predicted_decision: str,
    overridden_decision: Optional[str],
    reply_accepted: Optional[bool],
    edited_reply: Optional[str],
    notes: Optional[str] = None,
    thread_id: Optional[int] = None,
    pipeline_result: Optional[dict] = None,
) -> int:
    init_db()
    with _get_conn() as conn:
        cursor = conn.execute("""
            INSERT INTO feedback (
                created_at, customer_message, predicted_intent, corrected_intent,
                predicted_decision, overridden_decision, reply_accepted,
                edited_reply, notes, thread_id, pipeline_result_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            datetime.utcnow().isoformat(),
            customer_message,
            predicted_intent,
            corrected_intent,
            predicted_decision,
            overridden_decision,
            int(reply_accepted) if reply_accepted is not None else None,
            edited_reply,
            notes,
            thread_id,
            json.dumps(pipeline_result) if pipeline_result else None,
        ))
        conn.commit()
        return cursor.lastrowid


def load_feedback(limit: int = 500) -> list[dict]:
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def feedback_stats() -> dict:
    init_db()
    with _get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        accepted = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE reply_accepted=1"
        ).fetchone()[0]
        rejected = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE reply_accepted=0"
        ).fetchone()[0]
        overrides = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE overridden_decision IS NOT NULL"
        ).fetchone()[0]
        corrections = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE corrected_intent IS NOT NULL"
        ).fetchone()[0]
    return {
        "total": total,
        "accepted": accepted,
        "rejected": rejected,
        "overrides": overrides,
        "corrections": corrections,
    }
