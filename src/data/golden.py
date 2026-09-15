"""
Phase 4 — Golden Evaluation Set Sampling.

Samples 200 candidate examples from the TEST split (never train/val) using
stratified sampling to ensure coverage of all intent buckets.

The sampled pool is written to data/gold/candidate_pool.jsonl for human
labelling. The final golden set with human labels is stored in
data/gold/golden_set.jsonl with labelled_by="human".

IMPORTANT: This module NEVER auto-fills gold labels. The labelling tool
           (src/ui/label_mode.py) only lets the human write to golden_set.jsonl.

Sampling strategy (documented assumption):
  - Source: test split only (prevents any data leakage into train/val)
  - Stratification: 9 buckets (8 intents + other) × deliberate over-sampling of
    rare/ambiguous examples
  - Target: 200 examples (~22 per intent, adjusted for frequency)
  - Seed: settings.sample_seed for reproducibility
  - Ambiguous slice: 20% of pool are examples where top-2 intents are close
    (heuristic: message is short or contains multiple topics)

Decision log:
  - Using test split as source ensures the golden set is never contaminated by
    training data. The evaluator evaluates on the same test split, but the golden
    labels are independent of any model output.
  - Stratification prevents the common failure of a golden set dominated by the
    single most frequent intent (in Amazon support: order_delivery_tracking).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config.settings import settings

INTENTS = [
    "order_delivery_tracking",
    "product_return_refund",
    "account_access_login",
    "account_security_compromise",
    "product_question_usage",
    "technical_bug_malfunction",
    "prime_subscription_billing",
    "seller_third_party_issue",
    "general_feedback_other",
]

# Keyword heuristics for rough initial stratification
# (NOT used as labels — only to ensure all topic areas are sampled)
INTENT_KEYWORDS: dict[str, list[str]] = {
    "order_delivery_tracking": ["order", "track", "deliver", "ship", "package", "arrive", "dispatch"],
    "product_return_refund": ["return", "refund", "charge", "damaged", "wrong item", "money back"],
    "account_access_login": ["login", "log in", "password", "locked", "otp", "verify", "sign in"],
    "account_security_compromise": ["unauthorized", "fraud", "hack", "someone else", "strange order"],
    "product_question_usage": ["how do i", "does it", "compatible", "setup", "configure", "work with"],
    "technical_bug_malfunction": ["error", "crash", "bug", "broken", "not working", "issue", "glitch"],
    "prime_subscription_billing": ["prime", "subscription", "cancel", "membership", "renewal"],
    "seller_third_party_issue": ["seller", "third party", "marketplace", "counterfeit", "listing"],
    "general_feedback_other": [],  # catch-all
}


def _guess_intent(text: str) -> str:
    """Rough heuristic guess — used for stratification ONLY, not as a label."""
    t = text.lower()
    scores = {intent: 0 for intent in INTENTS}
    for intent, kws in INTENT_KEYWORDS.items():
        scores[intent] = sum(1 for kw in kws if kw in t)
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "general_feedback_other"


def _is_ambiguous(text: str) -> bool:
    """Heuristic: short messages or multi-topic are likely ambiguous."""
    words = text.split()
    if len(words) < 6:
        return True
    # Multiple intent keywords from different categories
    hit_intents = set()
    t = text.lower()
    for intent, kws in INTENT_KEYWORDS.items():
        if any(kw in t for kw in kws):
            hit_intents.add(intent)
    return len(hit_intents) >= 2


def sample_candidate_pool(brand: str, n: int = 200, seed: int | None = None) -> list[dict]:
    """
    Sample n candidates from the test split for human labelling.
    Returns list of dicts with thread context, customer message, and metadata.
    """
    seed = seed or settings.sample_seed
    rng = np.random.default_rng(seed)

    test_path = settings.cache_dir / f"{brand}_test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError("Run data pipeline first.")

    threads: list[dict] = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            threads.append(json.loads(line))

    # Build candidate records
    candidates: list[dict] = []
    for t in threads:
        cust_msg = next(
            (m for m in t["messages"] if m["role"] == "customer" and m["text"].strip()),
            None
        )
        if not cust_msg:
            continue
        text = cust_msg["text"].strip()
        guessed = _guess_intent(text)
        ambig = _is_ambiguous(text)
        candidates.append({
            "thread_id": t["thread_id"],
            "customer_message": text,
            "customer_tweet_id": cust_msg["tweet_id"],
            "thread_context": t["messages"],
            "n_turns": t["n_turns"],
            "guessed_intent_heuristic": guessed,
            "is_ambiguous_heuristic": ambig,
            "labelled_by": None,
            "gold_intent": None,
            "gold_escalate": None,
            "gold_notes": None,
        })

    # Stratified sample: ~equal per intent + oversample ambiguous
    per_intent_target = max(1, (n * 80 // 100) // len(INTENTS))
    ambig_target = n * 20 // 100

    by_intent: dict[str, list[int]] = {i: [] for i in INTENTS}
    ambig_pool: list[int] = []

    for idx, c in enumerate(candidates):
        if c["is_ambiguous_heuristic"]:
            ambig_pool.append(idx)
        by_intent[c["guessed_intent_heuristic"]].append(idx)

    chosen: set[int] = set()

    # Pick per-intent
    for intent in INTENTS:
        pool = [i for i in by_intent[intent] if i not in chosen]
        take = min(per_intent_target, len(pool))
        if take > 0:
            picked = rng.choice(pool, size=take, replace=False).tolist()
            chosen.update(picked)

    # Top up with ambiguous examples
    ambig_remaining = [i for i in ambig_pool if i not in chosen]
    take_ambig = min(ambig_target, len(ambig_remaining))
    if take_ambig > 0:
        picked = rng.choice(ambig_remaining, size=take_ambig, replace=False).tolist()
        chosen.update(picked)

    # Top up to n if needed
    remaining = [i for i in range(len(candidates)) if i not in chosen]
    if len(chosen) < n and remaining:
        extra = rng.choice(remaining,
                           size=min(n - len(chosen), len(remaining)),
                           replace=False).tolist()
        chosen.update(extra)

    pool = [candidates[i] for i in sorted(chosen)][:n]
    return pool


def save_candidate_pool(pool: list[dict], brand: str) -> Path:
    settings.gold_dir.mkdir(parents=True, exist_ok=True)
    path = settings.gold_dir / "candidate_pool.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in pool:
            f.write(json.dumps(rec) + "\n")
    print(f"[golden] Candidate pool ({len(pool)} examples) -> {path}")
    return path


def load_candidate_pool() -> list[dict]:
    path = settings.gold_dir / "candidate_pool.jsonl"
    if not path.exists():
        raise FileNotFoundError("Candidate pool not found. Run label_mode first.")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_golden_set() -> list[dict]:
    """Load ONLY human-labelled golden examples."""
    path = settings.gold_dir / "golden_set.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("labelled_by") == "human":
                records.append(rec)
    return records


def save_golden_label(record: dict) -> None:
    """Append a single human-labelled record to golden_set.jsonl."""
    assert record.get("labelled_by") == "human", \
        "Only records with labelled_by='human' may be written to golden_set.jsonl"
    path = settings.gold_dir / "golden_set.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
