"""
Phase 7 — Historically Grounded Reply Generator.

Generates a support reply grounded strictly in retrieved historical evidence.

Decision log:
  - System prompt explicitly forbids inventing policies, refunds, or promises
    not present in retrieved evidence.
  - If evidence is thin (top_similarity < threshold), reply acknowledges
    uncertainty and recommends escalation.
  - Output distinguishes "historical pattern suggests..." from "current policy".
  - Source tweet_ids are tracked in structured output for traceability.
"""
from __future__ import annotations

import json
from pydantic import BaseModel, Field
from typing import Optional

from src.ai.llm_client import chat
from src.retrieval.index import RetrievalResult

SYSTEM_PROMPT = """You are a helpful Amazon customer support assistant.

RULES — FOLLOW STRICTLY:
1. Only use information from the RETRIEVED EVIDENCE below to formulate your reply.
2. Do NOT invent, assume, or promise: refunds, policies, timelines, discounts, or
   specific procedures unless they appear in the retrieved evidence.
3. If the evidence is thin, contradictory, or irrelevant, say so honestly and
   recommend that the agent escalate to a human specialist.
4. Distinguish clearly between "historical patterns suggest..." and current policy.
   Never assert something as current policy from historical data alone.
5. Keep the tone professional, empathetic, and concise (2–4 sentences).
6. Do not reveal that you are an AI or that you are using retrieved examples.

FORMAT: Respond with a JSON object:
{
  "reply": "<the customer-facing support reply>",
  "resolution_pattern": "<1 sentence summarising the historical pattern>",
  "evidence_sufficient": <true|false>,
  "confidence_note": "<honest 1-sentence note about evidence quality>"
}
"""


class GenerationOutput(BaseModel):
    reply: str
    resolution_pattern: str
    evidence_sufficient: bool
    confidence_note: str
    evidence_items: list[dict] = Field(default_factory=list)


def generate_reply(
    customer_message: str,
    intent: str,
    retrieval_result: RetrievalResult,
    thread_context: Optional[list[dict]] = None,
) -> GenerationOutput:
    """
    Generate a grounded reply using retrieved evidence.

    Args:
        customer_message: The customer's message text
        intent: Predicted intent label
        retrieval_result: Evidence from the retrieval index
        thread_context: Optional previous messages for context
    """
    # Build evidence block
    evidence_lines = []
    for i, ev in enumerate(retrieval_result.items[:5], 1):
        evidence_lines.append(
            f"  [{i}] Customer said: \"{ev.customer_message[:200]}\"\n"
            f"      Amazon replied: \"{ev.brand_reply[:300]}\"\n"
            f"      Similarity: {ev.similarity:.2f} | {ev.relevance_reason}"
        )
    evidence_text = "\n".join(evidence_lines) if evidence_lines else "  (No relevant historical evidence found)"

    # Thread context (last 3 turns)
    context_text = ""
    if thread_context:
        prev = [m for m in thread_context[:-1] if m.get("text", "").strip()][-3:]
        if prev:
            context_text = "\nPrevious turns:\n" + "\n".join(
                f"  [{m['role']}]: {m['text'][:150]}" for m in prev
            )

    user_prompt = f"""RETRIEVED EVIDENCE (from historical Amazon support conversations):
{evidence_text}

CURRENT CUSTOMER MESSAGE (intent: {intent}):
{context_text}
Customer: "{customer_message}"

Generate a support reply following all rules above."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    resp = chat(messages, temperature=0.3, max_tokens=512)
    raw = resp.content.strip()

    # Parse JSON response
    if raw.startswith("```"):
        raw = "\n".join(
            line for line in raw.splitlines()
            if not line.strip().startswith("```")
        ).strip()

    try:
        data = json.loads(raw)
        output = GenerationOutput(
            reply=data.get("reply", raw),
            resolution_pattern=data.get("resolution_pattern", ""),
            evidence_sufficient=data.get("evidence_sufficient", retrieval_result.has_sufficient_evidence),
            confidence_note=data.get("confidence_note", ""),
            evidence_items=retrieval_result.to_dict(),
        )
    except (json.JSONDecodeError, Exception):
        # Graceful fallback: treat entire response as the reply
        output = GenerationOutput(
            reply=raw,
            resolution_pattern="(Could not extract resolution pattern)",
            evidence_sufficient=retrieval_result.has_sufficient_evidence,
            confidence_note="JSON parsing failed; reply shown as-is.",
            evidence_items=retrieval_result.to_dict(),
        )

    return output
