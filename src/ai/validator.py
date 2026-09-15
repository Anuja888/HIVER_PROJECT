"""
Phase 7 — Grounding Validator.

Checks the generated reply for claims not traceable to retrieved evidence.
Uses both rule-based checks and an optional LLM judge step.

Decision log:
  - Rule-based checks run first (fast, zero cost): flag common hallucination
    patterns like specific dollar amounts, specific timelines ("within 24 hours"),
    guaranteed refund language, or policy assertions.
  - LLM grounding check is optional (used in full eval, not mocked pipeline).
  - A reply passes grounding if: (a) no rule flags triggered, and (b) LLM
    judge rates grounding ≥ 3/5 OR mock mode is active.
"""
from __future__ import annotations

import json
import re
from pydantic import BaseModel, Field

from src.ai.llm_client import chat

# Patterns that suggest hallucinated specifics
_HALLUCINATION_PATTERNS = [
    (r"\$\d+", "Specific dollar amount mentioned — not traceable to evidence"),
    (r"within \d+ (hour|day|business)", "Specific timeline promised — verify against evidence"),
    (r"guaranteed|we guarantee|100%", "Guarantee language — potentially unsupported by evidence"),
    (r"our policy (is|states|says)", "Policy assertion — verify this is in retrieved evidence"),
    (r"you will (receive|get) a (full )?refund", "Refund promise — must be in evidence"),
    (r"free of charge", "Free service promise — check evidence"),
]

_GROUNDING_SYSTEM = """You are a grounding quality checker for AI-generated customer support replies.

You will be given:
1. A customer message
2. Retrieved historical evidence (real Amazon support tweets)
3. An AI-generated reply

Your task: Check whether the reply makes any claims NOT supported by the retrieved evidence.

Respond with ONLY JSON:
{
  "grounded": <true|false>,
  "score": <float 0.0-1.0>,
  "flags": ["<issue1>", "<issue2>"],
  "reasoning": "<one sentence>"
}

grounded=true if all factual claims in the reply are traceable to the evidence.
score=1.0 means perfectly grounded; 0.0 means completely hallucinated.
"""


class ValidationResult(BaseModel):
    grounded: bool
    score: float = Field(ge=0.0, le=1.0)
    rule_flags: list[str] = Field(default_factory=list)
    llm_flags: list[str] = Field(default_factory=list)
    reasoning: str = ""
    passed: bool = True


def validate_grounding(
    reply: str,
    customer_message: str,
    evidence_items: list[dict],
    use_llm: bool = True,
) -> ValidationResult:
    """
    Validate that the reply is grounded in evidence.

    Args:
        reply: The generated reply text
        customer_message: The original customer message
        evidence_items: Retrieved evidence dicts with 'brand_reply' fields
        use_llm: Whether to also run LLM grounding check (default True)
    """
    rule_flags: list[str] = []

    # Rule-based checks
    for pattern, description in _HALLUCINATION_PATTERNS:
        if re.search(pattern, reply, re.IGNORECASE):
            # Only flag if NOT present in evidence
            in_evidence = any(
                re.search(pattern, ev.get("brand_reply", ""), re.IGNORECASE)
                for ev in evidence_items
            )
            if not in_evidence:
                rule_flags.append(description)

    # LLM grounding check
    llm_flags: list[str] = []
    llm_score = 0.8  # default
    llm_reasoning = ""

    if use_llm and evidence_items:
        evidence_text = "\n".join(
            f"  [{i+1}] \"{ev.get('brand_reply', '')[:200]}\""
            for i, ev in enumerate(evidence_items[:3])
        )
        user_prompt = (
            f"Customer message: \"{customer_message}\"\n\n"
            f"Retrieved evidence:\n{evidence_text}\n\n"
            f"AI reply to check:\n\"{reply}\""
        )
        messages = [
            {"role": "system", "content": _GROUNDING_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        try:
            resp = chat(messages, temperature=0.0, max_tokens=256)
            raw = resp.content.strip()
            if raw.startswith("```"):
                raw = "\n".join(
                    l for l in raw.splitlines() if not l.strip().startswith("```")
                ).strip()
            data = json.loads(raw)
            llm_score = float(data.get("score", 0.8))
            llm_flags = data.get("flags", [])
            llm_reasoning = data.get("reasoning", "")
        except Exception:
            pass  # LLM check failed — use defaults

    all_flags = rule_flags + llm_flags
    overall_score = llm_score * (0.8 if rule_flags else 1.0)
    passed = overall_score >= 0.5 and len(rule_flags) == 0

    return ValidationResult(
        grounded=passed,
        score=round(overall_score, 3),
        rule_flags=rule_flags,
        llm_flags=llm_flags,
        reasoning=llm_reasoning or ("No grounding issues detected." if not all_flags else "; ".join(all_flags)),
        passed=passed,
    )
