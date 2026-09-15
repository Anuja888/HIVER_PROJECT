"""
Phase 9 — LLM-as-Judge for Reply Quality.

Scores generated replies on a 1-5 rubric across 7 dimensions.
The judge sees: customer message + retrieved evidence + generated reply.
It does NOT see the model's own confidence/self-assessment (to avoid trivial agreement).

Rubric dimensions (from config.yaml):
  relevance, correctness, grounding, helpfulness, completeness, tone,
  no_unsupported_claims

Decision log:
  - Judge model = gpt-4o-mini (cost-efficient, good enough for rubric scoring).
  - Agreement with human ratings is measured on a 30-50 example subset and
    reported as Spearman r + weighted kappa. If agreement is weak, we say so.
  - Mock mode returns deterministic scores based on prompt hash.
"""
from __future__ import annotations

import logging
from pydantic import BaseModel, Field

from src.ai.llm_client import chat_json

log = logging.getLogger(__name__)

RUBRIC_DIMS = [
    "relevance",
    "correctness",
    "grounding",
    "helpfulness",
    "completeness",
    "tone",
    "no_unsupported_claims",
]

JUDGE_SYSTEM = """You are an impartial evaluator of customer support reply quality.

You will score an AI-generated reply on 7 dimensions using a 1-5 scale:
  1=very poor, 2=poor, 3=acceptable, 4=good, 5=excellent

Dimensions:
- relevance: Does the reply address the customer's actual question/problem?
- correctness: Is the information in the reply factually accurate based on the evidence?
- grounding: Are the claims in the reply traceable to the retrieved evidence provided?
- helpfulness: Would this reply actually help the customer resolve their issue?
- completeness: Does the reply cover the key aspects needed to address the issue?
- tone: Is the reply professional, empathetic, and appropriate in tone?
- no_unsupported_claims: Are all claims in the reply supported by evidence? (5=fully supported, 1=many unsupported claims)

You will be given:
1. The customer's message
2. Retrieved historical evidence (real support tweets used to ground the reply)
3. The AI-generated reply to evaluate

Do NOT consider the AI model's own self-assessment or confidence scores.
Base your evaluation only on the customer message, the evidence, and the reply.

Respond with ONLY a JSON object:
{
  "relevance": <1-5>,
  "correctness": <1-5>,
  "grounding": <1-5>,
  "helpfulness": <1-5>,
  "completeness": <1-5>,
  "tone": <1-5>,
  "no_unsupported_claims": <1-5>,
  "overall": <float average>,
  "reasoning": "<2-3 sentence explanation of scores>"
}
"""


class JudgeScore(BaseModel):
    relevance: int = Field(ge=1, le=5)
    correctness: int = Field(ge=1, le=5)
    grounding: int = Field(ge=1, le=5)
    helpfulness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    tone: int = Field(ge=1, le=5)
    no_unsupported_claims: int = Field(ge=1, le=5)
    overall: float = Field(ge=1.0, le=5.0)
    reasoning: str = ""


def judge_reply(
    customer_message: str,
    evidence_items: list[dict],
    generated_reply: str,
) -> JudgeScore:
    """Score a reply using the LLM judge."""
    evidence_text = "\n".join(
        f"  [{i+1}] Amazon support said: \"{ev.get('brand_reply', '')[:250]}\""
        for i, ev in enumerate(evidence_items[:4])
    ) or "  (No retrieved evidence available)"

    user_msg = (
        f"Customer message:\n\"{customer_message}\"\n\n"
        f"Retrieved historical evidence:\n{evidence_text}\n\n"
        f"AI-generated reply to evaluate:\n\"{generated_reply}\""
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    try:
        return chat_json(messages, JudgeScore, temperature=0.0, max_tokens=512)
    except Exception as e:
        log.error("Judge call failed: %s", e)
        # Return middle scores on failure
        return JudgeScore(
            relevance=3, correctness=3, grounding=3,
            helpfulness=3, completeness=3, tone=3,
            no_unsupported_claims=3, overall=3.0,
            reasoning=f"Judge call failed: {e}",
        )


def judge_batch(
    examples: list[dict],  # each has: customer_message, evidence_items, generated_reply
    max_examples: int = 200,
) -> list[JudgeScore]:
    """Run judge on a batch of examples."""
    results = []
    for ex in examples[:max_examples]:
        score = judge_reply(
            customer_message=ex["customer_message"],
            evidence_items=ex.get("evidence_items", []),
            generated_reply=ex["generated_reply"],
        )
        results.append(score)
    return results
