"""
Phase 8 — Escalation Router.

Combines multiple signals to decide AUTO_HANDLE vs. ESCALATE with a
specific, human-legible reason.

Signals combined:
  1. Intent confidence (uncalibrated — see note below)
  2. Retrieval quality (top similarity score)
  3. Grounding validator score
  4. Intent escalation sensitivity flag (from taxonomy)
  5. Multi-intent / ambiguity flag (needs_context from classifier)

Decision log:
  - Thresholds are initial values from config.yaml. Final values are selected
    by sweeping on the validation split in the eval harness (scripts/run_eval.py).
  - Escalation sensitivity is a hard gate: even high-confidence account-security
    intents are escalated (false-auto-handle risk is too high there).
  - The reason string is ALWAYS specific and operational — never "AI confidence
    is low." This is an explicit requirement from the assignment rubric.
  - confidence in RouterOutput is NOT a calibrated probability. UI copy says so.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from src.config.settings import settings
from src.ai.classifier import ClassificationResult
from src.ai.validator import ValidationResult
from src.retrieval.index import RetrievalResult


class RouterOutput(BaseModel):
    decision: str = Field(..., description="'AUTO_HANDLE' or 'ESCALATE'")
    reason: str = Field(..., description="Specific operational reason")
    risk_flags: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0,
                              description="Uncalibrated system signal — not a probability")


def route(
    classification: ClassificationResult,
    retrieval: RetrievalResult,
    validation: ValidationResult,
    intent_conf_threshold: float | None = None,
    retrieval_sim_threshold: float | None = None,
    grounding_threshold: float | None = None,
) -> RouterOutput:
    """
    Decide whether to auto-handle or escalate this support request.

    Args:
        classification: Intent classification result
        retrieval: Evidence retrieval result
        validation: Grounding validation result
        *_threshold: Override config thresholds (used during validation sweep)
    """
    conf_th = intent_conf_threshold or settings.intent_confidence_threshold
    sim_th  = retrieval_sim_threshold or settings.retrieval_similarity_threshold

    risk_flags: list[str] = []
    escalate_reasons: list[str] = []

    # ---- Gate 1: Escalation-sensitive intent ----
    # Account security and compromise are always escalated regardless of confidence.
    if classification.intent == "account_security_compromise":
        escalate_reasons.append(
            "Intent 'account_security_compromise' is always escalated "
            "due to high potential for financial/account harm."
        )
        risk_flags.append("high_risk_intent")

    # ---- Gate 2: Intent confidence ----
    if classification.confidence < conf_th:
        escalate_reasons.append(
            f"Intent confidence ({classification.confidence:.2f}) is below threshold "
            f"({conf_th:.2f}) — classification is uncertain."
        )
        risk_flags.append("low_intent_confidence")

    # ---- Gate 3: Needs more context ----
    if classification.needs_context:
        escalate_reasons.append(
            "Classifier flagged this message as ambiguous — more context is needed."
        )
        risk_flags.append("ambiguous_message")

    # ---- Gate 4: Retrieval quality ----
    if retrieval.top_similarity < sim_th:
        escalate_reasons.append(
            f"No historical resolution with similarity above threshold "
            f"({retrieval.top_similarity:.2f} < {sim_th:.2f}) was found for this issue type."
        )
        risk_flags.append("weak_retrieval")

    # ---- Gate 5: Grounding validation ----
    if not validation.passed:
        escalate_reasons.append(
            f"Grounding validator flagged potential unsupported claims "
            f"(score={validation.score:.2f}). Human review required."
        )
        risk_flags.append("grounding_failed")

    if validation.rule_flags:
        for flag in validation.rule_flags:
            risk_flags.append(f"rule_flag: {flag[:60]}")

    # ---- Final decision ----
    if escalate_reasons:
        decision = "ESCALATE"
        reason = escalate_reasons[0]  # primary reason
    else:
        decision = "AUTO_HANDLE"
        reason = (
            f"High-confidence classification ({classification.intent}, "
            f"conf={classification.confidence:.2f}), strong retrieval match "
            f"(sim={retrieval.top_similarity:.2f}), and grounding check passed."
        )

    # Combined confidence signal (geometric mean of available signals)
    combined_conf = (
        classification.confidence *
        min(1.0, retrieval.top_similarity / max(sim_th, 0.01)) *
        validation.score
    ) ** (1/3)

    return RouterOutput(
        decision=decision,
        reason=reason,
        risk_flags=list(set(risk_flags)),
        confidence=round(combined_conf, 3),
    )
