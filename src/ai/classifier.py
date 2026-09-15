"""
Phase 5/6 — Intent Classifier.

Three implementations:
  (a) KeywordBaseline — rule-based, no LLM
  (b) LLMClassifier  — few-shot LLM with structured JSON output
  (c) EmbeddingKNN   — optional, used in simple baseline evaluation

All return ClassificationResult (pydantic model).

Decision log:
  - confidence is explicitly labeled as an UNCALIBRATED model signal in all
    UI copy and eval reports. LLM self-reported confidence has no reliable
    probability interpretation without calibration.
  - Few-shot examples are drawn from the training split, never from val/test.
  - Threshold for downstream escalation/routing is selected by sweeping on the
    validation split (see scripts/run_eval.py), not hardcoded here.
"""
from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from src.config.settings import settings
from src.ai.llm_client import chat_json

log = logging.getLogger(__name__)

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

ESCALATION_SENSITIVE = {
    "product_return_refund",
    "account_access_login",
    "account_security_compromise",
    "prime_subscription_billing",
}


class ClassificationResult(BaseModel):
    intent: str = Field(..., description="Predicted intent label")
    confidence: float = Field(..., ge=0.0, le=1.0,
                              description="Uncalibrated model signal, NOT a probability")
    reason: str = Field(..., description="One-sentence explanation")
    needs_context: bool = Field(False, description="True if more context would change prediction")
    is_escalation_sensitive: bool = Field(False)
    classifier_type: str = Field("unknown")


# ---------------------------------------------------------------------------
# (a) Keyword Baseline
# ---------------------------------------------------------------------------

_KW_MAP: dict[str, list[str]] = {
    "order_delivery_tracking": [
        "order", "track", "deliver", "ship", "package", "arrive", "dispatch",
        "where is", "status", "expected", "eta", "late", "delay"
    ],
    "product_return_refund": [
        "return", "refund", "charge", "damaged", "wrong item", "money back",
        "reimburs", "cancel order", "broken", "faulty", "replace"
    ],
    "account_access_login": [
        "login", "log in", "log-in", "password", "locked", "otp", "verify",
        "sign in", "cant access", "can't access", "authentication"
    ],
    "account_security_compromise": [
        "unauthorized", "fraud", "hack", "someone else", "strange order",
        "suspicious", "not me", "account breach", "stolen", "scam",
        "unauthorized order", "didn't place", "didn't make", "not mine",
        "someone placed", "someone bought",
    ],
    "product_question_usage": [
        "how do i", "does it work", "compatible", "setup", "configure",
        "how to use", "specification", "features", "what is", "can it"
    ],
    "technical_bug_malfunction": [
        "error", "crash", "bug", "broken", "not working", "glitch", "fail",
        "issue with app", "stopped working", "error code", "freezing"
    ],
    "prime_subscription_billing": [
        "prime", "subscription", "cancel", "membership", "renewal", "auto renew",
        "prime charge", "prime video", "prime benefit"
    ],
    "seller_third_party_issue": [
        "seller", "third party", "marketplace", "counterfeit", "listing",
        "third-party", "vendor", "a-to-z", "atoz"
    ],
}


class KeywordBaseline:
    """
    Trivial keyword-matching baseline.
    No LLM, no embeddings — counts keyword hits per intent.
    Falls back to 'general_feedback_other' when no keyword matches.
    """

    def predict(self, message: str) -> ClassificationResult:
        t = message.lower()
        scores = {intent: 0 for intent in INTENTS}
        for intent, kws in _KW_MAP.items():
            for kw in kws:
                if kw in t:
                    scores[intent] += 1

        best = max(scores, key=lambda k: scores[k])
        best_score = scores[best]

        if best_score == 0:
            intent = "general_feedback_other"
            conf = 0.30
            reason = "No keyword matches found."
        else:
            intent = best
            total = sum(scores.values())
            conf = min(0.90, best_score / total if total > 0 else 0.30)
            reason = f"Matched {best_score} keyword(s) for '{intent}'."

        return ClassificationResult(
            intent=intent,
            confidence=conf,
            reason=reason,
            needs_context=conf < 0.45,
            is_escalation_sensitive=intent in ESCALATION_SENSITIVE,
            classifier_type="keyword_baseline",
        )

    def predict_batch(self, messages: list[str]) -> list[ClassificationResult]:
        return [self.predict(m) for m in messages]


# ---------------------------------------------------------------------------
# (b) LLM Few-Shot Classifier
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLES: list[dict] = [
    {"message": "Where is my package? It was supposed to arrive 3 days ago.",
     "intent": "order_delivery_tracking", "confidence": 0.92,
     "reason": "Customer asks about delivery status of existing order."},
    {"message": "I received a completely different item from what I ordered. I need a refund.",
     "intent": "product_return_refund", "confidence": 0.95,
     "reason": "Wrong item received, refund requested."},
    {"message": "I can't log into my account. It keeps saying my password is wrong.",
     "intent": "account_access_login", "confidence": 0.90,
     "reason": "Customer locked out of account."},
    {"message": "Someone placed 3 orders on my account that I didn't make!",
     "intent": "account_security_compromise", "confidence": 0.95,
     "reason": "Unauthorized orders — account compromise suspected."},
    {"message": "Does this Echo Dot work with Spotify?",
     "intent": "product_question_usage", "confidence": 0.85,
     "reason": "Compatibility question about a product."},
    {"message": "The Amazon app keeps crashing every time I try to pay.",
     "intent": "technical_bug_malfunction", "confidence": 0.88,
     "reason": "App crash during checkout — technical bug."},
    {"message": "I cancelled my Prime trial a week ago but was still charged.",
     "intent": "prime_subscription_billing", "confidence": 0.92,
     "reason": "Unexpected charge after Prime cancellation."},
    {"message": "The seller on your site sent me a fake product and isn't responding.",
     "intent": "seller_third_party_issue", "confidence": 0.87,
     "reason": "Third-party seller complaint with counterfeit item."},
]

_SYSTEM_PROMPT = """You are an intent classification system for Amazon customer support.

Classify the customer message into EXACTLY ONE of these intents:
{intent_list}

Respond with ONLY a JSON object (no other text) with this exact schema:
{{
  "intent": "<one of the intents above>",
  "confidence": <float 0.0-1.0>,
  "reason": "<one sentence explaining why>",
  "needs_context": <true|false>
}}

confidence is your estimated certainty (0=very uncertain, 1=very certain).
needs_context = true if the surrounding thread context would significantly change your prediction.
"""


class LLMClassifier:
    """
    LLM few-shot classifier. Returns structured ClassificationResult.
    Supports mock mode (deterministic, zero cost).
    """

    def __init__(self, n_examples: int | None = None):
        self.n_examples = n_examples or settings.yaml_config.get(
            "classifier", {}
        ).get("max_examples_in_prompt", 6)

    def _build_messages(self, message: str, context: list[dict] | None = None) -> list[dict]:
        intent_list = "\n".join(f"  - {i}" for i in INTENTS)
        system = _SYSTEM_PROMPT.format(intent_list=intent_list)

        few_shot_text = "\n".join(
            f"Customer: \"{ex['message']}\"\n"
            f"Result: {json.dumps({'intent': ex['intent'], 'confidence': ex['confidence'], 'reason': ex['reason'], 'needs_context': False})}"
            for ex in _FEW_SHOT_EXAMPLES[:self.n_examples]
        )

        context_text = ""
        if context:
            prev = [m for m in context[:-1] if m.get("text", "").strip()][-3:]
            if prev:
                context_text = "\nPrevious conversation context:\n" + "\n".join(
                    f"  [{m['role']}]: {m['text'][:200]}" for m in prev
                )

        user_msg = (
            f"Here are some examples:\n\n{few_shot_text}\n\n"
            f"Now classify this customer message:"
            f"{context_text}\n\n"
            f"Customer message: \"{message}\""
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]

    def predict(self, message: str,
                context: list[dict] | None = None) -> ClassificationResult:
        messages = self._build_messages(message, context)
        try:
            result = chat_json(messages, _LLMClassificationOutput)
            intent = result.intent if result.intent in INTENTS else "general_feedback_other"
            return ClassificationResult(
                intent=intent,
                confidence=float(result.confidence),
                reason=result.reason,
                needs_context=result.needs_context,
                is_escalation_sensitive=intent in ESCALATION_SENSITIVE,
                classifier_type="llm_fewshot",
            )
        except Exception as e:
            log.error("LLM classifier error: %s", e)
            # Fallback to keyword baseline
            kb = KeywordBaseline()
            res = kb.predict(message)
            res.classifier_type = "llm_fewshot_fallback_to_keyword"
            return res

    def predict_batch(self, messages: list[str]) -> list[ClassificationResult]:
        return [self.predict(m) for m in messages]


class _LLMClassificationOutput(BaseModel):
    intent: str
    confidence: float
    reason: str
    needs_context: bool = False


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def get_classifier(classifier_type: str = "auto") -> "KeywordBaseline | LLMClassifier":
    if classifier_type == "auto":
        classifier_type = settings.yaml_config.get("classifier", {}).get("type", "llm_fewshot")
    if classifier_type == "keyword_baseline":
        return KeywordBaseline()
    return LLMClassifier()
