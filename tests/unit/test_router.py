"""Unit tests for the escalation router."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from src.ai.classifier import ClassificationResult
from src.ai.validator import ValidationResult
from src.ai.router import route, RouterOutput
from src.retrieval.index import RetrievalResult, EvidenceItem


def _make_clf(intent="order_delivery_tracking", conf=0.85, needs_context=False):
    return ClassificationResult(
        intent=intent, confidence=conf, reason="test",
        needs_context=needs_context,
        is_escalation_sensitive=intent in {
            "product_return_refund", "account_access_login",
            "account_security_compromise", "prime_subscription_billing"
        },
        classifier_type="test",
    )


def _make_retrieval(sim=0.75):
    items = [EvidenceItem(
        thread_id=1, customer_message="test", brand_reply="test reply",
        similarity=sim, relevance_reason="test"
    )]
    ret = RetrievalResult(items=items, query="test")
    return ret


def _make_validation(passed=True, score=0.85):
    return ValidationResult(grounded=passed, score=score, passed=passed)


class TestRouter:
    def test_auto_handle_happy_path(self):
        result = route(
            _make_clf(conf=0.85),
            _make_retrieval(sim=0.75),
            _make_validation(passed=True, score=0.85),
        )
        assert result.decision == "AUTO_HANDLE"
        assert result.confidence > 0

    def test_escalate_low_confidence(self):
        result = route(
            _make_clf(conf=0.30),  # below threshold
            _make_retrieval(sim=0.75),
            _make_validation(passed=True),
        )
        assert result.decision == "ESCALATE"
        assert "confidence" in result.reason.lower() or "threshold" in result.reason.lower()

    def test_escalate_weak_retrieval(self):
        result = route(
            _make_clf(conf=0.85),
            _make_retrieval(sim=0.10),  # very low
            _make_validation(passed=True),
        )
        assert result.decision == "ESCALATE"
        assert "retrieval" in result.reason.lower() or "similarity" in result.reason.lower()

    def test_always_escalate_security(self):
        result = route(
            _make_clf(intent="account_security_compromise", conf=0.99),
            _make_retrieval(sim=0.99),
            _make_validation(passed=True),
        )
        assert result.decision == "ESCALATE"
        assert "security" in result.reason.lower() or "security_compromise" in result.reason.lower()

    def test_escalate_grounding_failed(self):
        result = route(
            _make_clf(conf=0.85),
            _make_retrieval(sim=0.75),
            _make_validation(passed=False, score=0.2),
        )
        assert result.decision == "ESCALATE"

    def test_reason_is_specific(self):
        """Reason must never be just 'AI confidence is low'."""
        result = route(
            _make_clf(conf=0.30),
            _make_retrieval(sim=0.75),
            _make_validation(passed=True),
        )
        assert len(result.reason) > 20
        assert result.reason.lower() != "ai confidence is low"
