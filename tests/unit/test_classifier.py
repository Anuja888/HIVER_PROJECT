"""Unit tests for the intent classifier."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from src.ai.classifier import KeywordBaseline, LLMClassifier, INTENTS


class TestKeywordBaseline:
    def setup_method(self):
        self.clf = KeywordBaseline()

    def test_returns_classification_result(self):
        result = self.clf.predict("Where is my package?")
        assert result.intent in INTENTS
        assert 0.0 <= result.confidence <= 1.0
        assert result.reason
        assert result.classifier_type == "keyword_baseline"

    def test_delivery_intent(self):
        result = self.clf.predict("My package was supposed to arrive yesterday and it hasn't.")
        assert result.intent == "order_delivery_tracking"

    def test_return_intent(self):
        result = self.clf.predict("I need to return this item and get a refund.")
        assert result.intent == "product_return_refund"

    def test_account_security(self):
        result = self.clf.predict("Someone placed unauthorized orders on my account.")
        assert result.intent == "account_security_compromise"

    def test_prime_intent(self):
        result = self.clf.predict("I was charged for Prime after cancellation.")
        assert result.intent == "prime_subscription_billing"

    def test_no_match_falls_back(self):
        result = self.clf.predict("Hello there")
        assert result.intent == "general_feedback_other"
        assert result.confidence < 0.5

    def test_batch_predict(self):
        messages = ["Where is my order?", "Need a refund", "App is crashing"]
        results = self.clf.predict_batch(messages)
        assert len(results) == 3
        assert all(r.intent in INTENTS for r in results)

    def test_escalation_sensitive_flagged(self):
        result = self.clf.predict("Someone hacked my account")
        assert result.is_escalation_sensitive

    def test_non_sensitive_not_flagged(self):
        result = self.clf.predict("What are the features of this product?")
        assert not result.is_escalation_sensitive


class TestLLMClassifier:
    """Tests with mock mode (no API cost)."""

    def setup_method(self):
        # Uses mock provider (default when no API key set)
        self.clf = LLMClassifier()

    def test_returns_valid_intent(self):
        result = self.clf.predict("My order hasn't arrived yet.")
        assert result.intent in INTENTS
        assert 0.0 <= result.confidence <= 1.0
        assert result.classifier_type in ("llm_fewshot", "llm_fewshot_fallback_to_keyword")

    def test_with_context(self):
        ctx = [{"role": "customer", "text": "I ordered something a week ago."}]
        result = self.clf.predict("Still no delivery confirmation.", context=ctx)
        assert result.intent in INTENTS
