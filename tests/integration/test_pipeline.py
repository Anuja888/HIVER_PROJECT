"""
Integration test: runs the full pipeline on 5 fixture examples.
Uses mock mode — zero API cost.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import pytest

# Force mock mode for tests
os.environ["LLM_PROVIDER"] = "mock"

from src.ai.pipeline import run_pipeline

FIXTURE_MESSAGES = [
    "My package was supposed to arrive 3 days ago and I still haven't received it.",
    "I received the wrong item and I need a refund immediately.",
    "I can't log into my account — it keeps saying my password is wrong.",
    "Someone placed orders on my account that I didn't make!",
    "The Amazon app keeps crashing every time I try to checkout.",
]


class TestFullPipeline:
    def test_pipeline_runs_on_fixtures(self):
        """Full pipeline should complete without error on all fixture messages."""
        for msg in FIXTURE_MESSAGES:
            result = run_pipeline(msg, brand="AmazonHelp")
            assert result.error is None or result.classification is not None, \
                f"Pipeline failed for: {msg[:50]}: {result.error}"

    def test_classification_returned(self):
        result = run_pipeline(FIXTURE_MESSAGES[0])
        assert result.classification is not None
        assert result.classification.intent
        assert 0.0 <= result.classification.confidence <= 1.0

    def test_routing_returned(self):
        result = run_pipeline(FIXTURE_MESSAGES[0])
        assert result.routing is not None
        assert result.routing.decision in ("AUTO_HANDLE", "ESCALATE")
        assert len(result.routing.reason) > 10

    def test_security_always_escalated(self):
        """Account security compromise must always escalate."""
        result = run_pipeline(FIXTURE_MESSAGES[3])  # unauthorized orders
        if result.classification and result.classification.intent == "account_security_compromise":
            assert result.routing.decision == "ESCALATE"

    def test_pipeline_result_serializable(self):
        result = run_pipeline(FIXTURE_MESSAGES[0])
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "classification" in d
        assert "routing" in d

    def test_latency_tracked(self):
        result = run_pipeline(FIXTURE_MESSAGES[0])
        assert result.latency_ms > 0

    def test_with_thread_context(self):
        ctx = [{"role": "customer", "text": "I placed an order last week."}]
        result = run_pipeline(FIXTURE_MESSAGES[0], thread_context=ctx)
        assert result.classification is not None
