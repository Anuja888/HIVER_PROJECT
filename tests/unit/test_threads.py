"""Unit tests for thread reconstruction."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import pytest
from src.data.threads import reconstruct_threads, _clean_text as clean_text, MAX_CONTEXT_TURNS


def _make_df(rows):
    df = pd.DataFrame(rows)
    df["in_response_to_tweet_id"] = pd.to_numeric(
        df["in_response_to_tweet_id"], errors="coerce"
    )
    return df


class TestCleanText:
    def test_strips_at_mentions(self):
        result = clean_text("@AmazonHelp my order is late", is_customer=True)
        assert "@AmazonHelp" not in result
        assert "late" in result

    def test_keeps_at_mentions_in_brand(self):
        result = clean_text("@customer123 please DM us", is_customer=False)
        assert "@customer123" in result

    def test_replaces_url(self):
        result = clean_text("check https://amzn.to/xyz for details", is_customer=True)
        assert "https://" not in result
        assert "[URL]" in result

    def test_handles_none(self):
        assert clean_text(None, is_customer=True) == ""

    def test_collapses_whitespace(self):
        result = clean_text("hello   world   there", is_customer=True)
        assert "  " not in result


class TestReconstructThreads:
    def test_basic_reconstruction(self):
        rows = [
            {"tweet_id": 1, "author_id": "user1", "inbound": True,
             "created_at": "Mon Oct 30 10:00:00 +0000 2017",
             "text": "My order is late", "in_response_to_tweet_id": None},
            {"tweet_id": 2, "author_id": "AmazonHelp", "inbound": False,
             "created_at": "Mon Oct 30 10:05:00 +0000 2017",
             "text": "We are looking into it", "in_response_to_tweet_id": 1},
        ]
        df = _make_df(rows)
        threads = reconstruct_threads(df, "AmazonHelp", verbose=False)
        assert len(threads) == 1
        thread = threads[0]
        assert any(m.role == "customer" for m in thread.messages)
        assert any(m.role == "brand" for m in thread.messages)

    def test_no_customer_message_dropped(self):
        rows = [
            {"tweet_id": 1, "author_id": "AmazonHelp", "inbound": False,
             "created_at": "Mon Oct 30 10:00:00 +0000 2017",
             "text": "We are happy to help", "in_response_to_tweet_id": None},
        ]
        df = _make_df(rows)
        threads = reconstruct_threads(df, "AmazonHelp", verbose=False)
        assert len(threads) == 0  # no customer message → dropped

    def test_truncation(self):
        rows = [
            {
                "tweet_id": i,
                "author_id": "user1" if i % 2 == 1 else "AmazonHelp",
                "inbound": i % 2 == 1,
                "created_at": f"Mon Oct 30 10:{i:02d}:00 +0000 2017",
                "text": f"Message {i}",
                "in_response_to_tweet_id": i - 1 if i > 1 else None,
            }
            for i in range(1, MAX_CONTEXT_TURNS + 5)  # more than MAX
        ]
        df = _make_df(rows)
        threads = reconstruct_threads(df, "AmazonHelp", verbose=False)
        if threads:
            assert threads[0].n_turns <= MAX_CONTEXT_TURNS
            assert threads[0].truncated
