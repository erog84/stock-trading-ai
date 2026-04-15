"""Tests for sentiment analysis."""

import pytest

from src.data.sentiment import SentimentScorer, score_text, score_detailed


class TestSentimentScorer:
    def setup_method(self):
        self.scorer = SentimentScorer()

    def test_positive_text(self):
        text = "The company achieved record growth and excellent profit with strong earnings"
        score = self.scorer.score_text(text)
        assert score > 0

    def test_negative_text(self):
        text = "The company faces severe losses and declining revenue with uncertainty"
        score = self.scorer.score_text(text)
        assert score < 0

    def test_neutral_text(self):
        text = "The quick brown fox jumps over the lazy dog"
        score = self.scorer.score_text(text)
        assert score == 0.0

    def test_empty_text(self):
        assert self.scorer.score_text("") == 0.0

    def test_detailed_output(self):
        text = "Strong growth and excellent profit despite some risk and uncertainty"
        result = self.scorer.score_detailed(text)
        assert "sentiment_score" in result
        assert "positive_count" in result
        assert "negative_count" in result
        assert result["positive_count"] > 0
        assert result["total_words"] > 0

    def test_score_range(self):
        """Score should always be in [-1, 1]."""
        texts = [
            "great excellent perfect amazing wonderful best profit growth",
            "terrible awful horrible loss decline failure risk crisis",
            "the cat sat on the mat",
        ]
        for text in texts:
            score = self.scorer.score_text(text)
            assert -1 <= score <= 1


class TestConvenienceFunctions:
    def test_score_text(self):
        assert isinstance(score_text("growth and profit"), float)

    def test_score_detailed(self):
        result = score_detailed("growth and risk")
        assert isinstance(result, dict)
