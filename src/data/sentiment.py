"""Financial sentiment analysis using Loughran-McDonald dictionary.

The Loughran-McDonald (LM) dictionary is specifically designed for financial
text analysis, unlike general-purpose sentiment tools. It categorizes words
into: positive, negative, uncertainty, litigious, constraining, superfluous.
"""

import re
from pathlib import Path
from typing import Optional
import csv

from src.utils.logger import logger

# Compact built-in word lists (subset of LM dictionary for no-download use)
# Full dictionary can be loaded from CSV if available
_POSITIVE_WORDS = {
    "achieve", "achievement", "advance", "advantage", "benefit", "best",
    "better", "boost", "breakthrough", "confident", "creative", "deliver",
    "dividend", "earn", "earnings", "effective", "efficiency", "enhance",
    "exceed", "excellent", "exceptional", "favorable", "gain", "good",
    "great", "grow", "growth", "high", "highest", "improve", "improvement",
    "increase", "innovation", "innovative", "leadership", "momentum",
    "opportunity", "optimal", "outperform", "positive", "premier", "profit",
    "profitable", "progress", "prosper", "record", "recovery", "reward",
    "strong", "stronger", "strongest", "succeed", "success", "successful",
    "superior", "surpass", "upside", "upturn", "win",
}

_NEGATIVE_WORDS = {
    "abandon", "adverse", "against", "caution", "challenge", "close",
    "closing", "concern", "costly", "crisis", "critical", "cut", "damage",
    "danger", "decline", "decrease", "default", "deficit", "delay",
    "deteriorate", "difficult", "difficulty", "diminish", "disappoint",
    "discontinue", "disruption", "doubt", "downturn", "drop", "failure",
    "fall", "fear", "foreclose", "fraud", "halt", "harm", "impair",
    "impairment", "inability", "inadequate", "lawsuit", "layoff", "liability",
    "liquidate", "litigation", "lose", "loss", "losses", "negative",
    "obstacle", "penalty", "plunge", "poor", "problem", "recall", "recession",
    "restructure", "restructuring", "risk", "risky", "severe", "shortage",
    "shrink", "slump", "stagnant", "sue", "sued", "terminate", "threat",
    "trouble", "uncertain", "uncertainty", "underperform", "unfavorable",
    "volatile", "volatility", "vulnerability", "warn", "warning", "weak",
    "weaken", "weakness", "worsen", "worst", "writedown", "writeoff",
}

_UNCERTAINTY_WORDS = {
    "almost", "ambiguity", "ambiguous", "appear", "approximate",
    "approximately", "assume", "assumption", "believe", "conditional",
    "contingency", "contingent", "could", "depend", "depends",
    "doubt", "estimate", "estimated", "expect", "fluctuate", "indefinite",
    "indicate", "likelihood", "may", "maybe", "might", "nearly",
    "occasionally", "pending", "perhaps", "possible", "possibly",
    "predict", "prediction", "preliminary", "presume", "probable",
    "probably", "project", "projected", "risk", "roughly", "seem",
    "seldom", "sometimes", "somewhat", "suggest", "tentative",
    "uncertain", "uncertainty", "unclear", "undetermined", "unforeseeable",
    "unknown", "unlikely", "unpredictable", "unquantifiable", "unsettled",
    "variable",
}


class SentimentScorer:
    """Score financial text sentiment using word-list approach."""

    def __init__(self, dictionary_path: Optional[str] = None):
        self.positive_words = _POSITIVE_WORDS.copy()
        self.negative_words = _NEGATIVE_WORDS.copy()
        self.uncertainty_words = _UNCERTAINTY_WORDS.copy()

        if dictionary_path and Path(dictionary_path).exists():
            self._load_full_dictionary(dictionary_path)

    def _load_full_dictionary(self, path: str) -> None:
        """Load full Loughran-McDonald dictionary from CSV."""
        try:
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    word = row.get("Word", "").lower()
                    if not word:
                        continue
                    if row.get("Positive", "0") != "0":
                        self.positive_words.add(word)
                    if row.get("Negative", "0") != "0":
                        self.negative_words.add(word)
                    if row.get("Uncertainty", "0") != "0":
                        self.uncertainty_words.add(word)
            logger.info(f"Loaded LM dictionary: {len(self.positive_words)} pos, {len(self.negative_words)} neg words")
        except Exception as e:
            logger.warning(f"Could not load dictionary from {path}: {e}")

    def score_text(self, text: str) -> float:
        """Score text sentiment. Returns value in [-1, +1].

        -1 = very negative, 0 = neutral, +1 = very positive
        """
        words = self._tokenize(text)
        if not words:
            return 0.0

        pos_count = sum(1 for w in words if w in self.positive_words)
        neg_count = sum(1 for w in words if w in self.negative_words)
        total = len(words)

        if total == 0:
            return 0.0

        # Net sentiment normalized by total word count
        score = (pos_count - neg_count) / total
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, score * 10))  # Scale factor for readability

    def score_detailed(self, text: str) -> dict:
        """Get detailed sentiment breakdown."""
        words = self._tokenize(text)
        total = len(words)

        pos_count = sum(1 for w in words if w in self.positive_words)
        neg_count = sum(1 for w in words if w in self.negative_words)
        unc_count = sum(1 for w in words if w in self.uncertainty_words)

        return {
            "sentiment_score": self.score_text(text),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "uncertainty_count": unc_count,
            "total_words": total,
            "positive_pct": pos_count / max(total, 1) * 100,
            "negative_pct": neg_count / max(total, 1) * 100,
            "uncertainty_pct": unc_count / max(total, 1) * 100,
        }

    def _tokenize(self, text: str) -> list[str]:
        """Simple word tokenization for financial text."""
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        return [w for w in text.split() if len(w) > 2]


# Module-level singleton
_scorer = SentimentScorer()


def score_text(text: str) -> float:
    """Convenience function for quick sentiment scoring."""
    return _scorer.score_text(text)


def score_detailed(text: str) -> dict:
    """Convenience function for detailed sentiment breakdown."""
    return _scorer.score_detailed(text)
