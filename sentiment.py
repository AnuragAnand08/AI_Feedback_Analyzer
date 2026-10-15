"""Sentiment classification with VADER (lexicon-based, runs fully offline)."""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

POSITIVE_THRESHOLD = 0.05   # standard VADER cut-offs
NEGATIVE_THRESHOLD = -0.05

# Campus-specific words that the general-purpose VADER lexicon may score weakly
DOMAIN_LEXICON = {
    "outdated": -2.0, "overcrowded": -2.2, "unhygienic": -2.5, "insufficient": -1.8,
    "irregular": -1.5, "unhelpful": -2.2, "delayed": -1.6, "disorganized": -2.0,
    "disorganised": -2.0, "cramped": -1.5, "overpriced": -2.0, "unreliable": -2.0,
    "knowledgeable": 2.2, "supportive": 2.0, "well-maintained": 2.2, "spacious": 1.8,
}

_analyzer = None


def get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
        _analyzer.lexicon.update(DOMAIN_LEXICON)
    return _analyzer


def classify_text(text):
    """Return (label, compound_score) for one feedback string."""
    score = get_analyzer().polarity_scores(str(text))["compound"]
    if score >= POSITIVE_THRESHOLD:
        return "Positive", score
    if score <= NEGATIVE_THRESHOLD:
        return "Negative", score
    return "Neutral", score


def classify_texts(texts):
    """Classify many texts. Returns (labels, scores)."""
    results = [classify_text(t) for t in texts]
    return [r[0] for r in results], [r[1] for r in results]