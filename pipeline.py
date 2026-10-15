"""Runs sentiment + topic analysis on a cleaned feedback DataFrame."""
from sentiment import classify_texts
from topics import extract_topics


def analyze(df):
    """Return (analysed_df, emerging_terms)."""
    out = df.copy()
    texts = out["feedback"].tolist()
    labels, scores = classify_texts(texts)
    topic_lists, emerging = extract_topics(texts)
    out["sentiment"] = labels
    out["score"] = scores
    out["topics"] = topic_lists
    out["topics_text"] = [", ".join(t) for t in topic_lists]
    return out, emerging