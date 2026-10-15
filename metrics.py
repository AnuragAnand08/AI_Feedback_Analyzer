"""Dashboard statistics and filtering. Everything is computed from the data."""
import pandas as pd

SENTIMENTS = ["Positive", "Neutral", "Negative"]


def sentiment_distribution(df):
    total = len(df)
    counts = df["sentiment"].value_counts() if total else pd.Series(dtype=int)
    rows = []
    for s in SENTIMENTS:
        c = int(counts.get(s, 0))
        rows.append({"Sentiment": s, "Count": c, "Percent": round(100 * c / total, 1) if total else 0.0})
    return pd.DataFrame(rows)


def topic_summary(df):
    """One row per topic with sentiment breakdown (a feedback can count in several topics)."""
    columns = ["Topic"] + SENTIMENTS + ["Total", "Negative %"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    exploded = df[["sentiment", "topics"]].explode("topics").rename(columns={"topics": "Topic"})
    table = pd.crosstab(exploded["Topic"], exploded["sentiment"])
    for s in SENTIMENTS:
        if s not in table.columns:
            table[s] = 0
    table = table[SENTIMENTS].copy()
    table.columns.name = None
    table["Total"] = table.sum(axis=1)
    table["Negative %"] = (100 * table["Negative"] / table["Total"]).round(1)
    table = table.reset_index().sort_values(["Total", "Topic"], ascending=[False, True])
    return table.reset_index(drop=True)


def overall_summary(df):
    total = len(df)
    return {"total": total, "avg_score": float(df["score"].mean()) if total else 0.0}


def apply_filters(df, sentiments=None, topics=None, search=""):
    """Empty selections mean 'no filter'."""
    if df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    if sentiments:
        mask &= df["sentiment"].isin(sentiments)
    if topics:
        mask &= df["topics"].apply(lambda ts: any(t in topics for t in ts))
    if search and search.strip():
        mask &= df["feedback"].str.contains(search.strip(), case=False, regex=False, na=False)
    return df[mask]