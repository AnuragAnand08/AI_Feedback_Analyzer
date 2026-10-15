import io
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_loader import DataLoadError, clean_feedback_df, load_csv
from metrics import apply_filters, sentiment_distribution, topic_summary
from pipeline import analyze
from report import build_report
from sentiment import classify_text
from topics import extract_topics

SAMPLE = ROOT / "data" / "sample_feedback.csv"


def sample_analyzed():
    df, _ = load_csv(str(SAMPLE))
    out, _ = analyze(df)
    return out


def test_positive_sentiment():
    assert classify_text("The faculty is excellent and very helpful")[0] == "Positive"

def test_negative_sentiment():
    assert classify_text("The Wi-Fi is terrible and the food is awful")[0] == "Negative"

def test_neutral_sentiment():
    assert classify_text("The class is held on Monday.")[0] == "Neutral"

def test_wifi_topic():
    assert "Internet & Wi-Fi" in extract_topics(["The Wi-Fi is slow"])[0][0]

def test_multiple_topics():
    topics = extract_topics(["The library has good books but the canteen food is bad"])[0][0]
    assert "Library" in topics and "Canteen" in topics

def test_general_topic():
    assert extract_topics(["It was okay overall"])[0][0] == ["General"]

def test_cleaning_removes_missing_and_duplicates():
    raw = pd.DataFrame({"feedback": ["Good teaching", "good teaching!", "", None, "Bad Wi-Fi"]})
    cleaned, warnings = clean_feedback_df(raw)
    assert len(cleaned) == 2 and warnings

def test_csv_without_feedback_column():
    with pytest.raises(DataLoadError):
        load_csv(io.StringIO("a,b\n1,2\n"))

def test_empty_csv():
    with pytest.raises(DataLoadError):
        load_csv(io.StringIO(""))

def test_valid_csv_loads():
    df, _ = load_csv(io.StringIO("feedback\nGreat lectures\nTerrible Wi-Fi\n"))
    assert len(df) == 2

def test_distribution_sums_to_total():
    df = sample_analyzed()
    assert sentiment_distribution(df)["Count"].sum() == len(df) >= 20

def test_sentiment_filter():
    df = sample_analyzed()
    assert set(apply_filters(df, sentiments=["Negative"])["sentiment"]) == {"Negative"}

def test_topic_filter():
    df = sample_analyzed()
    result = apply_filters(df, topics=["Library"])
    assert len(result) > 0 and all("Library" in t for t in result["topics"])

def test_report_uses_real_totals():
    df = sample_analyzed()
    rep = build_report(df, topic_summary(df))
    assert str(len(df)) in rep["summary"] and rep["recommendations"]