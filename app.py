"""AI Feedback Analyzer - Streamlit dashboard."""
import os
from datetime import date, datetime

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from data_loader import DataLoadError, clean_feedback_df, load_csv
from metrics import SENTIMENTS, apply_filters, overall_summary, sentiment_distribution, topic_summary
from pipeline import analyze
from report import build_llm_facts, build_report, generate_llm_insights, report_to_text
from sentiment import classify_text
from topics import extract_topics

st.set_page_config(page_title="AI Feedback Analyzer", page_icon="🎓", layout="wide")

SAMPLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_feedback.csv")
COLORS = {"Positive": "#2E9E6B", "Neutral": "#F2A93B", "Negative": "#D64545"}
CELL_BG = {"Positive": "#d4f5e3", "Neutral": "#fff1cc", "Negative": "#fbd9d9"}
PAGES = ["📊 Dashboard", "🗂 Feedback Table", "🧠 AI Insights & Report", "✍️ Add Feedback"]

st.markdown("""
<style>
.metric-card{background:#ffffff;border-radius:14px;padding:16px 18px;border-left:6px solid #4C6EF5;
  box-shadow:0 2px 8px rgba(0,0,0,.10);margin-bottom:8px;}
.metric-label{font-size:.85rem;color:#6b7280;font-weight:600;text-transform:uppercase;letter-spacing:.04em;}
.metric-value{font-size:2rem;font-weight:700;color:#1f2937;line-height:1.2;}
.metric-sub{font-size:.9rem;color:#6b7280;}
</style>
""", unsafe_allow_html=True)


def metric_card(label, value, sub="", color="#4C6EF5"):
    st.markdown(
        f'<div class="metric-card" style="border-left-color:{color}">'
        f'<div class="metric-label">{label}</div><div class="metric-value">{value}</div>'
        f'<div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)


def color_cell(val):
    bg = CELL_BG.get(val)
    return f"background-color:{bg};color:#1f2937;font-weight:600" if bg else ""


@st.cache_data(show_spinner=False)
def run_analysis(df):
    return analyze(df)


def reset_filters():
    st.session_state["f_sent"] = []
    st.session_state["f_topic"] = []
    st.session_state["f_search"] = ""


st.session_state.setdefault("manual", [])
st.session_state.setdefault("report", None)

# ------------------------------------------------------------------ sidebar (data source)
with st.sidebar:
    st.title("🎓 Feedback Analyzer")
    page = st.radio("Navigate", PAGES)
    st.divider()
    st.subheader("Data source")
    uploaded = st.file_uploader("Upload feedback CSV", type=["csv"],
                                help="Needs a column named 'feedback' (or comment/review/text).")
    use_sample = st.checkbox("Use sample data", value=True,
                             help="An uploaded CSV replaces the sample data. Manual entries are always added.")

# ------------------------------------------------------------------ build dataset
frames, notices, errors = [], [], []
if uploaded is not None:
    try:
        df_up, warns = load_csv(uploaded, "Uploaded CSV")
        frames.append(df_up)
        notices += warns
    except DataLoadError as exc:
        errors.append(f"Upload failed: {exc}")
if not frames and use_sample:
    try:
        df_sample, _ = load_csv(SAMPLE_PATH, "Sample data")
        frames.append(df_sample)
    except DataLoadError as exc:
        errors.append(f"Sample data problem: {exc}")
if st.session_state["manual"]:
    frames.append(pd.DataFrame(st.session_state["manual"]))

if frames:
    combined, warns = clean_feedback_df(pd.concat(frames, ignore_index=True))
    notices += warns
else:
    combined = pd.DataFrame()

if combined.empty:
    analyzed, emerging = pd.DataFrame(), []
else:
    analyzed, emerging = run_analysis(combined)

with st.sidebar:
    for e in errors:
        st.error(e)
    for n in notices:
        st.warning(n)
    if not analyzed.empty:
        st.success(f"{len(analyzed)} feedback entries loaded.")

    st.divider()
    st.subheader("Filters")
    options = list(topic_summary(analyzed)["Topic"]) if not analyzed.empty else []
    st.session_state["f_topic"] = [t for t in st.session_state.get("f_topic", []) if t in options]
    sent_filter = st.multiselect("Sentiment", SENTIMENTS, key="f_sent")
    topic_filter = st.multiselect("Topic", options, key="f_topic")
    search = st.text_input("Search feedback text", key="f_search")
    st.button("Reset filters", on_click=reset_filters)
    st.caption("Leave a filter empty to include everything.")

filtered = apply_filters(analyzed, sent_filter, topic_filter, search)
filters_active = bool(sent_filter or topic_filter or search.strip())

# ------------------------------------------------------------------ pages
if page == "✍️ Add Feedback":
    st.title("✍️ Add Feedback")
    flash = st.session_state.pop("flash", None)
    if flash:
        getattr(st, flash[0])(flash[1])
    with st.form("add_form", clear_on_submit=True):
        text = st.text_area("Student feedback", height=140, placeholder="e.g. The library is great but the Wi-Fi is slow.")
        course = st.text_input("Course / department (optional)")
        submitted = st.form_submit_button("Submit feedback")
    if submitted:
        clean_text = " ".join(text.split())
        existing = set(analyzed["feedback"].str.lower()) if not analyzed.empty else set()
        if not clean_text:
            st.warning("Please type some feedback before submitting.")
        elif len(clean_text) < 3:
            st.warning("Feedback is too short to analyse.")
        elif clean_text.lower() in existing:
            st.warning("This exact feedback already exists, so it was not added again.")
        else:
            st.session_state["manual"].append({
                "feedback": clean_text, "course": course.strip(),
                "date": date.today().isoformat(), "source": "Manual entry"})
            label, score = classify_text(clean_text)
            topics_found, _ = extract_topics([clean_text])
            st.session_state["flash"] = (
                "success", f"Feedback added. Sentiment: {label} ({score:+.2f}); topics: {', '.join(topics_found[0])}.")
            st.rerun()
    if st.session_state["manual"]:
        st.subheader(f"Manual entries this session ({len(st.session_state['manual'])})")
        st.dataframe(pd.DataFrame(st.session_state["manual"])[["date", "course", "feedback"]],
                     use_container_width=True, hide_index=True)
        if st.button("Clear manual entries"):
            st.session_state["manual"] = []
            st.rerun()

elif analyzed.empty:
    st.title("🎓 AI Feedback Analyzer")
    st.warning("No feedback loaded. Upload a CSV, tick 'Use sample data' in the sidebar, or add feedback manually.")

elif page == "📊 Dashboard":
    st.title("📊 Feedback Dashboard")
    st.caption(f"Showing {len(filtered)} of {len(analyzed)} entries" + (" (filters active)" if filters_active else ""))
    if emerging:
        st.info("🔍 Auto-discovered recurring terms outside the predefined themes: "
                + ", ".join(f"**{t}**" for t in emerging))
    if filtered.empty:
        st.warning("No feedback matches the current filters.")
        st.stop()

    summary = overall_summary(filtered)
    dist = sentiment_distribution(filtered).set_index("Sentiment")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total feedback", summary["total"], f"Avg score {summary['avg_score']:+.2f}")
    for col, s in zip((c2, c3, c4), SENTIMENTS):
        with col:
            metric_card(s, int(dist.loc[s, "Count"]), f"{dist.loc[s, 'Percent']}% of entries", COLORS[s])

    ts = topic_summary(filtered)
    left, right = st.columns(2)
    with left:
        pie = px.pie(dist.reset_index(), names="Sentiment", values="Count", hole=0.5,
                     color="Sentiment", color_discrete_map=COLORS, title="Sentiment distribution")
        pie.update_traces(textinfo="percent+label")
        st.plotly_chart(pie, use_container_width=True)
    with right:
        long = ts.melt(id_vars="Topic", value_vars=SENTIMENTS, var_name="Sentiment", value_name="Count")
        bar = px.bar(long, x="Topic", y="Count", color="Sentiment", color_discrete_map=COLORS,
                     category_orders={"Topic": list(ts["Topic"]), "Sentiment": SENTIMENTS},
                     title="Feedback by topic and sentiment")
        bar.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(bar, use_container_width=True)

    st.subheader("Most discussed topics")
    top3 = ts.head(3)
    st.write("  •  ".join(f"**{r.Topic}** ({int(r.Total)})" for r in top3.itertuples()))
    st.dataframe(ts, use_container_width=True, hide_index=True)
    st.caption("One feedback entry can belong to several topics, so topic counts can add up to more than the total.")

elif page == "🗂 Feedback Table":
    st.title("🗂 Feedback Table")
    st.caption(f"Showing {len(filtered)} of {len(analyzed)} entries. Click a column header to sort.")
    if filtered.empty:
        st.warning("No feedback matches the current filters.")
        st.stop()
    table = filtered[["id", "date", "course", "feedback", "sentiment", "score", "topics_text"]].copy()
    table.columns = ["ID", "Date", "Course", "Feedback", "Sentiment", "Score", "Topics"]
    styler = table.style.format({"Score": "{:+.2f}"})
    style_fn = getattr(styler, "map", None) or styler.applymap   # pandas >= 2.1 uses .map
    styler = style_fn(color_cell, subset=["Sentiment"])
    st.dataframe(styler, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download filtered data (CSV)",
                       filtered.drop(columns=["topics"]).to_csv(index=False).encode("utf-8"),
                       "filtered_feedback.csv", "text/csv")

elif page == "🧠 AI Insights & Report":
    st.title("🧠 AI Insights & Recommendations")
    st.caption("The report is built from all loaded feedback (sidebar filters do not apply here).")
    has_key = bool(os.getenv("GEMINI_API_KEY"))
    use_llm = st.checkbox("Also generate an optional Gemini narrative", value=False, disabled=not has_key)
    if not has_key:
        st.caption("Optional: set GEMINI_API_KEY in a .env file to enable. The report works without it.")
    else:
        st.caption("If enabled, aggregated statistics and a few short example comments are sent to the Gemini API.")

    if st.button("Generate / refresh report", type="primary"):
        with st.spinner("Analysing feedback..."):
            data = build_report(analyzed, topic_summary(analyzed))
            llm_text, llm_note = (None, None)
            if use_llm:
                llm_text, llm_note = generate_llm_insights(build_llm_facts(analyzed, data))
            st.session_state["report"] = {
                "data": data, "llm": llm_text, "note": llm_note,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"), "n": len(analyzed)}

    saved = st.session_state["report"]
    if saved is None:
        st.info("Click **Generate / refresh report** to create the report.")
    else:
        if saved["n"] != len(analyzed):
            st.warning("The data has changed since this report was generated. Click refresh to update it.")
        rep = saved["data"]
        st.success(f"Report generated {saved['time']}")
        st.subheader("Part A - Computed from your data")
        st.markdown("**Overall summary**")
        st.write(rep["summary"])
        for title, key in [("Key positive findings", "positives"), ("Major complaints and concerns", "complaints"),
                           ("Frequently mentioned topics", "topics"), ("Actionable recommendations", "recommendations")]:
            st.markdown(f"**{title}**")
            for item in rep[key]:
                st.markdown(f"- {item}")
        with st.expander("Notes and limitations"):
            for item in rep["caveats"]:
                st.markdown(f"- {item}")
        if saved["llm"]:
            st.subheader("Part B - Gemini-written narrative (LLM output)")
            st.caption("Generated by an LLM from the aggregated data. Verify against Part A.")
            st.write(saved["llm"])
        elif saved["note"]:
            st.warning(saved["note"])
        st.download_button("⬇️ Download report (.txt)", report_to_text(rep, saved["llm"], saved["time"]),
                           "feedback_report.txt", "text/plain")