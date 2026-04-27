from pathlib import Path
import json
from datetime import datetime
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
FINAL_DIR = BASE_DIR / "data" / "final"

SENTIMENT_COLORS = {
    "POSITIVE": "#2ecc71",
    "NEGATIVE": "#e74c3c",
    "NEUTRAL": "#95a5a6",
    "CAUTIOUS": "#f39c12",
}

BASE_BADGE_STYLE = (
    "padding:3px 9px;"
    "border-radius:12px;"
    "font-size:0.75rem;"
    "font-weight:600;"
    "display:inline-block;"
)

SENTIMENT_BADGE_STYLE = (
    "padding:5px 12px;"
    "border-radius:14px;"
    "font-size:0.85rem;"
    "font-weight:700;"
    "display:inline-block;"
)

SMALL_LABEL_STYLE = "font-size:0.85rem; color:#6b7280; font-weight:500;"
SECTION_TITLE_STYLE = "font-size:1.1rem; font-weight:700; margin-top:13px; margin-bottom:4px;"


def get_date_from_file(path):
    return path.stem.replace("combined_", "")


def parse_date(date_str):
    return datetime.strptime(date_str, "%d-%m-%Y")


@st.cache_data(ttl=60) 
def load_dates():
    files = list(FINAL_DIR.glob("combined_*.json"))
    dates = [get_date_from_file(file) for file in files]
    return sorted(dates, key=parse_date, reverse=True)


@st.cache_data
def load_data(date):
    path = FINAL_DIR / f"combined_{date}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def badge(text, bg="#ecf0f1", color="#2c3e50", style=BASE_BADGE_STYLE):
    return (
        f"<span style='{style}"
        f"background-color:{bg};"
        f"color:{color};"
        f"'>"
        f"{text}"
        f"</span>"
    )


def labelled_badge(label, badge_html):
    return f"<span style='{SMALL_LABEL_STYLE}'>{label}:</span> {badge_html}"


def sentiment_badge(label):
    if not label:
        return badge("N/A", style=SENTIMENT_BADGE_STYLE)

    label = str(label).upper()
    color = SENTIMENT_COLORS.get(label, "#bdc3c7")
    return badge(label, color, "white", style=SENTIMENT_BADGE_STYLE)


def confidence_badge(label):
    if not label:
        return badge("N/A")

    label_lower = label.lower()

    if "high" in label_lower:
        return badge(label, "#2ecc71", "white")
    elif "medium" in label_lower:
        return badge(label, "#f39c12", "white")
    elif "low" in label_lower:
        return badge(label, "#e74c3c", "white")
    else:
        return badge(label)


def topic_confidence_badge(prob):
    if prob is None:
        return badge("N/A")

    pct = prob * 100
    text = f"{pct:.1f}%"

    if pct >= 70:
        return badge(text, "#2ecc71", "white")
    elif pct >= 40:
        return badge(text, "#f39c12", "white")
    else:
        return badge(text, "#e74c3c", "white")


def sentiment_stat(label, value):
    return f"""
    <div>
        <div style="font-size:0.85rem; color:#6b7280; font-weight:500; margin-bottom:4px;">
            {label}
        </div>
        <div style="font-size:1.85rem; font-weight:500; color:#262730; line-height:1.2;">
            {value}
        </div>
    </div>
    """


st.set_page_config(page_title="Policy Insight Platform", layout="wide")
st.markdown("""
    <style>
    [data-baseweb="tag"] {
        background-color: #e5e7eb !important;
        color: #374151 !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ Policy Insight Platform")

dates = load_dates()
if not dates:
    st.warning("No combined JSON files found in data/final")
    st.stop()

selected_date = st.selectbox("Select Sitting Date", dates, index=0)

data = load_data(selected_date)
results = data.get("results", [])

st.info(
    "**ℹ️ How Sentiment is Calculated**\n\n"
    "Each speaker turn is classified individually. The overall topic sentiment "
    "is then calculated using a weighted score — speeches by senior office holders "
    "(Prime Minister, Ministers) carry more weight than backbench contributions, "
    "on the basis that their statements carry greater policy significance.\n\n"
    "The turn breakdown (% Positive, % Negative, etc.) shows the raw count of how "
    "many speakers expressed each sentiment, regardless of seniority. In some cases, "
    "the topic sentiment label may differ from the majority turn count."
)

sectors = sorted({
    item.get("sector_classification")
    for item in results
    if item.get("sector_classification")
})

sentiments = sorted({
    item.get("topic_sentiment")
    for item in results
    if item.get("topic_sentiment")
})

filter_col1, filter_col2 = st.columns(2)

with filter_col1:
    selected_sector = st.multiselect("Filter by sector", sectors)

with filter_col2:
    selected_sentiment = st.multiselect("Filter by sentiment", sentiments)

if selected_sector:
    results = [
        item for item in results
        if item.get("sector_classification") in selected_sector
    ]

if selected_sentiment:
    results = [
        item for item in results
        if item.get("topic_sentiment") in selected_sentiment
    ]

st.caption(f"Showing {len(results)} debates for {selected_date}")

groups = {}
for item in results:
    topic = item.get("meta_topic_label") or "Unclassified"
    groups.setdefault(topic, []).append(item)

for idx, (topic_label, items) in enumerate(groups.items()):
    with st.expander(f"{topic_label} — {len(items)} debates", expanded=(idx == 0)):

        for card_idx, item in enumerate(items):
            with st.container(border=True):
                st.markdown(f"#### {item.get('title', 'Untitled')}")

                sector = item.get("sector_classification") or "Pending"
                confidence = item.get("sector_confidence_band") or "N/A"
                prob = item.get("topic_prob")

                col1, col2, col3 = st.columns([2, 2, 2])

                with col1:
                    st.markdown(
                        labelled_badge("Sector", badge(sector, "#34495e", "white")),
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.markdown(
                        labelled_badge("Sector Confidence", confidence_badge(confidence)),
                        unsafe_allow_html=True,
                    )

                with col3:
                    st.markdown(
                        labelled_badge("Topic Relevance", topic_confidence_badge(prob)),
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f"<div style='{SECTION_TITLE_STYLE}'>📊 Sentiment</div>",
                    unsafe_allow_html=True,
                )

                sent_col1, sent_col2, sent_col3, sent_col4, sent_col5, sent_col6 = st.columns(6)

                with sent_col1:
                    st.markdown(
                        "<div style='font-size:0.85rem; color:#6b7280; font-weight:500; margin-bottom:4px;'>Overall</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        sentiment_badge(item.get("topic_sentiment")),
                        unsafe_allow_html=True,
                    )

                with sent_col2:
                    st.markdown(
                        sentiment_stat("Turns", item.get("sentiment_num_turns", "N/A")),
                        unsafe_allow_html=True,
                    )

                with sent_col3:
                    st.markdown(
                        sentiment_stat("Cautious", f"{item.get('sentiment_cautious_pct', 0):.1f}%"),
                        unsafe_allow_html=True,
                    )

                with sent_col4:
                    st.markdown(
                        sentiment_stat("Negative", f"{item.get('sentiment_negative_pct', 0):.1f}%"),
                        unsafe_allow_html=True,
                    )

                with sent_col5:
                    st.markdown(
                        sentiment_stat("Neutral", f"{item.get('sentiment_neutral_pct', 0):.1f}%"),
                        unsafe_allow_html=True,
                    )

                with sent_col6:
                    st.markdown(
                        sentiment_stat("Positive", f"{item.get('sentiment_positive_pct', 0):.1f}%"),
                        unsafe_allow_html=True,
                    )

                summary = item.get("summary") or "No summary available"

                st.markdown(
                    f"<div style='{SECTION_TITLE_STYLE}'>📝 Summary</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div style="
                        margin-top:6px;
                        margin-bottom:8px;
                        padding:12px 14px;
                        background-color:#f8fafc;
                        border-left:4px solid #cbd5e1;
                        border-radius:8px;
                        font-size:0.95rem;
                        line-height:1.6;
                        color:#333;
                    ">
                        {summary}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if card_idx < len(items) - 1:
                st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)