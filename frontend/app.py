from pathlib import Path
import json
from datetime import datetime
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
FINAL_DIR = BASE_DIR / "data" / "final"

SENTIMENT_COLORS = {
    "POSITIVE": "#2ecc71",   # green
    "NEGATIVE": "#e74c3c",   # red
    "NEUTRAL": "#95a5a6",    # grey
    "CAUTIOUS": "#f39c12",   # orange
}

def get_date_from_file(path):
    return path.stem.replace("combined_", "")


def parse_date(date_str):
    return datetime.strptime(date_str, "%d-%m-%Y")


def load_dates():
    files = list(FINAL_DIR.glob("combined_*.json"))
    dates = [get_date_from_file(file) for file in files]
    return sorted(dates, key=parse_date, reverse=True)


def load_data(date):
    path = FINAL_DIR / f"combined_{date}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    

def sentiment_badge(label):
    if not label:
        return "<span style='color: grey;'>N/A</span>"

    color = SENTIMENT_COLORS.get(label, "#bdc3c7")

    return f"""
    <span style="
        background-color: {color};
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    ">
        {label}
    </span>
    """


st.set_page_config(page_title="Policy Insight Platform", layout="wide")

st.title("🏛️ Policy Insight Platform")

dates = load_dates()
if not dates:
    st.warning("No combined JSON files found in data/final")
    st.stop()

selected_date = st.selectbox("Select sitting date", dates, index=0)

data = load_data(selected_date)
results = data.get("results", [])

st.caption(f"Showing {len(results)} debates for {selected_date}")

groups = {}
for item in results:
    topic = item.get("meta_topic_label") or "Unclassified"
    groups.setdefault(topic, []).append(item)

for idx, (topic_label, items) in enumerate(groups.items()):
    with st.expander(f"{topic_label} — {len(items)} debates", expanded=(idx == 0)):
        for card_idx, item in enumerate(items):
            with st.container(border=True):
                st.markdown(f"### {item.get('title', 'Untitled')}")
                
                col1, col2, col3 = st.columns([1, 1, 1])

                with col1:
                    st.markdown(
                        f"**Sentiment:** {sentiment_badge(item.get('topic_sentiment'))}",
                        unsafe_allow_html=True
                    )

                with col2:
                    sector = item.get("sector_classification") or "Pending"
                    st.markdown(f"**Sector:** `{sector}`")

                with col3:
                    prob = item.get("topic_prob")
                    prob_percent = f"{prob * 100:.1f}%" if prob is not None else "N/A"
                    st.markdown(f"**Topic confidence:** `{prob_percent}`")

                st.markdown("**📝 Summary**")
                st.info(item.get("summary") or "No summary available")

            if card_idx < len(items) - 1:
                st.divider()