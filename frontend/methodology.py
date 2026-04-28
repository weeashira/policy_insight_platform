import streamlit as st

def render_methodology():
    st.title("Methodology & Disclaimers")

    st.markdown(
        """
        This page explains how the dashboard outputs are generated and the key disclaimers users should keep in mind.
        """
    )

    st.divider()

    with st.expander("1. Sector Classification", expanded=False):
        render_sector_classification()

    with st.expander("2. Topic Modelling", expanded=False):
        render_topic_modelling()

    with st.expander("3. Sentiment Classification", expanded=False):
        render_sentiment_classification()

    with st.expander("4. Summarization", expanded=False):
        render_summarization()


def render_sector_classification():
    st.subheader("Methodology")

    st.markdown(
        """
        The module assigns a sector to each debate using a supervised classification model that analyses the title and the first two speaker turns. 
        The "Sector Confidence" is derived from the model’s probability score and is classified as follows:
        """
    )

    st.markdown(
        """
        - **High:** Score > 0.3  
        - **Medium:** Score 0.16 – 0.3  
        - **Low / Needs Review:** Score < 0.16
        """
    )

    st.subheader("Disclaimers")
    st.warning(
        """
        **Focus on Opening:** The sector classification is based on the opening of the debate, using the title and first two speaker turns as a proxy for the main topic. 
        It may not capture a shift in subject matter if the discussion pivots or extends over time.

        **Cross-Cutting:** Debates that span multiple policy areas are labeled as "Other / Cross-Cutting."
        """
    )


def render_topic_modelling():
    st.subheader("Methodology")
    st.markdown(
        """
        This module groups debates into themes by looking at the title and the first two speaker turns.
        It uses LDA (Latent Dirichlet Allocation) to find patterns in words and Claude (AI) to provide a readable name for each theme.
        Each debate is assigned a topic relevance score, derived from the model, which reflects how strongly the content aligns with the identified theme.
        """
    )

    st.subheader("Disclaimers")
    st.warning(
        """
        **Automated Clustering:** Themes are grouped by mathematical patterns rather than human logic. Some topics may include a mix of debates that are not closely related.

        **Snapshot Context:** Topics are generated based on the vocabulary used within 22 specific sittings. They represent trends from this particular time period rather than a permanent list of categories.
        """
    )


def render_sentiment_classification():
    st.subheader("Methodology")
    st.markdown(
        """
        Each speaker turn is classified individually using a finetuned Longformer model.
        The overall "Topic Sentiment" is calculated using a weighted score.
        Contributions by senior office holders carry more weight than backbenchers on the basis that their statements carry greater policy significance.
        The following role weightings are applied:
        """
    )

    st.markdown(
        """
        **Role Weightings:**

        - **1.25:** Tier 1 (PM, DPM, Ministers, SMS, MOS, SPS)
        - **1.00:** Tier 2 (Majority Party Backbenchers)
        - **0.75:** Tier 3 (Opposition Leadership)
        - **0.50:** Tier 4 (Nominated MPs / Procedural Turns)
        """
    )

    st.subheader("Disclaimers")
    st.warning(
        """
        **Label vs. Volume:** The turn breakdown (% Positive, % Negative, etc.) reflects the raw count of speakers by sentiment, regardless of seniority. 
        As the overall topic sentiment is calculated using a weighted scoring system, the final label may differ from the majority turn count.       
        """
    )


def render_summarization():
    st.subheader("Methodology")
    st.markdown(
        """
        Summaries are produced by a Qwen 3.5 (9B) model to provide a quick overview of transcripts.
        The module uses two different paths based on the length of the debate:
        """
    )

    st.markdown(
        """
        - **Short Debates:** The module creates a summary directly from the transcript.
        - **Long Debates:** The module first breaks the transcript into chunks to extract key facts. It then combines these facts into a final summary to ensure important details are not missed.
        """
    )

    st.subheader("Disclaimers")
    st.warning(
        """
        **Policy Focus:** The module is instructed to prioritize key facts like dates, agency names, and specific measures. It may omit conversational or procedural details to keep the summary concise.

        **AI Generation:** While the system is designed to be factually dense, AI can occasionally misinterpret technical details or local references. These summaries should be used for quick insight rather than as a word-for-word record.
        """
    )

