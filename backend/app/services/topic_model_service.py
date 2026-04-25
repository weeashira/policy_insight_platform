from pathlib import Path
import json
import re
import string

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")

# Paths
BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_DIR = BASE_DIR / "backend" / "models" / "topic_modelling"

LDA_MODEL_PATH = MODEL_DIR / "lda_model.gensim"
DICTIONARY_PATH = MODEL_DIR / "lda_dictionary.gensim"

IN_DIR = BASE_DIR / "data" / "interim"
OUT_DIR = BASE_DIR / "data" / "processed" / "topics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Meta-topic names determined by LLM (Claude) based on top keywords
TOPIC_NAMES = {
    0:  "Housing & Property Policy",
    1:  "Community Schemes & Senior Support",
    2:  "Workforce Support & Business Grants",
    3:  "Public Transport & Commuting",
    4:  "Household Finance & Income Support",
    5:  "Parliamentary Procedures & Foreign Affairs",
    6:  "Road Safety & Public Security",
    7:  "Industry & Enterprise Development",
    8:  "Urban Planning & Estate Development",
    9:  "Labour & Workplace Relations",
    10: "Trade, Energy & Consumer Affairs",
    11: "Healthcare & Mental Health",
    12: "Education & Child Development",
    13: "Caregiving & Family Leave",
    14: "Sports, Recreation & Community",
    15: "Law Enforcement & Online Safety"
}


# Parliamentary-specific stopwords that carry no topic meaning
CUSTOM_STOPWORDS = {
    'mr', 'mrs', 'ms', 'dr', 'sir', 'minister', 'member', 'parliament',
    'singapore', 'government', 'asked', 'whether', 'would', 'could',
    'also', 'said', 'may', 'will', 'shall', 'upon', 'within','speaker',
    'people', 'one', 'must'
}


PROCEDURAL_KEYWORDS = [
    'written reply', 'parliamentary question',
    'written answer', 'oral reply',
    'order paper', 'sitting date',
    'proceedings on', 'report of',
    'questions without oral answer',
    'supply reporting progress',
    'committee of supply',
]


_lda_model = None
_dictionary = None
_lemmatizer = WordNetLemmatizer()


def load_topic_model():
    global _lda_model, _dictionary

    if _lda_model is None:
        _lda_model = LdaModel.load(str(LDA_MODEL_PATH))
    
    if _dictionary is None:
        _dictionary = Dictionary.load(str(DICTIONARY_PATH))

    return _lda_model, _dictionary


def clean_text(text):
    stop_words = set(stopwords.words('english'))
    stop_words = stop_words.union(CUSTOM_STOPWORDS)

    # Lowercase
    text = text.lower()

    # Remove non-ASCII characters (e.g. curly quotes, em-dashes, Chinese characters)
    text = text.encode('ascii', 'ignore').decode('ascii')

   # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()    

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and short words
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    # 詞形還原 / Lemmatize
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    return tokens


def prepare_topic_input(df):
    # For each debate title, keep only the first 2 speaker turns (segment_no = 1 and 2)
    df_top2 = df[df['segment_no'] <= 2].copy()

    # Concatenate the first 2 turns into a single text per title
    df_combined = (
        df_top2
        .groupby(['sittingDate', 'title'])['speech_text']
        .apply(lambda x: ' '.join(x.dropna().astype(str)))
        .reset_index()
    )

    # Prepend the title text to the combined speech text
    df_combined['full_text'] = df_combined['title'] + ' ' + df_combined['speech_text']

    # Remove procedural debates before topic assignment
    mask = df_combined["title"].str.lower().apply(
        lambda x: not any(kw in x for kw in PROCEDURAL_KEYWORDS)
    )

    df_combined = df_combined[mask].reset_index(drop=True)
    return df_combined


def assign_topics_to_df(df_combined):
    lda_model, dictionary = load_topic_model()
    df_combined["cleaned_tokens"] = df_combined["full_text"].apply(clean_text)

    # Corpus: convert each debate's tokens into Bag-of-Words format
    corpus = [dictionary.doc2bow(tokens) for tokens in df_combined['cleaned_tokens']]

    # Each debate has probabilities for all topics; we take the highest as dominant topic
    dominant_topics = []
    for _, bow in enumerate(corpus):
        # Get topic probability distribution for this debate
        topic_probs = lda_model.get_document_topics(bow, minimum_probability=0)

        # Find the topic with highest probability
        dominant_topic = max(topic_probs, key=lambda x: x[1])
        dominant_topics.append({
            'topic_id': int(dominant_topic[0]),
            'topic_probability': float(round(dominant_topic[1], 4))
        })

    # Merge back into the main DataFrame
    dominant_df = pd.DataFrame(dominant_topics)
    df_combined['dominant_topic_id'] = dominant_df['topic_id']
    df_combined['topic_probability'] = dominant_df['topic_probability']
    df_combined['meta_topic_label'] = df_combined["dominant_topic_id"].map(TOPIC_NAMES)
    return df_combined


def run_topic_model(date):
    input_csv = IN_DIR / f"speaker_turns_{date}.csv"
    output_json = OUT_DIR / f"topics_{date}.json"

    try:
        if output_json.exists():
            return {
                "success": True,
                "date": date,
                "topic_path": str(output_json)
            }

        if not input_csv.exists():
            return {
                "success": False,
                "date": date,
                "error": f"Input file not found: {input_csv}",
            }

        df = pd.read_csv(input_csv)
        df_combined = prepare_topic_input(df)
        df_combined = assign_topics_to_df(df_combined)

        results = []

        for _, row in df_combined.iterrows():
            results.append({
                "title": row["title"],
                "meta_topic_label": row["meta_topic_label"],
                "topic_id": int(row["dominant_topic_id"]),
                "topic_probability": float(row["topic_probability"]),
            })

        output = {
            "success": True,
            "date": date,
            "results": results,
        }

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        return {
            "success": True,
            "date": date,
            "topic_path": str(output_json)
        }

    except Exception as e:
        if output_json.exists():
            output_json.unlink()

        return {
            "success": False,
            "date": date,
            "error": str(e),
        }
