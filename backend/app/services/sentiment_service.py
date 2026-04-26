from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import LongformerTokenizer, LongformerForSequenceClassification

# Paths
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "longformer_sentiment"
DATA_DIR  = Path(__file__).resolve().parent.parent.parent.parent / "data"
IN_DIR    = DATA_DIR / "interim"
OUT_DIR   = DATA_DIR / "processed" / "sentiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN    = 4096
BATCH_SIZE = 2  # DirectML/CPU safe; increase to 4-8 if you have >=8GB VRAM

# Device selection: CUDA > DirectML (Intel Arc on Windows) > CPU
def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        pass
    return torch.device("cpu")

DEVICE = _get_device()
print(f"Sentiment service using device: {DEVICE}")

# Lazy-loaded artifacts
_model      = None
_tokenizer  = None
_le         = None
_thresholds = None


# ── Speaker tier lookup (based on 2025 Cabinet Appointments) ──────────

_PM_DPM = {'Mr Lawrence Wong', 'Mr Gan Kim Yong'}

_MINISTERS = {
    'Mr Chan Chun Sing', 'Mr Ong Ye Kung', 'Mr Chee Hong Tat',
    'Mr Desmond Lee', 'Dr Vivian Balakrishnan', 'Mr K Shanmugam',
    'Mr Edwin Tong Chun Fai', 'Dr Tan See Leng', 'Mr Masagos Zulkifli B M M',
    'Ms Grace Fu Hai Yien', 'Mrs Josephine Teo', 'Ms Indranee Rajah',
    'Mr Jeffrey Siow', 'Mr David Neo',
}

_SMS = {
    'Dr Koh Poh Koon', 'Mr Murali Pillai', 'Ms Sun Xueling',
    'Ms Sim Ann', 'Mr Zaqy Mohamad', 'Ms Goh Hanyan',
    'Mr Tan Kiat How', 'Ms Rahayu Mahzam',
}

_MOS = {
    'Mr Desmond Tan', 'Mr Baey Yam Keng', 'Mr Alvin Tan',
    'Ms Low Yen Ling', 'Ms Gan Siow Huang', 'Mr Dinesh Vasu Dash',
    'Mr Goh Pei Ming', 'Dr Janil Puthucheary', 'Mr Zhulkarnain Abdul Rahim',
    'Mr Shawn Huang Wei Zhong', 'Assoc Prof Dr Muhammad Faishal Ibrahim',
    'Mr Eric Chua',
}

_SPS = {'Ms Jasmin Lau', 'Dr Syed Harun Alhabsyi'}

_TIER1 = _PM_DPM | _MINISTERS | _SMS | _MOS | _SPS

_TIER3 = {
    'Mr Pritam Singh', 'Ms Sylvia Lim', 'Mr Gerald Giam Yean Song',
    'Mr Dennis Tan Lip Fong', 'Mr Chua Kheng Wee Louis', 'Ms He Ting Ru',
    'Assoc Prof Jamus Jerome Lim', 'Mr Kenneth Tiong Boon Kiat',
    'Mr Low Wu Yang Andre', 'Ms Eileen Chong Pei Shan', 'Mr Fadli Fawzi',
    'Mr Leong Mun Wai', 'Ms Hazel Poa',
}

_TIER4 = {
    'Ms Kuah Boon Theng', 'Assoc Prof Kenneth Goh', 'Mr Sanjeev Kumar Tiwari',
    'Prof Kenneth Poon', 'Mr Mark Lee', 'Dr Haresh Singaraju',
    'Assoc Prof Terence Ho', 'Dr Neo Kok Beng', 'Mr Azhar Othman',
}

_PROCEDURAL = {
    'Mr Deputy Speaker', 'Mr Speaker', 'Madam Speaker',
    'Mr Chairman', 'Madam Chairman',
}

_ROLE_WEIGHTS = {
    'Tier1': 1.25, 'Tier2': 1.00,
    'Tier3': 0.75, 'Tier4': 0.50, 'Procedural': 0.50,
}


def _assign_tier(speaker):
    if speaker in _TIER1:
        return 'Tier1'
    if speaker in _TIER3:
        return 'Tier3'
    if speaker in _TIER4:
        return 'Tier4'
    if speaker in _PROCEDURAL:
        return 'Procedural'
    return 'Tier2'


# ── Artifact loader ───────────────────────────────────────────────────

def _load_artifacts():
    global _model, _tokenizer, _le, _thresholds
    if _model is not None:
        return

    model_path = MODEL_DIR / "longformer_option_c"

    _tokenizer = LongformerTokenizer.from_pretrained(str(model_path))

    _model = LongformerForSequenceClassification.from_pretrained(str(model_path))
    _model.to(DEVICE)
    _model.eval()

    with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
        _le = pickle.load(f)

    with open(MODEL_DIR / "best_thresholds.pkl", "rb") as f:
        _thresholds = pickle.load(f)

    print(f"Longformer loaded | labels: {list(_le.classes_)} | thresholds: {dict(zip(_le.classes_, _thresholds))}")


# ── Inference helpers ─────────────────────────────────────────────────

class _TurnDataset(TorchDataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}


def _predict_with_thresholds(probs, thresholds):
    """Pick highest-prob class exceeding its threshold; fall back to argmax."""
    preds = []
    for prob_row in probs:
        above = [i for i, (p, t) in enumerate(zip(prob_row, thresholds)) if p >= t]
        if above:
            preds.append(above[int(np.argmax([prob_row[i] for i in above]))])
        else:
            preds.append(int(np.argmax(prob_row)))
    return np.array(preds)


def _aggregate_topic_sentiment(topic_df, sentiment_classes):
    """Role-weighted aggregation of turn predictions into a topic-level sentiment."""
    if len(topic_df) == 0:
        return {
            "topic_sentiment": "NEUTRAL",
            "num_turns": 0,
            "total_role_weight": 0.0,
            **{f"weighted_{cls.lower()}": 0.0 for cls in sentiment_classes},
            **{f"{cls.lower()}_turns": 0 for cls in sentiment_classes},
            **{f"{cls.lower()}_pct": 0.0 for cls in sentiment_classes},
        }

    weighted_scores = {cls: 0.0 for cls in sentiment_classes}
    total_weight = 0.0

    for _, turn in topic_df.iterrows():
        role_weight = turn["role_weight"]
        pred_label  = turn["predicted_label"]
        conf_score  = float(turn["confidence_score"])
        weighted_scores[pred_label] += role_weight * conf_score
        total_weight += role_weight

    if total_weight > 0:
        normalised = {cls: score / total_weight for cls, score in weighted_scores.items()}
    else:
        normalised = {cls: 0.0 for cls in sentiment_classes}

    topic_sentiment = max(normalised, key=normalised.get)

    turn_counts = topic_df["predicted_label"].value_counts().to_dict()
    total = len(topic_df)

    return {
        "topic_sentiment": topic_sentiment,
        "num_turns": total,
        "total_role_weight": round(total_weight, 3),
        **{f"weighted_{cls.lower()}": round(v, 4) for cls, v in normalised.items()},
        **{f"{cls.lower()}_turns": turn_counts.get(cls, 0) for cls in sentiment_classes},
        **{f"{cls.lower()}_pct": round(turn_counts.get(cls, 0) / total * 100, 1) for cls in sentiment_classes},
    }


# ── Main entry point ──────────────────────────────────────────────────

def run_sentiment_inference(date):
    """
    Run Longformer sentiment inference on speaker turns for a given date.

    Args:
        date: "DD-MM-YYYY"

    Returns:
        dict with keys: success, date, num_turns, num_topics,
        thresholds_applied, topic_sentiments, turn_sentiments, sentiment_path
    """
    input_csv   = IN_DIR / f"speaker_turns_{date}.csv"
    output_json = OUT_DIR / f"sentiments_{date}.json"

    if output_json.exists():
        return {"success": True, "date": date,"sentiment_path": str(output_json)}

    if not input_csv.exists():
        return {"success": False, "date": date, "error": f"Input file not found: {input_csv}"}

    try:
        _load_artifacts()

        df = pd.read_csv(input_csv)
        if df.empty:
            return {"success": False, "date": date, "error": f"Input file is empty: {input_csv}"}

        df = df.copy()
        df["speaker_tier"] = df["speaker"].apply(_assign_tier)
        df["role_weight"]  = df["speaker_tier"].map(_ROLE_WEIGHTS)

        # Tokenize
        texts = df["speech_text"].fillna("").tolist()
        encoded = _tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors=None,
        )
        encoded["global_attention_mask"] = [
            [1] + [0] * (MAX_LEN - 1) for _ in encoded["input_ids"]
        ]

        dataset    = _TurnDataset(encoded)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Batched inference
        all_logits = []
        with torch.no_grad():
            for batch in dataloader:
                outputs = _model(
                    input_ids             = batch["input_ids"].to(DEVICE),
                    attention_mask        = batch["attention_mask"].to(DEVICE),
                    global_attention_mask = batch["global_attention_mask"].to(DEVICE),
                )
                all_logits.append(outputs.logits.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_probs  = torch.softmax(all_logits, dim=-1).numpy()

        # Apply tuned thresholds
        all_preds = _predict_with_thresholds(all_probs, _thresholds)
        label_map = {i: cls for i, cls in enumerate(_le.classes_)}

        df["predicted_label"]  = [label_map[p] for p in all_preds]
        df["confidence_score"] = all_probs.max(axis=1).round(4).tolist()
        for i, cls in enumerate(_le.classes_):
            df[f"prob_{cls.lower()}"] = all_probs[:, i].round(4).tolist()

        # Turn-level records
        turn_cols = [
            "sittingDate", "title", "speaker", "speaker_tier", "role_weight",
            "speech_text", "predicted_label", "confidence_score",
        ] + [f"prob_{cls.lower()}" for cls in _le.classes_]

        turn_sentiments = df[turn_cols].to_dict(orient="records")

        # Topic-level aggregation
        topic_sentiments = []
        for (sitting_date, title), group in df.groupby(["sittingDate", "title"], sort=False):
            agg = _aggregate_topic_sentiment(group, list(_le.classes_))
            topic_sentiments.append({"sittingDate": sitting_date, "title": title, **agg})

        thresholds_applied = dict(zip(_le.classes_, _thresholds))

        payload = {
            "date": date,
            "num_turns": len(df),
            "num_topics": len(topic_sentiments),
            "thresholds_applied": thresholds_applied,
            "topic_sentiments": topic_sentiments,
            "turn_sentiments": turn_sentiments,
        }

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return {
            "success": True,
            "date": date,
            "num_turns": len(df),
            "num_topics": len(topic_sentiments),
            "sentiment_path": str(output_json),
        }

    except Exception as e:
        return {"success": False, "date": date, "error": str(e)}
