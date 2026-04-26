from pathlib import Path
import joblib
import pandas as pd
import json

# Paths 
BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "backend" / "models" / "sector_classification" / "sector_classifier_unweighted.joblib"

IN_DIR = BASE_DIR / "data" / "interim"
OUT_DIR = BASE_DIR / "data" / "processed" / "sectors"
OUT_DIR.mkdir(parents=True, exist_ok=True)


_sector_model = None


def load_sector_model():
    global _sector_model

    if _sector_model is None:
        _sector_model = joblib.load(MODEL_PATH)
    
    return _sector_model


def prepare_sector_input(df):
    # Keep only first 2 speaker turns per title
    df_top2 = df[df["segment_no"] <= 2].copy()

    # Combine first 2 turns into one row per debate
    df_combined = (
        df_top2
        .groupby(["sittingDate", "title"])["speech_text"]
        .apply(lambda x: " ".join(x.dropna().astype(str)))
        .reset_index()
    )

    df_combined["combined_text"] = (
        df_combined["title"].astype(str)
        + " [SEP] "
        + df_combined["speech_text"].astype(str)
    )

    return df_combined


def get_confidence_band(score):
    if score < 0.16:
        return "Low confidence / Needs Review"
    elif score < 0.3:
        return "Medium confidence"
    else:
        return "High confidence"


def run_sector_classification(date):
    input_csv = IN_DIR / f"speaker_turns_{date}.csv"
    output_json = OUT_DIR / f"sectors_{date}.json"

    try:
        if output_json.exists():
            return {
                "success": True,
                "date": date,
                "sector_path": str(output_json)
            }
        
        if not input_csv.exists():
            return {
                "success": False,
                "date": date,
                "error": f"Input file not found: {input_csv}",
            }
        
        df = pd.read_csv(input_csv) 
        df_combined = prepare_sector_input(df)

        model = load_sector_model()

        probs = model.predict_proba(df_combined["combined_text"])
        preds = model.classes_[probs.argmax(axis=1)]
        confidences = probs.max(axis=1)

        df_combined["sector_classification"] = preds
        df_combined["sector_confidence"] = confidences
        df_combined["sector_confidence_band"] = df_combined["sector_confidence"].apply(get_confidence_band)

        results = []
        for _, row in df_combined.iterrows():
            results.append({
                "title": row["title"],
                "sector_classification": str(row["sector_classification"]),
                "sector_confidence": round(float(row["sector_confidence"]), 4),
                "sector_confidence_band": row["sector_confidence_band"]
            })

        output = {
            "success": True,
            "date": date,
            "results": results
        }

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        return {
            "success": True,
            "date": date,
            "sector_path": str(output_json)
        }

    except Exception as e:
        if output_json.exists():
            output_json.unlink()
        
        return {
            "success": False,
            "date": date,
            "error": str(e)
        }