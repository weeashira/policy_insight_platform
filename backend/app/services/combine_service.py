from pathlib import Path
import json

# Paths
BASE_DIR = Path(__file__).resolve().parents[3]

SUMMARY_DIR = BASE_DIR / "data" / "processed" / "summaries"
TOPIC_DIR = BASE_DIR / "data" / "processed" / "topics"
SENTIMENT_DIR = BASE_DIR / "data" / "processed" / "sentiments"
SECTOR_DIR = BASE_DIR / "data" / "processed" / "sectors"

OUT_DIR = BASE_DIR / "data" / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    

def combine_results(date):
    summary_path = SUMMARY_DIR / f"summaries_{date}.json"
    topic_path = TOPIC_DIR / f"topics_{date}.json"
    sentiment_path = SENTIMENT_DIR / f"sentiments_{date}.json"
    sector_path = SECTOR_DIR / f"sectors_{date}.json"
    output_path = OUT_DIR / f"combined_{date}.json"

    try:
        if output_path.exists():
            return {"success": True, "date": date, "combined_path": str(output_path)}
        
        if not summary_path.exists():
            return {"success": False, "date": date, "error": f"Summary file not found: {summary_path}"}
        
        if not topic_path.exists():
            return {"success": False, "date": date, "error": f"Topic file not found: {topic_path}"}
        
        if not sentiment_path.exists():
            return {"success": False, "date": date, "error": f"Sentiment file not found: {sentiment_path}"}
        
        if not sector_path.exists():
            return {"success": False, "date": date, "error": f"Sector file not found: {sector_path}"}

        summary_json = load_json(summary_path)
        topic_json = load_json(topic_path)
        sentiment_json = load_json(sentiment_path)
        sector_json = load_json(sector_path)

        summary_map = {
            row["title"]: row
            for row in summary_json.get("results", [])
        }

        topic_map = {
            row["title"]: row 
            for row in topic_json.get("results")
        }

        sentiment_map = {
            row["title"]: row
            for row in sentiment_json.get("topic_sentiments", [])
        }

        sector_map = {
            row["title"]: row 
            for row in sector_json.get("results")
        }

        combined_results = []

        for title, summary_row in summary_map.items():
            topic_row = topic_map.get(title, {})
            sentiment_row = sentiment_map.get(title, {})
            sector_row = sector_map.get(title, {})

            combined_results.append({
                "title": title,
                
                # Summary
                "summary": summary_row.get("summary"),

                # Topic model
                "meta_topic_label": topic_row.get("meta_topic_label"),
                "topic_id": topic_row.get("topic_id"),
                "topic_prob": topic_row.get("topic_probability"),

                # Sentiment
                "topic_sentiment": sentiment_row.get("topic_sentiment"),
                "sentiment_num_turns": sentiment_row.get("num_turns"),
                "sentiment_total_role_weight": sentiment_row.get("total_role_weight"),
                "sentiment_weighted_cautious": sentiment_row.get("weighted_cautious"),
                "sentiment_weighted_negative": sentiment_row.get("weighted_negative"),
                "sentiment_weighted_neutral": sentiment_row.get("weighted_neutral"),
                "sentiment_weighted_positive": sentiment_row.get("weighted_positive"),
                "sentiment_cautious_pct": sentiment_row.get("cautious_pct"),
                "sentiment_negative_pct": sentiment_row.get("negative_pct"),
                "sentiment_neutral_pct": sentiment_row.get("neutral_pct"),
                "sentiment_positive_pct": sentiment_row.get("positive_pct"),

                # Sector
                "sector_classification": sector_row["sector_classification"],
                "sector_confidence": sector_row["sector_confidence"],
                "sector_confidence_band": sector_row["sector_confidence_band"]
            })

        output = {
            "success": True,
            "date": date,
            "results": combined_results
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
            
        return {
            "success": True,
            "date": date,
            "combined_path": str(output_path),
        }

    except Exception as e:
        if output_path.exists():
            output_path.unlink()
        
        return {
            "success": False,
            "date": date,
            "error": str(e)
        }
