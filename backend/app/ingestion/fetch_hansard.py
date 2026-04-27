import json
import requests
import sys
from pathlib import Path
from app.utils.date_utils import is_valid_date

# Config
URL = "https://sprs.parl.gov.sg/search/getHansardReport/"
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def fetch_hansard_data(date):
    """Fetches data for a single date"""
    file_path = DATA_DIR / f"raw_{date}.json"

    if file_path.exists() and file_path.stat().st_size > 0:
        return {
            "success": True,
            "date": date,
            "file_path": str(file_path),
        }

    with requests.Session() as session:
        try:
            params = {"sittingDate": date}
            response = session.get(URL, params=params, headers={"Accept": "application/json"}, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return {
                "success": True,
                "date": date,
                "file_path": str(file_path),
            }

        except requests.exceptions.HTTPError as e:
            return {"success": False, "date": date, "error": f"HTTP {e.response.status_code}"}
        except requests.exceptions.Timeout:
            return {"success": False, "date": date, "error": "request_timeout"}
        except requests.exceptions.ConnectionError:
            return {"success": False, "date": date, "error": "connection_error"}
        except Exception as e:
            return {"success": False, "date": date, "error": str(e)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: No date provided. Usage: python fetch_hansard.py DD-MM-YYYY")
        sys.exit(1)

    target_date = sys.argv[1]

    if not is_valid_date(target_date):
        print(f"Error: '{target_date}' is not in DD-MM-YYYY format")
        sys.exit(1)

    print(f"Initializing fetch for {target_date}...")
    fetch_hansard_data(target_date)