import sys

# Import custom logic
from app.ingestion.fetch_hansard import fetch_hansard_data
from app.ingestion.process_hansard import HansardProcessor
from app.utils.date_utils import is_valid_date
from app.services.summarizer_service import summarize_all_debates
from app.services.topic_model_service import run_topic_model

def run_pipeline(date):
    print(f"--- Starting Pipeline for {date} ---")

    # Step 1: Fetch hansard
    print(f"🚀 Step 1: Fetching hansard data for {date}...")
    fetch_success = fetch_hansard_data(date)
    if not fetch_success["success"]:
        print(f"❌ Pipeline halted: Fetching failed for {date}. Error: {fetch_success['error']}")
        return False
    else:
        print(f"✅ Step 1: Fetch hansard completed. File saved at --> {fetch_success['file_path']}")
    
    # Step 2: Process hansard into speaker turns
    print(f"🚀 Step 2: Processing hansard data for {date}...")
    processor = HansardProcessor()
    process_success = processor.preprocess_hansard(date)
    if not process_success["success"]:
        print(f"❌ Pipeline halted: Preprocessing failed for {date}. Error: {process_success['error']}")
        return False
    else:
        print(f"✅ Step 2: Process hansard completed for {date}. File saved at --> {process_success['csv_path']}")

    # Step 3: Assign meta-topic labels to debates
    print(f"🚀 Step 3: Starting run for topic modelling module for {date}...")
    topic_success = run_topic_model(date)
    if not topic_success["success"]:
        print(f"❌ Pipeline halted: Topic assignment failed for {date}. Error: {topic_success["error"]}")
        return False
    else:
        print(f"✅ Step 3: Topic assignment completed for {date}. File saved at --> {topic_success['topic_path']}")
    
    # Sector Classification

    # Sentiment Analysis

    # Step 6: Summarize
    print(f"🚀 Step 6: Starting run for summarization module for {date}...")
    summary_success = summarize_all_debates(date, max_workers=4)
    if not summary_success["success"]:
        print(f"❌ Pipeline halted: Summarization failed for {date}. Error: {summary_success["error"]}")
        return False
    else:
        print(f"✅ Step 6: Summarization completed for {date}. File saved at --> {summary_success['summary_path']}")

    # Combine 


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: No date provided. Usage: python run_pipeline.py DD-MM-YYYY")
        sys.exit(1)

    target_date = sys.argv[1]
    if not is_valid_date(target_date):
        print("Error: '{target_date}' is not in DD-MM-YYYY format")
        sys.exit(1)

    run_pipeline(target_date)

    