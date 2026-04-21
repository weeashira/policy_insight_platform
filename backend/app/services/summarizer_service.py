from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import json
import ollama

# Paths
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
IN_DIR = DATA_DIR / "interim"
OUT_DIR = DATA_DIR / "processed" / "summaries"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Config
SAFE_LIMIT = 8192
SINGLE_PASS_INPUT_BUDGET = 5200  # rough token estimate threshold
SINGLE_PRED_LIMIT = 220
CHUNK_PRED_LIMIT = 180
MULTI_PRED_LIMIT = 240

MODEL_NAME = "qwen3.5:9b"

# System Prompts
sys_msg_single = """
You are writing a dashboard summary of a Singapore parliamentary transcript for business users.

Goal:
Produce a concise, factual policy summary that helps a reader quickly decide whether the item is relevant to them.

Rules:
- Focus on the policy signal first: the key measure, clarification, timeline, eligibility rule, statistic, constraint, or stated action.
- Preserve material facts such as numbers, dates, agencies, programme names, laws, eligibility conditions, and operational constraints.
- If the response is purely factual and directly answers the question without ambiguity, keep the summary very brief and do not add inferred gaps or additional commentary.
- If part of the original question was not answered, not tracked, or not stated, say so briefly.
- Do not add facts, conclusions, motives, sentiment, or implications not explicitly stated in the transcript.
- Use neutral language.
- Do not begin with "Mr X asked" unless necessary for clarity.
- Use attribution sparingly. Prefer "The Minister said..." only where needed.
- Combine related measures into compact lists instead of repeating similar points.
- Omit procedural details unless they change interpretation.
- Write 2 to 4 sentences in one paragraph.
- Aim for about 80 to 130 words.
- Output only the summary paragraph.
"""

sys_msg_extract = """
You are extracting factual notes from one chunk of a Singapore parliamentary transcript for later dashboard summarization.

Goal:
Capture only the details needed to build a concise policy summary.

Rules:
- Extract only what is explicitly stated in the chunk.
- Keep notes brief and information-dense.
- Preserve material numbers, dates, agencies, programme names, laws, eligibility conditions, timelines, operational constraints, and concrete measures.
- Identify whether a point is a question, answer, clarification, or procedural remark only if that helps interpretation.
- If the chunk shows that information was not provided, not tracked, or not answered, note that explicitly.
- Omit repetition, rhetorical phrasing, and minor procedural details.
- Do not infer motives, conclusions, policy implications, or sentiment.
- Output only the requested bullet structure.
"""

sys_msg_multi = """
You are writing a dashboard summary of a Singapore parliamentary debate from extracted notes.

Goal:
Produce a concise, factual policy summary for business users that highlights the most relevant signal from the debate.

Rules:
- Start with the key policy measure, clarification, timeline, eligibility rule, statistic, or stated action.
- Cover the main issue raised and the main response given, but do not waste words on parliamentary framing.
- Preserve material facts such as numbers, dates, agencies, programme names, laws, eligibility conditions, and operational constraints.
- If the response is purely factual and directly answers the question without ambiguity, keep the summary very brief and do not add inferred gaps or additional commentary.
- If the notes show that part of the question was not answered, not tracked, or not stated, say so briefly.
- Use neutral language.
- Do not add conclusions, motives, sentiment, implications, or policy analysis not present in the notes.
- Use attribution sparingly and only where it improves clarity.
- Combine related measures or responses into compact lists.
- Omit repetition, procedural details, and low-value restatement.
- Write 2 to 4 sentences in one paragraph.
- Aim for about 90 to 140 words.
- Output only the summary paragraph.
"""

# Helpers
def word_count(text):
    return len(text.split()) if text else 0


def estimate_tokens(text):
    return len(text) // 4 if text else 0


def build_target_word_count(source_text, min_words=80, max_words=130, ratio=0.45):
    source_words = word_count(source_text)
    if source_words == 0:
        return min_words
    return min(max_words, max(min_words, int(source_words * ratio)))


def preprocess_debates(df):
    df = df.copy()

    # Ensure chronological sorting
    df["sort_date"] = pd.to_datetime(df["sittingDate"], format="%d-%m-%Y", errors="coerce")
    df = df.sort_values(["title", "sort_date", "segment_no"])

    debates = {}

    for (title, date_str), group in df.groupby(["title", "sittingDate"], sort=False):
        turns = "\n\n".join(
            f"{str(row['speaker']).upper()}: {str(row['speech_text']).strip()}"
            for _, row in group.iterrows()
            if pd.notna(row["speech_text"])
        )

        debates[title] = turns

    return debates


def get_text_chunks(text, chunk_size=18000, min_chunk_ratio=0.6):
    if not text or not text.strip():
        return []

    text = text.strip()
    chunks = []

    while len(text) > chunk_size:
        snip_idx = text.rfind("\n\n", 0, chunk_size)

        if snip_idx == -1:
            snip_idx = text.rfind("\n", 0, chunk_size)

        if snip_idx == -1:
            snip_idx = text.rfind(". ", 0, chunk_size)
            
        if snip_idx == -1 or snip_idx < int(chunk_size * min_chunk_ratio):
            snip_idx = chunk_size

        chunk = text[:snip_idx].strip()
        if chunk:
            chunks.append(chunk)

        text = text[snip_idx:].strip()

    if text:
        chunks.append(text)

    return chunks


def call_qwen(prompt, num_predict, system_msg="You are a helpful assistant.", context=8192):
    try:
        response = ollama.generate(
            model=MODEL_NAME,
            system=system_msg,
            prompt=prompt,
            stream=False,
            think=False,
            options={
                "num_ctx": context,
                "temperature": 0.0,
                "num_predict": num_predict,
                "top_p": 1.0,
                "repeat_penalty": 1.05,
                "stop": ["<|im_start|>", "<|im_end|>"]
            }
        )
        return {
            "success": True,
            "summary": response["response"].strip()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# Summarization
def summarize_debate(title, transcript):
    if not transcript or not transcript.strip():
        raise ValueError("No transcript provided.")

    est_tokens = estimate_tokens(transcript)

    # Single-pass path
    if est_tokens <= SINGLE_PASS_INPUT_BUDGET:
        target_words = build_target_word_count(
            transcript,
            min_words=80,
            max_words=130,
            ratio=0.45
        )

        prompt = f"""
        DEBATE TITLE:
        {title}

        TASK:
        Write one dashboard-ready policy summary.
        Target length: about {target_words} words, clearly shorter than the source.

        TRANSCRIPT:
        {transcript}

        SUMMARY:
        """
        summary = call_qwen(
            prompt=prompt,
            num_predict=SINGLE_PRED_LIMIT,
            system_msg=sys_msg_single,
            context=SAFE_LIMIT
        )

        if not summary.get("success"):
            raise Exception(summary.get("error"))

        return summary.get("summary")

    # Multi-pass path
    chunks = get_text_chunks(transcript)
    partial_notes = []

    for chunk in chunks:
        chunk_prompt = f"""
        DEBATE TITLE:
        {title}

        TASK:
        Extract only the key facts needed for a concise dashboard summary.

        TRANSCRIPT CHUNK:
        {chunk}

        Return bullets in this format:
        - Main issue or question:
        - Direct answer or position stated:
        - Key facts / figures / dates:
        - Measures, rules, programmes, or actions mentioned:
        - Missing / not stated / not tracked:
        """
        notes = call_qwen(
            prompt=chunk_prompt,
            num_predict=CHUNK_PRED_LIMIT,
            system_msg=sys_msg_extract,
            context=SAFE_LIMIT
        )

        if notes.get("success"):
            partial_notes.append(notes.get("summary"))

    if not partial_notes:
        raise ValueError("No extraction notes generated from transcript chunks.")

    combined_text = "\n\n".join(partial_notes)

    final_target_words = build_target_word_count(
        transcript,
        min_words=90,
        max_words=140,
        ratio=0.40
    )

    final_prompt = f"""
    DEBATE TITLE:
    {title}

    TASK:
    Write one dashboard-ready policy summary based only on the extracted notes below.
    Target length: about {final_target_words} words, clearly shorter than the source.

    EXTRACTED NOTES:
    {combined_text}

    SUMMARY:
    """

    summary = call_qwen(
        prompt=final_prompt,
        num_predict=MULTI_PRED_LIMIT,
        system_msg=sys_msg_multi,
        context=SAFE_LIMIT
    )

    if not summary.get("success"):
        raise Exception(summary.get("error"))

    return summary.get("summary")


# JSON checkpoint handling
def load_existing_results(output_json, date):
    if output_json.exists() and output_json.stat().st_size > 0:
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            results_dict = {}
            for item in data.get("results", []):
                title = item.get("title")
                if title:
                    results_dict[title] = item

            return {
                "date": data.get("date", date),
                "results": results_dict
            }
        except Exception:
            pass

    return {
        "date": date,
        "results": {}
    }


def save_results(output_json, data):
    payload = {
        "date": data["date"],
        "results": list(data["results"].values())
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def get_completed_titles(data):
    return {
        title
        for title, item in data.get("results", {}).items()
        if item.get("success") is True
    }


# Per-debate execution
def summarize_one(title, transcript):
    try:
        summary = summarize_debate(title, transcript)

        if not summary:
            raise ValueError("Empty summary returned.")

        return {
            "title": title,
            "summary": summary,
            "success": True
        }

    except Exception as e:
        return {
            "title": title,
            "success": False,
            "error": str(e)
        }


# Main entry point
def summarize_all_debates(date, max_workers=2):
    input_csv = IN_DIR / f"speaker_turns_{date}.csv"
    output_json = OUT_DIR / f"summaries_{date}.json"

    if not input_csv.exists():
        return {
            "success": False,
            "date": date,
            "error": f"Input file not found: {input_csv}"
        }

    try:
        df = pd.read_csv(input_csv)

        if df.empty:
            return {
                "success": False,
                "date": date,
                "error": f"Input file is empty: {input_csv}"
            }

        debates = preprocess_debates(df)

        # Resume from checkpoint if JSON already exists
        results_data = load_existing_results(output_json, date)
        completed_titles = get_completed_titles(results_data)

        remaining = [
            (title, transcript)
            for title, transcript in debates.items()
            if title not in completed_titles
        ]

        print(
            f"Total debates: {len(debates)} | "
            f"Completed: {len(completed_titles)} | "
            f"Remaining: {len(remaining)}"
        )

        if not remaining:
            return {
                "success": True,
                "date": date,
                "summary_path": str(output_json),
            }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(summarize_one, title, transcript)
                for title, transcript in remaining
            ]

            for i, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                title = result["title"]
                results_data["results"][title] = result

                # Save after each finished debate
                save_results(output_json, results_data)

                if i % 10 == 0:
                    print(f"Summarized [{i}/{len(remaining)}]")

        success_count = sum(1 for res in results_data["results"].values() if res.get("success") is True)
        total_count = len(results_data["results"])

        if success_count == 0 and total_count > 0:
            return {
                "success": False,
                "date": date,
                "error": f"{total_count} summarization attempts failed. Check JSON for details."
            }

        return {
            "success": True,
            "date": date,
            "summary_path": str(output_json),
        }

    except Exception as e:
        return {
            "success": False,
            "date": date,
            "error": str(e)
        }