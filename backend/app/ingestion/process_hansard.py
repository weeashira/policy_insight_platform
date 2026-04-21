import json
import csv
import sys
import re
import html as html_mod
from pathlib import Path
from app.utils.date_utils import is_valid_date

class HansardProcessor:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent.parent.parent
        self.in_dir = self.base_dir / "data" / "raw"
        self.out_dir = self.base_dir / "data" / "interim"
        self.skip_speakers = {"Mr Speaker", "The Chairman"}

        self.out_dir.mkdir(parents=True, exist_ok=True)

    
    def _fix_encoding(self, text):
        try:
            return text.encode('latin-1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
        

    def _clean_text(self, t):
        t = html_mod.unescape(t)
        t = self._fix_encoding(t)                    # fix encoding issues
        t = re.sub(r'<h\d[^>]*>.*?</h\d>', '', t)    # remove header tags and content e.g. <h6>10.31 am</h6>
        t = re.sub(r'<[^>]+>', ' ', t)               # remove remaining HTML tags
        t = re.sub(r'\s+', ' ', t)                   # collapse multiple spaces into one

        # Remove leading language markers like "(In English):"
        t = re.sub(
            r'^\s*\(\s*In\s+(English|Mandarin|Malay|Tamil)\s*\)\s*[:：]?\s*',
            '',
            t,
            flags=re.IGNORECASE
        )

        # Normalise typographic capitalisation markers e.g. [W]e to We - must come before bracket removal
        t = re.sub(r'\[([A-Z])\]', r'\1', t)

        # Remove all bracketed annotations e.g. [Applause.], [Laughter.], [Please refer to...]
        t = re.sub(r'\[.*?\]', '', t)

        # Remove inline Chairman/Speaker interjections
        t = re.sub(
            r'—?\s*The Chairman\s*:.*?(?=—|$)',
            '',
            t,
            flags=re.IGNORECASE
        )

        # Remove inline speaker transitions without <strong> tags
        t = re.sub(
            r'—?\s*(?:Mr|Ms|Mrs|Miss|Dr|Prof|Assoc Prof|Senior Minister)[^:]+:',
            '',
            t
        )

        return t.strip().lstrip(':—').strip().rstrip('—').strip()
    

    def _extract_core_name(self, raw):
        name_match = re.search(
            r'((?:Mr|Ms|Mrs|Miss|Dr|Prof|Assoc Prof)\s+[A-Z][A-Za-z\'-]*(?:\s+[A-Za-z\'-]+)*?)(?:\s*[\(\,\)]|$)',
            raw
        )
        if name_match:
            return name_match.group(1).strip()
        return raw.strip()
    

    def _parse_speech_segments(self, content, segment_no):
        # Split into paragraphs and discard empty segments
        paras = re.split(r'</?p>', content)
        paras = [p.strip() for p in paras if p.strip()]

        rows = []
        current_speaker = None
        current_parts = []

        for para in paras:
            # Skip procedural text segments
            if '(proc text)' in para.lower():
                continue

            # Check if paragraph opens with a bold speaker name
            m = re.match(r'^\s*\d*\s*<strong>\s*(.*?)\s*</strong>[\s:]*(.*)', para, re.DOTALL)

            if m:
                # Before switching to a new speaker, flush previous speaker block
                if current_speaker and current_parts and current_speaker not in self.skip_speakers:
                    rows.append((current_speaker, segment_no, ' '.join(current_parts)))
                    segment_no += 1

                # Extract and clean the speaker name from inside <strong> tags
                raw_speaker = html_mod.unescape(re.sub(r'<[^>]+>', '', m.group(1)))
                raw_speaker = self._fix_encoding(re.sub(r'\s+', ' ', raw_speaker).strip())
                raw_speaker = raw_speaker.rstrip(':')
                speaker = self._extract_core_name(raw_speaker)

                current_speaker = speaker if speaker else None      
                current_parts = []

                # Capture text after </strong>
                text = self._clean_text(m.group(2))
            else:
                # Continuation paragraph: append to current speaker block
                text = self._clean_text(para)
                
            if text and current_speaker:
                current_parts.append(text)

        # Flush final speaker block
        if current_speaker and current_parts and current_speaker not in self.skip_speakers:
            rows.append((current_speaker, segment_no, ' '.join(current_parts)))
            segment_no += 1

        return rows, segment_no  
    

    def preprocess_hansard(self, date):
        """
        Transforms raw Hansard JSON (scraped parliamentary data) into speaker-level turns and 
        outputs a CSV for a given date
        """
        input_file = self.in_dir / f"raw_{date}.json"
        output_file = self.out_dir / f"speaker_turns_{date}.csv"

        if not input_file.exists():
            return {"success": False, "date": date, "error": f"Error: {input_file} does not exist"}
        
        if output_file.exists() and output_file.stat().st_size > 0:
            return {"success": True, "date": date, "csv_path": output_file}
        
        try:
            with open(input_file, "r", encoding="utf-8") as file:
                data = json.load(file)

            sitting_date = data.get("metadata", {}).get("sittingDate")
            debates = data.get("takesSectionVOList", [])

            all_rows = []
            title_segment_tracker = {}

            for debate in debates:
                title = debate.get("title")
                content = debate.get("content")
                if not content:
                    continue

                # New title starts from 1, resumed title continues from last used segment_no
                starting_segment = title_segment_tracker.get(title, 1)
                content_rows, last_segment = self._parse_speech_segments(content, starting_segment)

                # Update tracker with next available segment number for this title
                title_segment_tracker[title] = last_segment

                for speaker, segment_no, speech_text in content_rows:
                    all_rows.append({
                        "sittingDate": sitting_date,
                        "title": title,
                        "speaker": speaker,
                        "segment_no": segment_no,
                        "speech_text": speech_text
                    })
                    
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["sittingDate", "title", "speaker", "segment_no", "speech_text"], quoting=csv.QUOTE_MINIMAL)
                writer.writeheader()
                writer.writerows(all_rows)

            return {"success": True, "date": date, "csv_path": output_file}
        
        except Exception as e:
            if output_file.exists():
                output_file.unlink()
            return {"success": False, "date": date, "error": str(e)}
        

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: No date provided. Usage: python process_hansard.py DD-MM-YYYY")
        sys.exit(1)

    target_date = sys.argv[1]
    if not is_valid_date(target_date):
        print("Error: '{target_date}' is not in DD-MM-YYYY format")
        sys.exit(1)

    print("Initializing preprocessing for {target_date}...")
    processor = HansardProcessor()
    processor.preprocess_hansard(target_date)