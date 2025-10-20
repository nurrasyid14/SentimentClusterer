#  pipeline/parser.py

import json
import joblib
import logging
from pathlib import Path
from typing import List, Union

# ------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ============================================================
#  JSON Parsing Function
# ============================================================

def parse_raw_json(raw_dir: Union[str, Path], comment_key: str = "comment_text") -> List[str]:
    """
    Reads all JSON files inside raw_dir and extracts comment texts.

    Args:
        raw_dir: Directory containing raw JSON files.
        comment_key: The key name that contains comment text.

    Returns:
        A list of comment strings.
    """
    raw_dir = Path(raw_dir)
    all_comments = []

    if not raw_dir.exists() or not raw_dir.is_dir():
        logging.error(f"Input directory not found: {raw_dir}")
        return []

    json_files = list(raw_dir.glob("*.json"))
    if not json_files:
        logging.warning(f"No JSON files found in {raw_dir}")
        return []

    logging.info(f"Found {len(json_files)} JSON files in {raw_dir}")

    for file in json_files:
        try:
            with file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            # Case 1: List of dictionaries
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        text = item.get(comment_key)
                        if isinstance(text, str) and text.strip():
                            all_comments.append(text.strip())

            # Case 2: Single dictionary
            elif isinstance(data, dict):
                text = data.get(comment_key)
                if isinstance(text, str) and text.strip():
                    all_comments.append(text.strip())

            else:
                logging.warning(f"Unsupported JSON structure in {file.name}")

        except json.JSONDecodeError:
            logging.error(f"JSON format error in file: {file.name}")
        except Exception as e:
            logging.error(f"Error processing {file.name}: {e}")

    logging.info(f"Parsing completed. Extracted {len(all_comments)} comments.")
    return all_comments


# ============================================================
#  Main Runner (if standalone execution)
# ============================================================

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[1]
    RAW_DATA_DIR = BASE_DIR / "data" / "raw"
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    OUTPUT_PKL = PROCESSED_DIR / "parsed_comments.pkl"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    comments = parse_raw_json(RAW_DATA_DIR, comment_key="comment_text")

    if comments:
        joblib.dump(comments, OUTPUT_PKL)
        logging.info(f"✅ Parsed comments saved to: {OUTPUT_PKL}")
    else:
        logging.warning("⚠️ No comments extracted; output file not created.")
