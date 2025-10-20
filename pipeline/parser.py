import json
import joblib
import logging
from pathlib import Path
from typing import List, Union

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

TEXT_KEYS = ["full_text", "text", "content", "comment", "body", "message"]

def parse_json(input_json_path: str, output_pkl_path: str = "data/processed/parsed_comments.pkl") -> List[str]:
    """
    Parse JSON berisi tweet / komentar menjadi list teks mentah,
    kemudian simpan hasil ke pickle (.pkl).
    """
    input_path = Path(input_json_path)
    output_path = Path(output_pkl_path)

    if not input_path.exists():
        logging.error(f"❌ File tidak ditemukan: {input_path}")
        return []

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"❌ Gagal membaca file JSON: {e}")
        return []

    texts: List[str] = []

    # JSON bisa berupa list atau dict
    if isinstance(data, dict):
        data = [data]

    for item in data:
        if isinstance(item, dict):
            for key in TEXT_KEYS:
                if key in item and isinstance(item[key], str):
                    texts.append(item[key].strip())
                    break
        elif isinstance(item, str):
            texts.append(item.strip())

    if not texts:
        logging.warning(f"⚠️ Tidak ditemukan kolom teks dari {input_json_path}")
    else:
        logging.info(f"✅ Berhasil parse {len(texts)} teks.")
        logging.info(f"🧩 Contoh pertama: {texts[0][:80]}...")

    # Simpan hasil ke .pkl
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(texts, output_path)
        logging.info(f"💾 Disimpan ke {output_path}")
    except Exception as e:
        logging.error(f"❌ Gagal menyimpan hasil ke {output_path}: {e}")

    return texts


if __name__ == "__main__":
    parse_json("data/raw/sample.json")
