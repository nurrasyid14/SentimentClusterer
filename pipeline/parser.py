import json
import joblib
import logging
from pathlib import Path
from typing import List, Union, Optional

# -------------------------------------------------
# Logging Configuration
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------------------------
# JSON Parser Class
# -------------------------------------------------
class JSONParser:
    """
    Parser untuk file JSON berisi tweet, komentar, atau dokumen teks.
    Mengambil kolom teks utama lalu menyimpannya sebagai .pkl.
    """

    TEXT_KEYS = ["full_text", "text", "content", "comment", "body", "message"]

    def __init__(self, input_json: Union[str, Path], output_pkl: Optional[Union[str, Path]] = None):
        self.input_path = Path(input_json)
        self.output_path = Path(output_pkl) if output_pkl else Path("data/processed/parsed_comments.pkl")

    def _load_json(self) -> Union[List, dict, None]:
        """Membaca file JSON mentah."""
        if not self.input_path.exists():
            logging.error(f"❌ File tidak ditemukan: {self.input_path}")
            return None

        try:
            with open(self.input_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"❌ Gagal membaca JSON: {e}")
            return None

    def _extract_texts(self, data: Union[List, dict]) -> List[str]:
        """Menarik teks dari berbagai kemungkinan struktur JSON."""
        texts: List[str] = []

        # JSON bisa berupa dict tunggal atau list of dicts
        if isinstance(data, dict):
            data = [data]

        for item in data:
            if isinstance(item, dict):
                for key in self.TEXT_KEYS:
                    if key in item and isinstance(item[key], str):
                        texts.append(item[key].strip())
                        break
            elif isinstance(item, str):
                texts.append(item.strip())

        return texts

    def parse(self) -> List[str]:
        """Menjalankan parsing penuh (baca JSON → ekstrak teks → simpan pkl)."""
        data = self._load_json()
        if data is None:
            return []

        texts = self._extract_texts(data)

        if not texts:
            logging.warning(f"⚠️ Tidak ditemukan kolom teks dari {self.input_path}")
        else:
            logging.info(f"✅ Berhasil parse {len(texts)} teks.")
            logging.info(f"🧩 Contoh pertama: {texts[0][:80]}...")

        # Simpan hasil ke .pkl
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(texts, self.output_path)
            logging.info(f"💾 Disimpan ke {self.output_path}")
        except Exception as e:
            logging.error(f"❌ Gagal menyimpan hasil ke {self.output_path}: {e}")

        return texts


# -------------------------------------------------
# Script Entry Point (manual run)
# -------------------------------------------------
if __name__ == "__main__":
    parser = JSONParser("data/raw/sample.json")
    parser.parse()
