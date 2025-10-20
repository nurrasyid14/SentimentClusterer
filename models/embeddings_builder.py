# models/embeddings_builder.py

import os
import joblib
import logging
import numpy as np
from typing import List, Union

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except ImportError:
    _HAS_SBERT = False
    logging.warning("sentence-transformers tidak ditemukan. Akan gunakan TF-IDF fallback.")

from sklearn.feature_extraction.text import TfidfVectorizer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EmbeddingsBuilder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_sbert: bool = True):
        """
        Builder untuk menghasilkan embedding dari teks/token.
        Args:
            model_name: Nama model SBERT (default: all-MiniLM-L6-v2)
            use_sbert: Gunakan SBERT jika tersedia, fallback ke TF-IDF jika tidak
        """
        self.model_name = model_name
        self.use_sbert = use_sbert and _HAS_SBERT
        self.model = None

        if self.use_sbert:
            logging.info(f"Memuat model SBERT: {model_name}")
            self.model = SentenceTransformer(model_name)
        else:
            logging.warning("SBERT tidak digunakan, fallback ke TF-IDF.")
            self.model = TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                tokenizer=lambda x: x,
                preprocessor=lambda x: x,
                token_pattern=None
            )
            self.fitted = False

    def fit(self, tokenized_docs: List[List[str]]):
        """
        Untuk TF-IDF: Latih model. Untuk SBERT: tidak perlu fit.
        """
        if not self.use_sbert:
            joined_docs = [" ".join(tokens) for tokens in tokenized_docs]
            self.model.fit(joined_docs)
            self.fitted = True
            logging.info("TF-IDF Embedding Builder dilatih.")
        else:
            logging.info("SBERT tidak perlu dilatih.")
        return self

    def transform(self, tokenized_docs: List[List[str]]) -> np.ndarray:
        """
        Ubah token list menjadi embedding vektor.
        """
        logging.info("Mengubah dokumen menjadi embedding...")
        joined_docs = [" ".join(tokens) for tokens in tokenized_docs]

        if self.use_sbert:
            embeddings = self.model.encode(joined_docs, show_progress_bar=True)
        else:
            if not getattr(self, "fitted", False):
                raise ValueError("TF-IDF belum dilatih. Jalankan fit() terlebih dahulu.")
            embeddings = self.model.transform(joined_docs).toarray()

        logging.info(f"Embedding selesai. Shape: {embeddings.shape}")
        return embeddings

    def save(self, path: str):
        """
        Simpan embedding builder (TF-IDF saja, SBERT tidak perlu disimpan).
        """
        if not self.use_sbert:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.model, path)
            logging.info(f"TF-IDF Embedding model disimpan ke {path}")
        else:
            logging.info("SBERT tidak perlu disimpan (model sudah tersedia online).")

    def load(self, path: str):
        """
        Muat kembali model TF-IDF.
        """
        if not self.use_sbert:
            self.model = joblib.load(path)
            self.fitted = True
            logging.info(f"Model TF-IDF dimuat dari {path}")
        return self


if __name__ == "__main__":
    from pathlib import Path

    processed_dir = Path("../data/processed")
    tokens_path = processed_dir / "tokens.pkl"
    embeddings_path = processed_dir / "embeddings.pkl"

    if not tokens_path.exists():
        logging.error(f"Tidak ditemukan: {tokens_path}")
    else:
        tokenized_docs = joblib.load(tokens_path)

        # Gunakan SBERT jika tersedia
        builder = EmbeddingsBuilder(use_sbert=True)
        if not builder.use_sbert:
            builder.fit(tokenized_docs)

        embeddings = builder.transform(tokenized_docs)
        joblib.dump(embeddings, embeddings_path)
        logging.info(f"Embeddings disimpan ke {embeddings_path}")
