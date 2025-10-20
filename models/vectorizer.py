# models/vectorizer.py

import os
import joblib
import logging
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Vectorizer:
    def __init__(self, max_features: int = 5000, ngram_range=(1, 2)):
        """
        Wrapper untuk TF-IDF Vectorizer.
        Args:
            max_features: batas maksimal fitur (default 5000)
            ngram_range: tuple ngram, misal (1,2) artinya unigram + bigram
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.model = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            tokenizer=lambda x: x,  # karena input berupa token list
            preprocessor=lambda x: x,  # lewati preprocessor default
            token_pattern=None
        )
        self.fitted = False

    def fit(self, tokenized_docs: List[List[str]]):
        """
        Melatih TF-IDF berdasarkan token list hasil preprocessing.
        """
        logging.info("Melatih TF-IDF Vectorizer...")
        joined_docs = [" ".join(tokens) for tokens in tokenized_docs]
        self.model.fit(joined_docs)
        self.fitted = True
        logging.info("Vectorizer berhasil dilatih.")
        return self

    def transform(self, tokenized_docs: List[List[str]]) -> np.ndarray:
        """
        Mengubah token list menjadi array numerik TF-IDF.
        """
        if not self.fitted:
            raise ValueError("Vectorizer belum dilatih. Jalankan fit() dulu.")
        joined_docs = [" ".join(tokens) for tokens in tokenized_docs]
        X = self.model.transform(joined_docs)
        return X.toarray()

    def save(self, path: str):
        """
        Simpan model vectorizer ke file .pkl
        """
        if not self.fitted:
            logging.warning("Menyimpan vectorizer yang belum dilatih.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        logging.info(f"Vectorizer disimpan ke {path}")

    def load(self, path: str):
        """
        Muat model vectorizer dari file .pkl
        """
        self.model = joblib.load(path)
        self.fitted = True
        logging.info(f"Vectorizer dimuat dari {path}")
        return self


if __name__ == "__main__":
    # Jalankan pipeline kecil untuk test manual
    import joblib
    from pathlib import Path

    processed_dir = Path("../data/processed")
    tokens_path = processed_dir / "tokens.pkl"
    embeddings_path = processed_dir / "embeddings.pkl"

    if not tokens_path.exists():
        logging.error(f"Tidak ditemukan: {tokens_path}")
    else:
        tokenized_docs = joblib.load(tokens_path)
        vectorizer = Vectorizer(max_features=3000)
        vectorizer.fit(tokenized_docs)
        X = vectorizer.transform(tokenized_docs)
        joblib.dump(X, embeddings_path)
        logging.info(f"Embeddings berhasil disimpan ke: {embeddings_path}")
