# pipeline/main.py

from pathlib import Path
import json
import logging
import numpy as np
import pandas as pd
from typing import List

from models.sentiment_machine import SentimentEngine
from models.vectorizer import Vectorizer  # Custom TF-IDF / embedding wrapper
from models.centroids import cluster_to_pkl  # Assuming clustering wrapper is here
from pipeline.preprocess import clean_and_tokenize_text

# -------------------------------------------------
# Logging Configuration
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")


class SentimentPipeline:
    def __init__(self, raw_dir: Path = DATA_RAW, processed_dir: Path = DATA_PROCESSED):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_comments(self) -> pd.DataFrame:
        """Load semua JSON komentar dari folder data/raw"""
        all_comments = []
        for file in self.raw_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    link = data.get("link", "")
                    likes = data.get("likes", 0)
                    comments = data.get("comments", [])
                    for c in comments:
                        all_comments.append({
                            "link": link,
                            "likes": likes,
                            "comment": c
                        })
            except Exception as e:
                logging.error(f"Failed to read {file}: {e}")

        df = pd.DataFrame(all_comments)
        logging.info(f"Loaded {len(df)} comments from {self.raw_dir}")
        return df

    def preprocess_comments(self, texts: List[str]) -> List[List[str]]:
        """Clean, normalize, and tokenize all comments"""
        tokenized = [clean_and_tokenize_text(t) for t in texts if t.strip()]
        logging.info(f"Preprocessed {len(tokenized)} comments")
        return tokenized

    def run(self):
        # 1. Load comments
        df_comments = self.load_comments()
        if df_comments.empty:
            logging.warning("No comments found. Pipeline terminated.")
            return

        # 2. Preprocessing
        texts = df_comments["comment"].tolist()
        tokenized_texts = self.preprocess_comments(texts)

        # 3. Embedding / Vectorization
        vectorizer = Vectorizer(max_features=3000)
        vectorizer.fit(tokenized_texts)
        vectors = vectorizer.transform(tokenized_texts)
        logging.info(f"Vectorized comments. Shape: {vectors.shape}")

        # 4. Clustering
        cluster_file = self.processed_dir / "cluster_results.pkl"
        df_clusters = cluster_to_pkl(
            vectors, method="fuzzy", n_clusters=3, save_path=cluster_file
        )
        logging.info(f"Clustering completed. Results saved to {cluster_file}")

        # 5. Sentiment Mapping
        sentiment_engine = SentimentEngine()
        # Placeholder dummy labels; replace with real training dataset
        dummy_labels = np.random.randint(0, 3, size=len(texts))
        sentiment_engine.prepare_and_train(texts, dummy_labels)
        sentiment_labels = sentiment_engine.predict(texts)
        df_comments["sentiment_label"] = sentiment_labels

        # 6. Summarize sentiment per link
        summary = df_comments.groupby("link").agg(
            likes=("likes", "first"),
            comment_count=("comment", "count"),
            positive_pct=("sentiment_label", lambda x: np.mean(np.array(x) == 2) * 100),
            neutral_pct=("sentiment_label", lambda x: np.mean(np.array(x) == 1) * 100),
            negative_pct=("sentiment_label", lambda x: np.mean(np.array(x) == 0) * 100),
        ).reset_index()

        # Sentiment mean (0=neg,1=neu,2=pos)
        summary["sentiment_mean"] = (
            summary["positive_pct"] * 2 + summary["neutral_pct"] * 1
        ) / 100

        # Majority sentiment label
        summary["sentiment_label"] = summary[["negative_pct", "neutral_pct", "positive_pct"]].idxmax(axis=1).map({
            "negative_pct": 0,
            "neutral_pct": 1,
            "positive_pct": 2
        })

        # 7. Save summary CSV
        summary_file = self.processed_dir / "sentiment_summary.csv"
        summary.to_csv(summary_file, index=False)
        logging.info(f"Pipeline finished successfully. Summary saved to {summary_file}")


if __name__ == "__main__":
    pipeline = SentimentPipeline()
    pipeline.run()
