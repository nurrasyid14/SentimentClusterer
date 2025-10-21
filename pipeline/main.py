# pipeline/main.py

import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from pipeline.preprocess import clean_and_tokenize_text
from models.vectorizer import Vectorizer
from models.sentiment_machine import SentimentEngine

# -------------------------------------------------
# Logging Configuration
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")


# =================================================
# PIPELINE CLASS
# =================================================
class SentimentPipeline:
    """
    Full sentiment analysis pipeline:
      1. Load comments from JSON
      2. Clean + tokenize
      3. Vectorize (TF-IDF or embedding)
      4. Run sentiment classification
      5. Reduce dimensionality (PCA → x, y)
      6. Export CSV ready for visualization
    """

    def __init__(self, raw_dir: Path = DATA_RAW, processed_dir: Path = DATA_PROCESSED):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # 1️⃣ LOAD RAW COMMENTS
    # -------------------------------------------------
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
                    if isinstance(c, dict) and "text" in c:
                        text = c["text"]
                    else:
                        text = str(c)
                    all_comments.append({"link": link, "likes": likes, "comment": text})

            except Exception as e:
                logging.error(f"❌ Failed to read {file}: {e}")

        df = pd.DataFrame(all_comments)
        logging.info(f"📥 Loaded {len(df)} comments from {self.raw_dir}")
        return df

    # -------------------------------------------------
    # 2️⃣ PREPROCESS TEXTS
    # -------------------------------------------------
    def preprocess_comments(self, texts: List[str]) -> List[List[str]]:
        """Clean, normalize, and tokenize all comments"""
        tokenized = [clean_and_tokenize_text(t) for t in texts if isinstance(t, str) and t.strip()]
        logging.info(f"🧹 Preprocessed {len(tokenized)} comments")
        return tokenized

    # -------------------------------------------------
    # 3️⃣ VECTORIZE + PCA PROJECTION
    # -------------------------------------------------
    def vectorize_and_project(self, tokenized_texts: List[List[str]]) -> pd.DataFrame:
        """Vectorize comments and reduce to 2D using PCA for visualization"""
        vectorizer = Vectorizer(max_features=3000)
        vectorizer.fit(tokenized_texts)
        vectors = vectorizer.transform(tokenized_texts)
        logging.info(f"📊 Vectorized comments. Shape: {vectors.shape}")

        # PCA to 2D (for scatter visualization)
        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(vectors.toarray() if hasattr(vectors, "toarray") else vectors)
        df_pca = pd.DataFrame(reduced, columns=["x", "y"])
        logging.info("📉 PCA dimensionality reduction completed.")
        return df_pca

    # -------------------------------------------------
    # 4️⃣ SENTIMENT CLASSIFICATION
    # -------------------------------------------------
    def analyze_sentiment(self, texts: List[str]) -> List[int]:
        """Run sentiment engine to predict polarity labels"""
        sentiment_engine = SentimentEngine()
        sentiment_labels = sentiment_engine.predict(texts)
        logging.info("🧠 Sentiment analysis completed.")
        return sentiment_labels

    # -------------------------------------------------
    # 5️⃣ RUN FULL PIPELINE
    # -------------------------------------------------
    def run(self):
        # Step 1: Load data
        df_comments = self.load_comments()
        if df_comments.empty:
            logging.warning("⚠️ No comments found. Pipeline terminated.")
            return

        # Step 2: Preprocessing
        texts = df_comments["comment"].tolist()
        tokenized_texts = self.preprocess_comments(texts)

        # Step 3: Vectorization + PCA projection
        df_pca = self.vectorize_and_project(tokenized_texts)
        df_comments = pd.concat([df_comments.reset_index(drop=True), df_pca], axis=1)

        # Step 4: Sentiment classification
        sentiment_labels = self.analyze_sentiment(texts)
        df_comments["sentiment_label"] = sentiment_labels

        # Step 5: Summarize per link
        summary = (
            df_comments.groupby("link")
            .agg(
                likes=("likes", "first"),
                comment_count=("comment", "count"),
                positive_pct=("sentiment_label", lambda x: np.mean(np.array(x) == 2) * 100),
                neutral_pct=("sentiment_label", lambda x: np.mean(np.array(x) == 1) * 100),
                negative_pct=("sentiment_label", lambda x: np.mean(np.array(x) == 0) * 100),
            )
            .reset_index()
        )

        # Weighted mean sentiment
        summary["sentiment_mean"] = (
            summary["positive_pct"] * 2 + summary["neutral_pct"] * 1
        ) / 100

        # Majority label
        summary["sentiment_label"] = summary[
            ["negative_pct", "neutral_pct", "positive_pct"]
        ].idxmax(axis=1).map({
            "negative_pct": 0,
            "neutral_pct": 1,
            "positive_pct": 2
        })

        # Step 6: Save outputs
        comments_file = self.processed_dir / "comments_with_sentiment.csv"
        summary_file = self.processed_dir / "sentiment_summary.csv"
        df_comments.to_csv(comments_file, index=False)
        summary.to_csv(summary_file, index=False)

        logging.info(f"✅ Pipeline completed successfully.")
        logging.info(f"💾 Comments saved to: {comments_file}")
        logging.info(f"💾 Summary saved to: {summary_file}")


# =================================================
# ENTRY POINT
# =================================================
if __name__ == "__main__":
    pipeline = SentimentPipeline()
    pipeline.run()
