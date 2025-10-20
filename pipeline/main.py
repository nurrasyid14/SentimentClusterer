from pathlib import Path
import json
import pickle
import pandas as pd
import numpy as np

from models.sentiment_mapper import SentimentEngine
from models.clustering import cluster_to_pkl  # atau centroids.py wrapper
from models.vectorizer import EmbeddingModel  # jika ada custom vectorizer

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")

def load_comments() -> pd.DataFrame:
    """Load semua JSON komentar dari folder data/raw"""
    all_comments = []
    for file in DATA_RAW.glob("*.json"):
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
    return pd.DataFrame(all_comments)


def run_pipeline():
    """Run full pipeline: parse → embed → cluster → sentiment summary"""
    # 1. Load comments
    df_comments = load_comments()
    if df_comments.empty:
        print("No comments found in data/raw/")
        return

    # 2. Preprocessing & embedding
    texts = df_comments["comment"].tolist()
    embedding_model = EmbeddingModel()
    embedding_model.fit(texts)
    vectors = embedding_model.transform(texts)

    # 3. Clustering
    cluster_file = DATA_PROCESSED / "cluster_results.pkl"
    df_clusters = cluster_to_pkl(vectors, method="fuzzy", n_clusters=3, save_path=cluster_file)
    
    # 4. Sentiment mapping
    sentiment_engine = SentimentEngine()
    # Training dummy (harus diganti dataset nyata / label manual)
    dummy_labels = np.random.randint(0, 3, size=len(texts))  # placeholder
    sentiment_engine.prepare_and_train(texts, dummy_labels)
    sentiment_labels = sentiment_engine.predict(texts)
    
    df_comments["sentiment_label"] = sentiment_labels

    # 5. Hitung summary per link
    summary = df_comments.groupby("link").agg(
        likes=("likes", "first"),
        comment_count=("comment", "count"),
        positive_pct=("sentiment_label", lambda x: np.mean(np.array(x)==2)*100),
        neutral_pct=("sentiment_label", lambda x: np.mean(np.array(x)==1)*100),
        negative_pct=("sentiment_label", lambda x: np.mean(np.array(x)==0)*100),
    ).reset_index()
    # Sentiment mean (0,1,2)
    summary["sentiment_mean"] = (
        summary["positive_pct"]*2 + summary["neutral_pct"]*1 + summary["negative_pct"]*0
    ) / 100
    # Majority sentiment label
    summary["sentiment_label"] = summary[["negative_pct","neutral_pct","positive_pct"]].idxmax(axis=1).map({
        "negative_pct":0, "neutral_pct":1, "positive_pct":2
    })

    # 6. Simpan CSV
    summary_file = DATA_PROCESSED / "sentiment_summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"Pipeline selesai. Cluster PKL: {cluster_file}, Sentiment CSV: {summary_file}")


if __name__ == "__main__":
    run_pipeline()
