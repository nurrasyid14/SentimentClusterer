from typing import Optional
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from models.centroids import FuzzyCMeansClustering, KMeansClustering

def cluster_to_pkl(
    vectors: np.ndarray,
    method: str = "fuzzy",
    n_clusters: int = 3,
    random_state: int = 42,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Cluster vektor komentar dan simpan hasilnya ke PKL.
    
    Args:
        vectors: np.ndarray, shape (n_samples, n_features)
        method: "fuzzy" or "kmeans"
        n_clusters: jumlah cluster
        random_state: seed
        save_path: path untuk simpan cluster PKL
        
    Returns:
        DataFrame dengan kolom: ["vectors", "labels"]
    """
    if method == "fuzzy":
        cluster_model = FuzzyCMeansClustering(n_clusters=n_clusters, seed=random_state)
        cluster_model.fit(vectors)
        labels = np.argmax(cluster_model.u, axis=0)  # probabilitas -> label hard
    elif method == "kmeans":
        cluster_model = KMeansClustering(n_clusters=n_clusters, random_state=random_state)
        cluster_model.fit(vectors)
        labels = cluster_model.predict(vectors)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # Buat DataFrame
    df_clusters = pd.DataFrame({
        "vectors": list(vectors),  # simpan vektor mentah
        "labels": labels
    })

    # Simpan ke PKL jika path diberikan
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(df_clusters, f)

    return df_clusters


# Contoh penggunaan
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    vectors, _ = make_blobs(n_samples=10, n_features=5, centers=3, random_state=42)

    save_file = Path("data/processed/cluster_results.pkl")
    df = cluster_to_pkl(vectors, method="fuzzy", n_clusters=3, save_path=save_file)
    print("Cluster PKL tersimpan:", save_file)
    print(df)
