# models/clustering.py
"""
Modular clustering models for sentiment pipeline:
Includes KMeans, Fuzzy C-Means (scikit-fuzzy), and KModes (optional).
Provides a unified, sklearn-like interface for integration into pipelines.
"""

from __future__ import annotations
import logging
from typing import Optional
import numpy as np
from sklearn.cluster import KMeans

# --- Optional Dependencies ---
try:
    from skfuzzy.cluster import cmeans
    from skfuzzy import cmeans_predict
    _HAS_SKFUZZY = True
except Exception:
    _HAS_SKFUZZY = False

try:
    from kmodes.kmodes import KModes
    _HAS_KMODES = True
except Exception:
    _HAS_KMODES = False

# --- Logging Setup ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ====================================================================
# KMEANS CLUSTERING
# ====================================================================

class KMeansClustering:
    """Light wrapper around sklearn.cluster.KMeans."""

    def __init__(self, n_clusters: int = 3, random_state: int = 42, **kwargs):
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, **kwargs)
        self._is_fitted = False
        logger.debug(f"KMeans initialized with {self.n_clusters} clusters.")

    def fit(self, X: np.ndarray) -> "KMeansClustering":
        X = np.asarray(X)
        self.model.fit(X)
        self._is_fitted = True
        logger.info(f"KMeans fitted on {X.shape[0]} samples.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("KMeansClustering: call fit(...) before predict(...).")
        return self.model.predict(np.asarray(X))

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.predict(X)

    def get_centroids(self) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("KMeansClustering: call fit(...) before get_centroids(...).")
        return self.model.cluster_centers_


# ====================================================================
# FUZZY C-MEANS CLUSTERING
# ====================================================================

class FuzzyCMeansClustering:
    """Wrapper around scikit-fuzzy c-means. Optional dependency."""

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        error: float = 1e-5,
        maxiter: int = 1000,
        seed: Optional[int] = 42,
    ):
        if not _HAS_SKFUZZY:
            raise ImportError(
                "scikit-fuzzy is required for FuzzyCMeansClustering. "
                "Install with `pip install scikit-fuzzy`."
            )

        self.n_clusters = int(n_clusters)
        self.m = float(m)
        self.error = float(error)
        self.maxiter = int(maxiter)
        self.seed = seed

        self.centroids: Optional[np.ndarray] = None
        self.u: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> "FuzzyCMeansClustering":
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        cntr, u, _, _, _, _, _ = cmeans(
            X.T,
            c=self.n_clusters,
            m=self.m,
            error=self.error,
            maxiter=self.maxiter,
            seed=self.seed,
        )

        self.centroids = np.asarray(cntr)
        self.u = np.asarray(u)
        self._is_fitted = True
        logger.info(f"Fuzzy C-Means fitted on {X.shape[0]} samples.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("FuzzyCMeansClustering: call fit(...) before predict(...).")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        u_pred, _, _, _, _, _ = cmeans_predict(
            X.T,
            self.centroids,
            m=self.m,
            error=self.error,
            maxiter=self.maxiter,
        )

        labels = np.argmax(u_pred, axis=0)
        return labels

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.predict(X)

    def get_centroids(self) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("FuzzyCMeansClustering: call fit(...) before get_centroids(...).")
        return self.centroids


# ====================================================================
# KMODES CLUSTERING
# ====================================================================

class KModesClustering:
    """Wrapper around kmodes.KModes. Optional dependency."""

    def __init__(self, n_clusters: int = 3, init: str = "Huang", n_init: int = 5, random_state: Optional[int] = 42, verbose: int = 0):
        if not _HAS_KMODES:
            raise ImportError("kmodes is required for KModesClustering. Install with `pip install kmodes`.")
        self.n_clusters = int(n_clusters)
        self.model = KModes(
            n_clusters=self.n_clusters,
            init=init,
            n_init=n_init,
            random_state=random_state,
            verbose=verbose,
        )
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> "KModesClustering":
        self.model.fit(X)
        self._is_fitted = True
        logger.info(f"KModes fitted on {X.shape[0]} samples.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("KModesClustering: call fit(...) before predict(...).")
        return self.model.predict(X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.predict(X)

    def get_centroids(self) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("KModesClustering: not fitted.")
        return np.asarray(self.model.cluster_centroids_)


# ====================================================================
# FACTORY FUNCTION
# ====================================================================

def create_clustering_model(method: str = "kmeans", **kwargs):
    """Factory for selecting clustering method dynamically."""
    method = method.lower()
    if method == "kmeans":
        return KMeansClustering(**kwargs)
    elif method in {"fuzzy", "cmeans"}:
        return FuzzyCMeansClustering(**kwargs)
    elif method == "kmodes":
        return KModesClustering(**kwargs)
    else:
        raise ValueError(f"Unknown clustering method: {method}")


__all__ = [
    "KMeansClustering",
    "FuzzyCMeansClustering",
    "KModesClustering",
    "create_clustering_model",
]
