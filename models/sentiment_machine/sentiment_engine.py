#sentiment_mapper.py

from typing import List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ============================================================
#  Embedding Model (TF-IDF)
# ============================================================

@dataclass
class EmbeddingModel:
    max_features: Optional[int] = 5000
    ngram_range: Tuple[int, int] = (1, 2)
    vectorizer: TfidfVectorizer = field(init=False, repr=False)

    def __post_init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range
        )

    def fit(self, texts: List[str]):
        """Fit the TF-IDF vectorizer to given texts."""
        if not texts or all(t.strip() == "" for t in texts):
            raise ValueError("❌ Cannot fit embedding model: input texts are empty.")
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts into TF-IDF embeddings."""
        if not texts or all(t.strip() == "" for t in texts):
            return np.zeros((0, self.vectorizer.max_features or 1))
        X = self.vectorizer.transform(texts)
        return X.toarray()

# ============================================================
#  Classifier Model (Logistic Regression)
# ============================================================

@dataclass
class ClassifierModel:
    random_state: int = 42
    model: LogisticRegression = field(init=False, repr=False)

    def __post_init__(self):
        self.model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            multi_class="multinomial"
        )

    def fit(self, X: np.ndarray, y: List[int]):
        if X.shape[0] == 0:
            raise ValueError("❌ Cannot train classifier: no samples available.")
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] == 0:
            return np.array([])
        return self.model.predict(X)

# ============================================================
#  Sentiment Engine (Full Pipeline)
# ============================================================

@dataclass
class SentimentEngine:
    embedding_model: EmbeddingModel = field(default_factory=EmbeddingModel)
    classifier_model: ClassifierModel = field(default_factory=ClassifierModel)
    is_fitted: bool = False

    def prepare_and_train(self, texts: List[str], labels: List[int]):
        """Train embedding + classifier on a batch of texts and labels (0,1,2)."""
        if texts is None or len(texts) == 0:
            raise ValueError("❌ Cannot train SentimentEngine: texts are empty.")
        if labels is None or len(labels) == 0:
            raise ValueError("❌ Cannot train SentimentEngine: labels are empty.")
        if len(texts) != len(labels):
            raise ValueError("❌ Length mismatch: texts and labels must have same length.")

        self.embedding_model.fit(texts)
        X_emb = self.embedding_model.transform(texts)
        self.classifier_model.fit(X_emb, labels)
        self.is_fitted = True


    def predict(self, texts: List[str]) -> List[int]:
        """Predict sentiment labels (0,1,2) for a batch of texts."""
        if not self.is_fitted:
            raise RuntimeError("⚠️ SentimentEngine is not trained yet.")
        X_emb = self.embedding_model.transform(texts)
        preds = self.classifier_model.predict(X_emb)
        return [int(p) for p in preds]
