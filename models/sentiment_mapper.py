from typing import List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

@dataclass
class EmbeddingModel:
    max_features: Optional[int] = 5000
    ngram_range: Tuple[int, int] = (1,2)
    vectorizer: TfidfVectorizer = field(init=False, repr=False)

    def __post_init__(self):
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=self.ngram_range)

    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return X.toarray()

@dataclass
class ClassifierModel:
    random_state: int = 42
    model: LogisticRegression = field(init=False, repr=False)

    def __post_init__(self):
        self.model = LogisticRegression(random_state=self.random_state, max_iter=1000, multi_class="multinomial")
    
    def fit(self, X: np.ndarray, y: List[int]):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

@dataclass
class SentimentEngine:
    embedding_model: EmbeddingModel = field(default_factory=EmbeddingModel)
    classifier_model: ClassifierModel = field(default_factory=ClassifierModel)
    is_fitted: bool = False

    def prepare_and_train(self, texts: List[str], labels: List[int]):
        """Train embedding + classifier on a batch of texts and labels (0,1,2)."""
        self.embedding_model.fit(texts)
        X_emb = self.embedding_model.transform(texts)
        self.classifier_model.fit(X_emb, labels)
        self.is_fitted = True

    def predict(self, texts: List[str]) -> List[int]:
        """Predict sentiment labels (0,1,2) for a batch of texts."""
        X_emb = self.embedding_model.transform(texts)
        preds = self.classifier_model.predict(X_emb)
        return [int(p) for p in preds]
