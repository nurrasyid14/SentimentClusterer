# models/sentiment_mapper.py

import logging
import numpy as np
from typing import List, Optional

# Internal imports
from models.embeddings_builder import EmbeddingsBuilder
from models.vectorizer import Vectorizer

# External ML libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class SentimentEngine:
    """
    Unified Sentiment Analysis Engine.
    Modes:
        - lexicon: rule-based using sentiment dictionaries
        - ml: classical machine learning
        - deep: embeddings + neural network / transformer
    """

    LEXICON = {
        "pos": ["good", "great", "excellent", "love", "happy", "fantastic"],
        "neg": ["bad", "terrible", "hate", "sad", "awful", "worst"],
        "neu": []  # optional
    }

    def __init__(self, method: str = "lexicon"):
        self.method = method.lower()
        self.embedding_model: Optional[EmbeddingsBuilder] = None
        self.vectorizer: Optional[Vectorizer] = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

        logging.info(f"SentimentEngine initialized with method: {self.method}")

        if self.method == "deep":
            self.embedding_model = EmbeddingsBuilder(use_sbert=True)
        elif self.method == "ml":
            self.vectorizer = Vectorizer(max_features=3000)

    # ----------------------------------------
    # TRAIN / FIT
    # ----------------------------------------
    def prepare_and_train(self, texts: List[str], labels: List[int]):
        """
        Train the sentiment model (for ML or Deep Learning modes)
        """
        if self.method == "lexicon":
            logging.info("Lexicon mode does not require training.")
            return self

        # Encode labels
        y = self.label_encoder.fit_transform(labels)

        # Feature preparation
        if self.method == "ml":
            tokenized = [t.split() for t in texts]
            X = self.vectorizer.fit(tokenized).transform(tokenized)
        elif self.method == "deep":
            tokenized = [t.split() for t in texts]
            self.embedding_model.fit(tokenized)
            X = self.embedding_model.transform(tokenized)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Simple classifier for demonstration
        self.model = LogisticRegression(max_iter=500)
        self.model.fit(X, y)
        self.is_fitted = True
        logging.info(f"{self.method.upper()} sentiment model trained on {len(texts)} samples.")
        return self

    # ----------------------------------------
    # PREDICTION
    # ----------------------------------------
    def predict(self, texts: List[str]) -> List[int]:
        """
        Predict sentiment labels for given texts
        Returns: list of integers (0=Neg, 1=Neu, 2=Pos)
        """
        if self.method == "lexicon":
            return [self._lexicon_predict(t) for t in texts]

        if not self.is_fitted:
            raise RuntimeError(f"Model not fitted. Run prepare_and_train first.")

        # Feature extraction
        tokenized = [t.split() for t in texts]
        if self.method == "ml":
            X = self.vectorizer.transform(tokenized)
        elif self.method == "deep":
            X = self.embedding_model.transform(tokenized)

        y_pred = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred)

    # ----------------------------------------
    # LEXICON RULE
    # ----------------------------------------
    def _lexicon_predict(self, text: str) -> int:
        text_lower = text.lower()
        pos_score = sum(word in text_lower for word in self.LEXICON["pos"])
        neg_score = sum(word in text_lower for word in self.LEXICON["neg"])
        if pos_score > neg_score:
            return 2
        elif neg_score > pos_score:
            return 0
        else:
            return 1  # neutral

    # ----------------------------------------
    # UTILITY
    # ----------------------------------------
    def predict_proba(self, texts: List[str]) -> List[List[float]]:
        """
        Returns probability estimates for ML/Deep modes
        """
        if self.method == "lexicon":
            raise NotImplementedError("Lexicon mode does not support probabilities.")

        tokenized = [t.split() for t in texts]
        if self.method == "ml":
            X = self.vectorizer.transform(tokenized)
        elif self.method == "deep":
            X = self.embedding_model.transform(tokenized)

        return self.model.predict_proba(X)

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Return embeddings for visualization (PCA or other projection).
        For lexicon method, returns empty array.
        """
        if self.method == "lexicon":
            logging.warning("Lexicon method has no embeddings. Returning empty array.")
            return np.zeros((len(texts), 0))
        return self.embedding_model.transform(texts)
