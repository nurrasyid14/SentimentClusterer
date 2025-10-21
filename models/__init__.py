# models/__init__.py
"""
Models package for sentiment analysis.
Includes embedding builders, vectorizers, visualizers, and sentiment models.
"""

# Core utilities
from .sentiment_machine.sentiment_utils import load_json_data, evaluate_model, plot_confusion_matrix

# Embedding & Vectorization
from .embeddings_builder import EmbeddingsBuilder
from .vectorizer import Vectorizer

# Visualization
from .visualizer import Visualizer

# Sentiment models (from previous setup)
from .sentiment_machine.deeplearning import DLSentimentAnalyzer
from .sentiment_machine.ml_methods import MLSentimentAnalyzer
from .sentiment_machine.lexicon_methods import LexiconSentimentAnalyzer
from .sentiment_machine.ensemble_sentiment import EnsembleSentimentAnalyzer
from .sentiment_machine.sentiment_mapper import SentimentEngine

__all__ = [
    # Utils
    "load_json_data", "evaluate_model", "plot_confusion_matrix",

    # Embeddings & Vectorizer
    "EmbeddingsBuilder", "Vectorizer",

    # Visualization
    "Visualizer",

    # Sentiment models
    "DLSentimentAnalyzer",
    "MLSentimentAnalyzer",
    "LexiconSentimentAnalyzer",
    "EnsembleSentimentAnalyzer",
    "SentimentEngine",
    "EmbeddingModel",
    "ClassifierModel"
]
