# models/sentiment_machine/__init__.py
"""
Models package for sentiment analysis.
Includes Deep Learning, ML, Lexicon, Ensemble, and utility modules.
"""

from .deeplearning import DLSentimentAnalyzer
from .ml_methods import MLSentimentAnalyzer
from .lexicon_methods import LexiconSentimentAnalyzer
from .ensemble_sentiment import EnsembleSentimentAnalyzer
from .sentiment_mapper import SentimentEngine
from .sentiment_utils import load_json_data, evaluate_model, plot_confusion_matrix

__all__ = [
    "DLSentimentAnalyzer",
    "MLSentimentAnalyzer",
    "LexiconSentimentAnalyzer",
    "EnsembleSentimentAnalyzer",
    "SentimentEngine",
    "EmbeddingModel",
    "ClassifierModel",
    "load_json_data",
    "evaluate_model",
    "plot_confusion_matrix"
]
