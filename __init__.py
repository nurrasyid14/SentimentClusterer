"""
SentimentClusterer
===================

Unified framework for end-to-end social media sentiment analysis.

Project structure:
- app: Streamlit interface (UI layer)
- models: Machine Learning, Deep Learning, and Lexicon sentiment modules
- pipeline: Text parsing, preprocessing, and analysis workflow
- data: Raw, processed, and log directories

This package initializer connects all submodules for clean imports.
"""

from . import app
from . import models
from . import pipeline
from . import data

__all__ = ["app", "models", "pipeline", "data"]

# Optional version info (can be used in Streamlit About page)
__version__ = "2.0.0"
__author__ = "SentimentClusterer Team [Nur Rasyid, et.al.]"
__license__ = "MIT"
