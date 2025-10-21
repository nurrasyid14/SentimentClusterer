#app/__init__.py

"""
App package for SentimentClusterer Streamlit interface.

This module exposes:
- Streamlit page modules under `app.pages`
- Utility helpers under `app.utils`
"""

from . import pages
from . import utils

__all__ = ["pages", "utils"]
