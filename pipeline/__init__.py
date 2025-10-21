"""
Pipeline package for processing, cleaning, translating, and analyzing text comments and tweets.

Modules:
- parser: JSON parsing and extraction
- preprocess: text cleaning, tokenization, and slang normalization
- translator: text translation using GoogleTranslator
- pipeline_utils: helper functions for DataFrame cleaning, filtering, and summary
- main: full pipeline runner (parse → preprocess → embed → cluster → sentiment)
"""

# Expose key classes/functions for easier imports
from .parser import JSONParser
from .preprocess import clean_and_tokenize_text, run_preprocess
from .translator import translate_text, translate_batch
from .pipeline_utils import (
    load_tweets,
    clean_text,
    preprocess_texts,
    filter_by_keyword,
    top_users,
    summarize_engagement,
    get_time_range
)

# Optionally: expose main pipeline runner
from .main import SentimentPipeline

__all__ = [
    "JSONParser",
    "clean_and_tokenize_text", "run_preprocess",
    "translate_text", "translate_batch",
    "load_tweets", "clean_text", "preprocess_texts",
    "filter_by_keyword", "top_users", "summarize_engagement", "get_time_range",
    "SentimentPipeline"
]