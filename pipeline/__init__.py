"""
Pipeline module — handles sequential processing:
1. Parsing raw JSON
2. Preprocessing text
3. Translating to English
4. Building embeddings
5. Clustering results
"""
from parser import parse_json
from .preprocess import run_preprocess
# translator & utils akan dipanggil langsung oleh main pipeline

__all__ = [
    "parse_json",
    "run_preprocess",

]