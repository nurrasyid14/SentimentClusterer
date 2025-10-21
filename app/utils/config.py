from pathlib import Path

# -------------------------------
# Data paths
# -------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # root of project
DATA_RAW = BASE_DIR / "data/raw"
DATA_PROCESSED = BASE_DIR / "data/processed"

# Parsed and tokenized files
PARSED_COMMENTS_FILE = DATA_PROCESSED / "parsed_comments.pkl"
TOKENIZED_COMMENTS_FILE = DATA_PROCESSED / "tokens.pkl"

# Embeddings
EMBEDDINGS_FILE = DATA_PROCESSED / "embeddings.pkl"

# Sentiment summary
SENTIMENT_SUMMARY_FILE = DATA_PROCESSED / "sentiment_summary.csv"
