#  pipeline/preprocess.py


import re
import joblib
import logging
from pathlib import Path
from typing import List
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Logging Configuration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Slang Dictionary

ENGLISH_SLANG_DICT = {
    'u': 'you', 'r': 'are', 'gr8': 'great', 'btw': 'by the way', 'lol': 'laughing out loud',
    'thx': 'thanks', 'plz': 'please', 'im': "i'm", 'gonna': 'going to', 'wanna': 'want to',
    'idk': "i don't know", 'omg': 'oh my god', 'brb': 'be right back', 'asap': 'as soon as possible'
    # TODO: extend slang dictionary if needed
}

# ------------------------
# Stopwords and Lemmatizer Setup
# ------------------------
custom_stopwords = {'rt', 'via', 'amp', 'cc'}
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    logging.info("Downloading NLTK assets (stopwords, wordnet)...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
stop_words.update(custom_stopwords)
lemmatizer = WordNetLemmatizer()

# ============================================================
#  Core Cleaning & Tokenization
# ============================================================

def clean_and_tokenize_text(text: str) -> List[str]:
    """Cleans, normalizes, and tokenizes a single text string."""
    if not isinstance(text, str):
        return []

    # Lowercasing
    text = text.lower()

    # Remove URLs, mentions, hashtags, and non-alphabetic characters
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    tokens = word_tokenize(text)

    # Slang normalization + stopword removal + lemmatization
    clean_tokens = [
        lemmatizer.lemmatize(ENGLISH_SLANG_DICT.get(tok, tok))
        for tok in tokens
        if tok not in stop_words and len(tok) > 1
    ]

    return clean_tokens


# ============================================================
#  Main Preprocessing Runner
# ============================================================

def run_preprocess(input_pkl: str, output_pkl: str) -> List[List[str]]:
    """
    Loads parsed comments, cleans and tokenizes them,
    and saves result into output_pkl.
    """
    logging.info("Starting preprocessing pipeline...")

    # Load parsed comments
    input_path = Path(input_pkl)
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}. Please run parser.py first.")
        return []

    try:
        parsed_comments = joblib.load(input_path)
    except Exception as e:
        logging.error(f"Failed to read {input_path}: {e}")
        return []

    # If data is dict-based, extract text field
    if parsed_comments and isinstance(parsed_comments[0], dict):
        parsed_comments = [c.get('text', '') for c in parsed_comments]

    logging.info(f"Loaded {len(parsed_comments)} comments for cleaning...")

    # Process all comments
    all_tokens = [
        clean_and_tokenize_text(comment)
        for comment in tqdm(parsed_comments, desc="Cleaning comments")
    ]
    all_tokens = [t for t in all_tokens if t]  # Remove empty entries

    logging.info(f"Preprocessing done. {len(all_tokens)} comments successfully tokenized.")

    # Save tokens
    try:
        joblib.dump(all_tokens, output_pkl)
        logging.info(f"Saved tokenized data to {output_pkl}")
    except Exception as e:
        logging.error(f"Failed to save token file: {e}")

    # Return token list for further processing
    if all_tokens:
        logging.info(f"Sample tokens: {all_tokens[0][:10]}")

    return all_tokens


# ============================================================
#  Script Entry Point
# ============================================================

if __name__ == "__main__":
    DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
    INPUT_PKL = DATA_DIR / "parsed_comments.pkl"
    OUTPUT_PKL = DATA_DIR / "tokens.pkl"

    run_preprocess(str(INPUT_PKL), str(OUTPUT_PKL))
