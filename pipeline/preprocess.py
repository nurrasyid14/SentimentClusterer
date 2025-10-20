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

try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    logging.info("Downloading NLTK assets (stopwords, wordnet, punkt, punkt_tab)...")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')


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
}

# Stopwords and Lemmatizer Setup
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

# Default: English
stop_words = set(stopwords.words('english'))
stop_words.update(custom_stopwords)
lemmatizer = WordNetLemmatizer()


def clean_and_tokenize_text(text: str, lang: str = "en") -> List[str]:
    """Cleans, normalizes, and tokenizes a single text string."""
    if not isinstance(text, str) or not text.strip():
        return []

    # Lowercasing
    text = text.lower()

    # Remove URLs, mentions, hashtags, non-alphabetic chars
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    clean_tokens = [
        lemmatizer.lemmatize(ENGLISH_SLANG_DICT.get(tok, tok))
        for tok in tokens
        if tok not in stop_words and len(tok) > 1
    ]

    return clean_tokens


def run_preprocess(input_pkl: str, output_pkl: str, lang: str = "en") -> List[List[str]]:
    """Main preprocessing runner."""
    logging.info("🚀 Starting preprocessing pipeline...")

    input_path = Path(input_pkl)
    if not input_path.exists():
        logging.error(f"❌ Input file not found: {input_path}")
        return []

    try:
        parsed_comments = joblib.load(input_path)
    except Exception as e:
        logging.error(f"❌ Failed to read {input_path}: {e}")
        return []

    if not parsed_comments:
        logging.warning("⚠️ Parsed comments file is empty.")
        return []

    # Handle dict input
    if isinstance(parsed_comments[0], dict):
        parsed_comments = [c.get('text', '') for c in parsed_comments]

    logging.info(f"📦 Loaded {len(parsed_comments)} comments for cleaning...")

    # Process comments
    all_tokens = []
    for comment in tqdm(parsed_comments, desc="🧹 Cleaning comments"):
        toks = clean_and_tokenize_text(comment, lang=lang)
        if toks:
            all_tokens.append(toks)

    if not all_tokens:
        logging.warning("⚠️ All comments were filtered out (only stopwords or empty).")
        return []

    logging.info(f"✅ Preprocessing done. {len(all_tokens)} valid comments remain.")

    try:
        joblib.dump(all_tokens, output_pkl)
        logging.info(f"💾 Saved tokenized data to {output_pkl}")
    except Exception as e:
        logging.error(f"❌ Failed to save token file: {e}")

    logging.info(f"🧩 Sample tokens: {all_tokens[0][:10]}")

    return all_tokens


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    input_pkl = data_dir / "parsed_comments.pkl"
    output_pkl = data_dir / "tokens.pkl"

    run_preprocess(str(input_pkl), str(output_pkl))
