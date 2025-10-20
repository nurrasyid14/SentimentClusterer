import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional


def load_tweets(json_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of tweet dictionaries into a cleaned pandas DataFrame.
    """
    df = pd.DataFrame(json_data)
    
    # Standardize column names
    df.columns = df.columns.str.lower()

    # Convert created_at to datetime
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)

    # Replace None with empty string for text columns
    text_cols = ["full_text", "in_reply_to_screen_name", "username", "location"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    return df


def clean_text(text: str) -> str:
    """
    Basic cleaning for tweet text — remove URLs, mentions, hashtags, etc.
    """
    import re
    text = re.sub(r"http\S+", "", text)          # remove URLs
    text = re.sub(r"@\w+", "", text)             # remove mentions
    text = re.sub(r"#\w+", "", text)             # remove hashtags
    text = re.sub(r"\s+", " ", text).strip()     # normalize spaces
    return text


def preprocess_texts(df: pd.DataFrame, column: str = "full_text") -> pd.DataFrame:
    """
    Clean tweet texts in the specified column and add a 'clean_text' column.
    """
    if column in df.columns:
        df["clean_text"] = df[column].apply(clean_text)
    return df


def filter_by_keyword(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    """
    Return tweets containing a given keyword (case-insensitive) in their clean text.
    """
    mask = df["clean_text"].str.contains(keyword, case=False, na=False)
    return df[mask]


def top_users(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Return top users by number of tweets.
    """
    if "user_id_str" not in df.columns:
        return pd.DataFrame()
    return df["user_id_str"].value_counts().head(n).reset_index(names=["user_id_str", "tweet_count"])


def summarize_engagement(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarize total and average engagement metrics (likes, retweets, replies, quotes).
    """
    metrics = ["favorite_count", "retweet_count", "reply_count", "quote_count"]
    summary = {}
    for m in metrics:
        if m in df.columns:
            summary[f"total_{m}"] = df[m].sum()
            summary[f"avg_{m}"] = df[m].mean()
    return summary


def get_time_range(df: pd.DataFrame) -> Optional[Dict[str, datetime]]:
    """
    Return the earliest and latest timestamps in the dataset.
    """
    if "created_at" not in df.columns or df["created_at"].isna().all():
        return None
    return {
        "start": df["created_at"].min(),
        "end": df["created_at"].max(),
    }
