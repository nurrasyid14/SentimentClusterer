#Sentiment_Analyzer.py

import streamlit as st
import pandas as pd
from pathlib import Path

DATA_PROCESSED = Path("data/processed")

st.title("Sentiment Analyzer")

sentiment_file = DATA_PROCESSED / "sentiment_summary.csv"
if sentiment_file.exists():
    df = pd.read_csv(sentiment_file)
    st.dataframe(df)

    st.bar_chart(df[["positive_%","neutral_%","negative_%"]])
else:
    st.warning("File sentiment_summary.csv belum tersedia. Jalankan pipeline dulu.")
