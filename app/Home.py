import streamlit as st
from pipeline.main import run_pipeline  # panggil pipeline utama
from pathlib import Path

st.set_page_config(page_title="Social Sentiment Dashboard", layout="wide")

st.title("Social Sentiment Analysis Dashboard")
st.write("Overview dan kontrol pipeline")

if st.button("Run Full Pipeline"):
    with st.spinner("Menjalankan pipeline..."):
        run_pipeline()  # jalankan semua: scrape → parse → embeddings → cluster → CSV/PKL
    st.success("Pipeline selesai!")

st.markdown("---")
st.write("Pilih menu di sidebar untuk eksplorasi data, analisis sentimen, atau visualisasi cluster.")
