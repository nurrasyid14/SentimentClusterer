# app.py

import streamlit as st
from pathlib import Path
import tempfile
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from collections import Counter

# Internal modules
from pipeline.parser import JSONParser
from pipeline.preprocess import run_preprocess
from models.sentiment_machine.sentiment_mapper import SentimentEngine

# ---------------------------
# Streamlit configuration
# ---------------------------
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")
st.title("📊 Sentiment Analyzer Dashboard")
st.write("Unggah file JSON, pilih metode analisis sentimen, dan lihat hasil visualisasi.")

# Sidebar: user settings
st.sidebar.header("⚙️ Pengaturan Analisis")
method = st.sidebar.selectbox("Metode Analisis Sentimen", ["lexicon", "ml", "deep"])
run_button = st.sidebar.button("🚀 Jalankan Analisis")

# File uploader
uploaded_file = st.file_uploader("Unggah file JSON hasil scraping", type=["json"])
data_dir = Path("data/processed")
data_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Full pipeline
# ---------------------------
def run_pipeline(json_path: str, method: str) -> pd.DataFrame:
    # --- Parsing ---
    st.info("📥 Parsing JSON...")
    parser = JSONParser(json_path, data_dir / "parsed_comments.pkl")
    parsed_comments = parser.parse()

    if not parsed_comments:
        st.error("❌ Tidak ada teks yang berhasil diparse.")
        return pd.DataFrame()

    # --- Preprocessing ---
    st.info("🧹 Membersihkan dan men-tokenisasi teks...")
    tokens_path = data_dir / "tokens.pkl"
    run_preprocess(str(parser.output_path), str(tokens_path))
    all_tokens = joblib.load(tokens_path)
    docs = [" ".join(toks) for toks in all_tokens if toks]

    if not docs:
        st.error("❌ Tidak ada teks yang bisa digunakan setelah preprocessing.")
        return pd.DataFrame()

    # --- Sentiment Analysis ---
    st.info(f"🔠 Menjalankan Sentiment Analysis ({method.upper()})...")
    engine = SentimentEngine(method=method)

    if method in ["ml", "deep"]:
        # Dummy labels for training placeholder
        dummy_labels = np.random.randint(0, 3, len(docs))
        engine.prepare_and_train(docs, dummy_labels)

    sentiments = engine.predict(docs)

    # --- Embeddings + PCA ---
    embeddings = engine.get_embeddings(docs)
    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        proj = pca.fit_transform(embeddings)
    elif embeddings.shape[1] == 2:
        proj = embeddings
    else:
        # For lexicon or empty embeddings
        proj = np.random.randn(len(docs), 2)

    df = pd.DataFrame({
        "text": docs,
        "sentiment": sentiments,
        "x": proj[:, 0],
        "y": proj[:, 1],
    })

    return df

# ---------------------------
# Streamlit execution
# ---------------------------
if run_button and uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("🚧 Pipeline sedang berjalan..."):
        df_result = run_pipeline(tmp_path, method)

    if not df_result.empty:
        st.success("✅ Analisis selesai!")

        # --- Sentiment summary ---
        st.subheader("📊 Ringkasan Sentimen")
        counts = df_result["sentiment"].value_counts().sort_index()
        sentiment_map = {0: "Negatif", 1: "Netral", 2: "Positif"}
        counts.index = counts.index.map(sentiment_map)

        fig_bar = px.bar(
            counts.reset_index(),
            x="index",
            y="sentiment",
            text_auto=True,
            title="Jumlah Komentar per Sentimen",
            labels={"index": "Sentimen", "sentiment": "Jumlah"}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # --- Top keywords per sentiment ---
        st.subheader("🔑 Kata Kunci Populer per Sentimen")
        for s in sorted(df_result["sentiment"].unique()):
            s_label = sentiment_map.get(s, str(s))
            texts_s = df_result[df_result["sentiment"] == s]["text"]
            words = [w for t in texts_s for w in t.split() if len(w) > 2]
            most_common = Counter(words).most_common(10)
            st.write(f"**{s_label}:**")
            st.write(", ".join([w for w, _ in most_common]))

        # --- PCA scatter plot ---
        st.subheader("📍 Visualisasi Teks (PCA Projection)")
        fig_scatter = px.scatter(
            df_result, x="x", y="y",
            color=df_result["sentiment"].map(sentiment_map),
            hover_data=["text"],
            title="PCA Projection of Comments by Sentiment",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # --- Data table ---
        st.subheader("📜 Data Detail")
        st.dataframe(df_result.head(50), use_container_width=True)
    else:
        st.error("❌ Pipeline gagal dijalankan, tidak ada data yang valid.")

else:
    st.info("👆 Unggah file JSON dan tekan **Jalankan Analisis** untuk memulai.")
