# app.py

import streamlit as st
from pathlib import Path
import tempfile
import joblib
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
import plotly.express as px

# === Internal modules ===
from pipeline.parser import JSONParser
from pipeline.preprocess import run_preprocess
from models.sentiment_machine.sentiment_mapper import SentimentEngine

# === Streamlit config ===
st.set_page_config(page_title="Sentiment Analyzer Dashboard", layout="wide")
st.title("Sentiment Analyzer")
st.write("Unggah file JSON hasil scraping dan jalankan analisis sentimen secara otomatis.")

# === Sidebar options ===
st.sidebar.header("⚙️ Pengaturan Analisis")
method_choice = st.sidebar.selectbox("Metode Analisis Sentimen", ["lexicon", "ml", "deep"])
run_button = st.sidebar.button("🚀 Jalankan Analisis")

# === File uploader ===
uploaded_file = st.file_uploader("Unggah file JSON hasil scraping", type=["json"])

data_dir = Path("data/processed")
data_dir.mkdir(parents=True, exist_ok=True)

# ====================================================
# 🚀 Pipeline
# ====================================================
def run_pipeline(json_path: str, method: str):
    st.info("📥 Parsing JSON...")
    parser = JSONParser(json_path, data_dir / "parsed_comments.pkl")
    parsed_comments = parser.parse()

    if not parsed_comments:
        st.error("❌ Tidak ada teks yang berhasil diparse dari file JSON.")
        return pd.DataFrame()

    st.info("🧹 Preprocessing & tokenisasi...")
    tokens_path = data_dir / "tokens.pkl"
    run_preprocess(str(parser.output_path), str(tokens_path))

    all_tokens = joblib.load(tokens_path)
    docs = [" ".join(toks) for toks in all_tokens if toks]

    if not docs:
        st.error("❌ Tidak ada teks yang bisa digunakan setelah preprocessing.")
        return pd.DataFrame()

    # --- Sentiment analysis ---
    st.info(f"🔠 Menganalisis sentimen menggunakan metode: {method.upper()}")
    engine = SentimentEngine(method=method)

    # Dummy labels for ML/Deep training
    if method in ["ml", "deep"]:
        dummy_labels = np.random.randint(0, 3, len(docs))
        engine.prepare_and_train(docs, dummy_labels)

    sentiments = engine.predict(docs)
    embeddings = engine.get_embeddings(docs)

    # --- PCA projection if needed ---
    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        proj = pca.fit_transform(embeddings)
    else:
        proj = embeddings

    df_result = pd.DataFrame({
        "text": docs,
        "sentiment": sentiments,
        "x": proj[:, 0] if proj.shape[1] > 0 else [0]*len(docs),
        "y": proj[:, 1] if proj.shape[1] > 1 else [0]*len(docs)
    })

    return df_result

# ====================================================
# Streamlit execution
# ====================================================
if run_button and uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("🚧 Pipeline sedang berjalan..."):
        df_result = run_pipeline(tmp_path, method_choice)

    if not df_result.empty:
        st.success("✅ Analisis selesai!")

        # --- Metrics ---
        st.subheader("Ringkasan Hasil Analisis")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Komentar", len(df_result))
        col2.metric("Jumlah Sentimen", df_result["sentiment"].nunique())
        col3.metric("Sentimen Dominan", int(pd.Series(df_result["sentiment"]).mode()[0]))

        # --- Sentiment distribution ---
        sentiment_counts = df_result["sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["sentiment", "count"]

        # Bar chart
        fig_bar = px.bar(
            sentiment_counts,
            x="sentiment",
            y="count",
            color="sentiment",
            text="count",
            title="Distribusi Sentimen"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Pie chart
        fig_pie = px.pie(
            sentiment_counts,
            values="count",
            names="sentiment",
            color="sentiment",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            title="Distribusi Sentimen (Pie Chart)"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # --- PCA scatter plot ---
        fig_scatter = px.scatter(
            df_result,
            x="x",
            y="y",
            color="sentiment",
            hover_data=["text"],
            title="Peta Sentimen (PCA Projection)",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # --- Top keywords per sentiment ---
        st.subheader("Kata Kunci Populer per Sentimen")
        for sent in df_result["sentiment"].unique():
            texts = df_result[df_result["sentiment"] == sent]["text"].tolist()
            all_words = [w.lower() for t in texts for w in t.split() if len(w) > 2]
            top_words = Counter(all_words).most_common(10)
            st.write(f"**{sent}**: ", ", ".join([w for w, _ in top_words]))

        # --- Optional: show dataframe ---
        st.subheader("📜 Detail Data")
        st.dataframe(df_result.head(20), use_container_width=True)
    else:
        st.error("❌ Pipeline gagal dijalankan, tidak ada data yang valid.")
else:
    st.warning("👆 Unggah file JSON dan tekan **Jalankan Analisis** untuk memulai.")
