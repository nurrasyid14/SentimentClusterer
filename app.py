# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tempfile
import joblib
from pathlib import Path
from sklearn.decomposition import PCA

# === Internal modules ===
from pipeline.parser import JSONParser
from pipeline.preprocess import run_preprocess
from models.sentiment_machine.sentiment_engine import SentimentEngine

# === Streamlit configuration ===
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

st.title("Sentiment Analyzer")
st.write("Unggah file JSON komentar dan jalankan analisis sentimen secara otomatis.")

# === Sidebar ===
st.sidebar.header("⚙️ Pengaturan Analisis")
run_button = st.sidebar.button("🚀 Jalankan Analisis")

# === File uploader ===
uploaded_file = st.file_uploader("Unggah file JSON hasil scraping", type=["json"])

data_dir = Path("data/processed")
data_dir.mkdir(parents=True, exist_ok=True)

# ====================================================
# 🚀 FULL PIPELINE FUNCTION
# ====================================================
def run_full_pipeline(json_path: str) -> pd.DataFrame:
    # --- PARSING ---
    st.info("📥 Memulai parsing JSON...")
    parser = JSONParser(json_path, data_dir / "parsed_comments.pkl")
    parsed_comments = parser.parse()

    if not parsed_comments:
        st.error("❌ Tidak ada teks yang berhasil diparse dari file JSON.")
        return pd.DataFrame()

    # --- PREPROCESSING ---
    st.info("🧹 Membersihkan & men-tokenisasi teks...")
    tokens_path = data_dir / "tokens.pkl"
    run_preprocess(str(parser.output_path), str(tokens_path))

    all_tokens = joblib.load(tokens_path)
    docs = [" ".join(toks) for toks in all_tokens if toks]

    if not docs:
        st.error("❌ Tidak ada teks yang bisa digunakan setelah preprocessing.")
        return pd.DataFrame()

    # --- EMBEDDING & SENTIMENT ---
    st.info("🔠 Membuat embeddings & memprediksi sentimen...")
    sentiment_engine = SentimentEngine()
    dummy_labels = np.random.randint(0, 3, len(docs))  # placeholder training
    sentiment_engine.prepare_and_train(docs, dummy_labels)
    vectors = sentiment_engine.embedding_model.transform(docs)

    predicted_labels = sentiment_engine.predict(docs)
    SENTIMENT_MAP = {0: "Negatif", 1: "Netral", 2: "Positif"}

    df = pd.DataFrame({
        "text": docs,
        "sentiment_label": [SENTIMENT_MAP.get(l, "Tidak diketahui") for l in predicted_labels]
    })

    # --- PCA PROJECTION ---
    st.info("📊 Menghitung PCA untuk visualisasi...")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]

    return df

# ====================================================
# 🚦 PIPELINE EXECUTION
# ====================================================
if run_button and uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("🚧 Pipeline sedang berjalan..."):
        df_result = run_full_pipeline(tmp_path)

    if not df_result.empty:
        st.success("✅ Analisis selesai!")

        # --- METRICS ---
        st.subheader("Ringkasan Hasil Analisis")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Komentar", len(df_result))
        col2.metric("Jumlah Kategori Sentimen", df_result["sentiment_label"].nunique())
        col3.metric("Sentimen Dominan", df_result["sentiment_label"].mode()[0])

        # --- PIE CHART ---
        st.subheader("Distribusi Sentimen")
        sentiment_counts = df_result["sentiment_label"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentimen", "Jumlah"]
        fig_pie = px.pie(sentiment_counts, values="Jumlah", names="Sentimen",
                         color="Sentimen", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pie, use_container_width=True)

        # --- PCA SCATTER PLOT ---
        st.subheader("Visualisasi PCA 2D dari Embeddings")
        fig_scatter = px.scatter(df_result, x="x", y="y",
                                 color="sentiment_label",
                                 hover_data=["text"],
                                 title="Peta Sentimen (PCA 2D)")
        st.plotly_chart(fig_scatter, use_container_width=True)

        # --- DATA TABLE ---
        st.subheader("📜 Detail Data")
        st.dataframe(df_result.head(20), use_container_width=True)
    else:
        st.error("❌ Pipeline gagal dijalankan, tidak ada data yang valid.")

else:
    st.warning("👆 Unggah file JSON dan tekan **Jalankan Analisis** untuk memulai.")
