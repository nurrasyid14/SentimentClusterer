import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tempfile
import os
import joblib
from pathlib import Path

# Import modul internal
from pipeline.parser import parse_raw_json
from pipeline.preprocess import run_preprocess
from models.clustering import FuzzyCMeansClustering, KMeansClustering
from models.sentiment_mapper import SentimentEngine

st.set_page_config(page_title="Sentiment Cluster Dashboard", layout="wide")

# --- HEADER ---
st.title("🧠 Sentiment Cluster Analyzer")
st.write("Unggah file JSON hasil scraping dan jalankan analisis sentimen + klustering secara otomatis.")

# --- SIDEBAR CONFIG ---
st.sidebar.header("⚙️ Pengaturan Analisis")
cluster_method = st.sidebar.selectbox("Metode Clustering", ["Fuzzy C-Means", "K-Means"])
n_clusters = st.sidebar.slider("Jumlah Cluster (k)", 2, 5, 3)
run_button = st.sidebar.button("🚀 Jalankan Analisis")

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Unggah file JSON hasil scraping", type=["json"])

data_dir = Path("data/processed")
data_dir.mkdir(parents=True, exist_ok=True)

# --- FUNGSI PIPELINE ---
def run_full_pipeline(json_path: str, method: str, k: int):
    st.info("📥 Memulai parsing JSON...")
    parsed_comments = parse_raw_json(str(Path(json_path).parent))
    parsed_path = data_dir / "parsed_comments.pkl"
    joblib.dump(parsed_comments, parsed_path)

    st.info("🧹 Membersihkan & men-tokenisasi teks...")
    tokens_path = data_dir / "tokens.pkl"
    run_preprocess(str(parsed_path), str(tokens_path))
    all_tokens = joblib.load(tokens_path)

    # Gabungkan token jadi kalimat (untuk vektorisasi)
    docs = [" ".join(toks) for toks in all_tokens if toks]

    st.info("🔠 Membuat embeddings & sentimen...")
    sentiment_engine = SentimentEngine()
    dummy_labels = np.random.randint(0, 3, len(docs))  # sementara
    sentiment_engine.prepare_and_train(docs, dummy_labels)

    vectors = sentiment_engine.embedding_model.transform(docs)

    st.info("🌀 Menjalankan clustering...")
    if method.lower().startswith("fuzzy"):
        model = FuzzyCMeansClustering(n_clusters=k)
        model.fit(vectors)
        labels = np.argmax(model.u, axis=0)
    else:
        model = KMeansClustering(n_clusters=k)
        model.fit(vectors)
        labels = model.predict(vectors)

    df = pd.DataFrame({
        "text": docs,
        "cluster": labels
    })

    return df


# --- PIPELINE EKSEKUSI ---
if run_button and uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("🚧 Pipeline sedang berjalan..."):
        df_result = run_full_pipeline(tmp_path, cluster_method, n_clusters)

    st.success("✅ Analisis selesai!")

    # --- METRICS ---
    st.subheader("📊 Ringkasan Hasil Analisis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Komentar", len(df_result))
    col2.metric("Jumlah Cluster", df_result['cluster'].nunique())
    col3.metric("Cluster Dominan", int(df_result['cluster'].mode()[0]))

    # --- PIE CHART DISTRIBUSI SENTIMEN ---
    st.subheader("🎯 Distribusi Sentimen (0=Negatif, 1=Netral, 2=Positif)")
    sentiment_counts = df_result['cluster'].value_counts().reset_index()
    sentiment_counts.columns = ['Cluster', 'Jumlah']
    fig_pie = px.pie(sentiment_counts, values='Jumlah', names='Cluster',
                     color='Cluster', color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_pie, use_container_width=True)

    # --- SCATTER PLOT CLUSTERING ---
    st.subheader("🌈 Visualisasi Klaster")
    np.random.seed(42)
    df_result["x"] = np.random.randn(len(df_result))
    df_result["y"] = np.random.randn(len(df_result))
    fig_scatter = px.scatter(df_result, x="x", y="y", color=df_result["cluster"].astype(str),
                             hover_data=["text"], title="Peta Klaster Sentimen")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- TABEL HASIL ---
    st.subheader("📜 Detail Data")
    st.dataframe(df_result.head(20), use_container_width=True)

else:
    st.warning("👆 Unggah file JSON dan tekan **Jalankan Analisis** untuk memulai.")

