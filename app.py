import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tempfile
import joblib
from pathlib import Path

# === Import modul internal ===
from pipeline.parser import JSONParser
from pipeline.preprocess import run_preprocess
from models.clustering import FuzzyCMeansClustering, KMeansClustering
from models.sentiment_mapper import SentimentEngine

# === Konfigurasi Streamlit ===
st.set_page_config(page_title="Sentiment Cluster Dashboard", layout="wide")

st.title("Sentiment Cluster Analyzer")
st.write("Unggah file JSON hasil scraping dan jalankan analisis sentimen + klustering secara otomatis.")

# === Sidebar ===
st.sidebar.header("⚙️ Pengaturan Analisis")
cluster_method = st.sidebar.selectbox("Metode Clustering", ["Fuzzy C-Means", "K-Means"])
n_clusters = st.sidebar.slider("Jumlah Cluster (k)", 2, 5, 3)
run_button = st.sidebar.button("🚀 Jalankan Analisis")

# === File uploader ===
uploaded_file = st.file_uploader("Unggah file JSON hasil scraping", type=["json"])

data_dir = Path("data/processed")
data_dir.mkdir(parents=True, exist_ok=True)

# ====================================================
# 🚀 FULL PIPELINE
# ====================================================
def run_full_pipeline(json_path: str, method: str, k: int) -> pd.DataFrame:
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

    # --- EMBEDDING & SENTIMEN ---
        st.info("🔠 Membuat embeddings & sentimen (dummy training)...")
    sentiment_engine = SentimentEngine()

    # Dummy label untuk pelatihan awal (sementara)
    dummy_labels = np.random.randint(0, 3, len(docs))
    sentiment_engine.prepare_and_train(docs, dummy_labels)
    vectors = sentiment_engine.embedding_model.transform(docs)

    # --- CLUSTERING ---
    st.info(f"🌀 Menjalankan {method} clustering...")
    if method.lower().startswith("fuzzy"):
        model = FuzzyCMeansClustering(n_clusters=k)
        model.fit(vectors)
        labels = np.argmax(model.u, axis=0)
    else:
        model = KMeansClustering(n_clusters=k)
        model.fit(vectors)
        labels = model.predict(vectors)

    # --- MAP LABEL KE SENTIMEN ---
    SENTIMENT_MAP = {0: "Negatif", 1: "Netral", 2: "Positif"}

    df = pd.DataFrame({
        "text": docs,
        "cluster": labels,
    })
    df["sentiment_label"] = df["cluster"].map(SENTIMENT_MAP).fillna("Tidak diketahui")

    # --- METADATA TAMBAHAN UNTUK VISUALISASI ---
    np.random.seed(42)
    df["x"] = np.random.randn(len(df))
    df["y"] = np.random.randn(len(df))

    return df

# ====================================================
# 🚦 PIPELINE EKSEKUSI STREAMLIT
# ====================================================
if run_button and uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("🚧 Pipeline sedang berjalan..."):
        df_result = run_full_pipeline(tmp_path, cluster_method, n_clusters)

    if not df_result.empty:
        st.success("✅ Analisis selesai!")

        # --- METRICS ---
        st.subheader("Ringkasan Hasil Analisis")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Komentar", len(df_result))
        col2.metric("Jumlah Cluster", df_result["cluster"].nunique())
        col3.metric("Cluster Dominan", int(df_result["cluster"].mode()[0]))

        # --- PIE CHART ---
        st.subheader("Distribusi Sentimen (0=Negatif, 1=Netral, 2=Positif)")
        sentiment_counts = df_result["cluster"].value_counts().reset_index()
        sentiment_counts.columns = ["Cluster", "Jumlah"]
        fig_pie = px.pie(sentiment_counts, values="Jumlah", names="Cluster",
                         color="Cluster", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pie, use_container_width=True)

        # --- SCATTER PLOT ---
        st.subheader("Visualisasi Klaster")
        np.random.seed(42)
        df_result["x"] = np.random.randn(len(df_result))
        df_result["y"] = np.random.randn(len(df_result))
        fig_scatter = px.scatter(df_result, x="x", y="y",
                                 color=df_result["cluster"].astype(str),
                                 hover_data=["text"], title="Peta Klaster Sentimen")
        st.plotly_chart(fig_scatter, use_container_width=True)

        # --- DATA TABLE ---
        st.subheader("📜 Detail Data")
        st.dataframe(df_result.head(20), use_container_width=True)
    else:
        st.error("❌ Pipeline gagal dijalankan, tidak ada data yang valid.")

else:
    st.warning("👆 Unggah file JSON dan tekan **Jalankan Analisis** untuk memulai.")
