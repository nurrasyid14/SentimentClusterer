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
from models.sentiment_machine.sentiment_mapper import SentimentEngine

# === Streamlit config ===
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")
st.title("Sentiment Analyzer Dashboard")
st.write("Unggah file JSON hasil scraping dan jalankan analisis sentimen secara otomatis.")

# === Sidebar: Pengaturan Sentimen ===
st.sidebar.header("⚙️ Pengaturan Analisis")
method = st.sidebar.selectbox("Metode Analisis Sentimen", ["lexicon", "ml", "deep"])
run_button = st.sidebar.button("🚀 Jalankan Analisis")

# === File uploader ===
uploaded_file = st.file_uploader("Unggah file JSON hasil scraping", type=["json"])
data_dir = Path("data/processed")
data_dir.mkdir(parents=True, exist_ok=True)

# --- Pipeline runner ---
def run_pipeline(json_path: str, method: str) -> pd.DataFrame:
    st.info("📥 Memulai parsing JSON...")
    parser = JSONParser(json_path, data_dir / "parsed_comments.pkl")
    parsed_comments = parser.parse()

    if not parsed_comments:
        st.error("❌ Tidak ada teks yang berhasil diparse dari file JSON.")
        return pd.DataFrame()

    st.info("🧹 Membersihkan & men-tokenisasi teks...")
    tokens_path = data_dir / "tokens.pkl"
    run_preprocess(str(parser.output_path), str(tokens_path))
    all_tokens = joblib.load(tokens_path)
    docs = [" ".join(toks) for toks in all_tokens if toks]

    if not docs:
        st.error("❌ Tidak ada teks yang bisa digunakan setelah preprocessing.")
        return pd.DataFrame()

    st.info(f"🔠 Menjalankan analisis sentimen ({method})...")
    engine = SentimentEngine(method=method)
    
    # Dummy training for ML/Deep
    if method in ["ml", "deep"]:
        dummy_labels = np.random.randint(0, 3, len(docs))
        engine.prepare_and_train(docs, dummy_labels)

    # Predict
    sentiments_numeric = engine.predict(docs)
    SENTIMENT_LABELS = {0: "Negatif", 1: "Netral", 2: "Positif"}
    sentiments_human = [SENTIMENT_LABELS[s] for s in sentiments_numeric]

    # Embeddings for PCA
    embeddings = engine.get_embeddings(docs)
    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        proj = pca.fit_transform(embeddings)
    else:
        proj = embeddings if embeddings.shape[1] == 2 else np.random.randn(len(docs), 2)

    df = pd.DataFrame({
        "text": docs,
        "sentiment": sentiments_human,
        "x": proj[:, 0],
        "y": proj[:, 1]
    })

    return df

# === Streamlit execution ===
if run_button and uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("🚧 Pipeline sedang berjalan..."):
        df_result = run_pipeline(tmp_path, method)

    if not df_result.empty:
        st.success("✅ Analisis selesai!")

        # --- Metrics summary ---
        st.subheader("Ringkasan Hasil Analisis")
        sentiment_counts = df_result["sentiment"].value_counts()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Komentar", len(df_result))
        col2.metric("Jumlah Sentimen", len(sentiment_counts))
        col3.metric("Sentimen Dominan", sentiment_counts.idxmax())

        # --- Pie chart ---
        st.subheader("Distribusi Sentimen")
        df_counts = sentiment_counts.reset_index()
        df_counts.columns = ["sentiment", "count"]
        fig_pie = px.pie(df_counts, values="count", names="sentiment",
                         color="sentiment", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pie, use_container_width=True)

        # --- PCA scatter ---
        st.subheader("Visualisasi Sentimen (PCA)")
        fig_scatter = px.scatter(df_result, x="x", y="y",
                                 color="sentiment", hover_data=["text"],
                                 title="Proyeksi 2D Sentimen")
        st.plotly_chart(fig_scatter, use_container_width=True)

        # --- Data table ---
        st.subheader("📜 Detail Data")
        st.dataframe(df_result.head(20), use_container_width=True)

        # --- Top keywords per sentiment ---
        st.subheader("🔑 Kata Kunci Populer per Sentimen")
        from collections import Counter
        for s in SENTIMENT_LABELS.values():
            words = [w for doc, sent in zip(df_result["text"], df_result["sentiment"]) if sent == s
                     for w in doc.split()]
            top_words = Counter(words).most_common(10)
            st.markdown(f"**{s}:** " + ", ".join([w for w, _ in top_words]))
    else:
        st.error("❌ Pipeline gagal dijalankan, tidak ada data yang valid.")
else:
    st.warning("👆 Unggah file JSON dan tekan **Jalankan Analisis** untuk memulai.")
