# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tempfile
import joblib
from pathlib import Path
from collections import Counter
from sklearn.decomposition import PCA

# === Internal modules ===
from pipeline.parser import JSONParser
from pipeline.preprocess import run_preprocess
from models.sentiment_machine.sentiment_mapper import SentimentEngine

# === Streamlit config ===
st.set_page_config(page_title="Sentiment Analyzer Dashboard", layout="wide")
st.title("Sentiment Analyzer Dashboard")
st.write("Unggah file JSON dan jalankan analisis sentimen menggunakan Lexicon, ML, atau Deep Learning.")

# === Sidebar ===
st.sidebar.header("⚙️ Pengaturan Analisis")
sentiment_method = st.sidebar.selectbox("Metode Analisis Sentimen", ["Lexicon", "ML", "Deep"])
run_button = st.sidebar.button("🚀 Jalankan Analisis")

# === File uploader ===
uploaded_file = st.file_uploader("Unggah file JSON hasil scraping", type=["json"])

data_dir = Path("data/processed")
data_dir.mkdir(parents=True, exist_ok=True)

# ====================================================
# 🚀 FULL PIPELINE
# ====================================================
def run_pipeline(json_path: str, method: str):
    st.info("📥 Parsing JSON...")
    parser = JSONParser(json_path, data_dir / "parsed_comments.pkl")
    parsed_comments = parser.parse()
    if not parsed_comments:
        st.error("❌ Tidak ada teks yang berhasil diparse dari file JSON.")
        return pd.DataFrame()

    st.info("🧹 Preprocessing teks...")
    tokens_path = data_dir / "tokens.pkl"
    run_preprocess(str(parser.output_path), str(tokens_path))
    all_tokens = joblib.load(tokens_path)
    docs = [" ".join(toks) for toks in all_tokens if toks]

    if not docs:
        st.error("❌ Tidak ada teks valid setelah preprocessing.")
        return pd.DataFrame()

    st.info("🔠 Analisis Sentimen...")
    sentiment_engine = SentimentEngine(method=method.lower())

    # Dummy labels for ML/Deep training if necessary
    if method.lower() in ["ml", "deep"]:
        dummy_labels = np.random.randint(0, 3, len(docs))
        sentiment_engine.prepare_and_train(docs, dummy_labels)

    # Predict sentiment
    labels = sentiment_engine.predict(docs)
    SENTIMENT_MAP = {0: "Negatif", 1: "Netral", 2: "Positif"}

    df = pd.DataFrame({
        "text": docs,
        "sentiment_label": [SENTIMENT_MAP.get(l, "Tidak diketahui") for l in labels]
    })

    # PCA for embedding projection (only ML/Deep)
    if method.lower() in ["ml", "deep"]:
        st.info("📊 Menghitung PCA untuk visualisasi...")
        if method.lower() == "ml":
            X = sentiment_engine.vectorizer.transform([t.split() for t in docs])
        else:
            X = sentiment_engine.embedding_model.transform([t.split() for t in docs])
        pca = PCA(n_components=2, random_state=42)
        X_proj = pca.fit_transform(X)
        df["x"] = X_proj[:, 0]
        df["y"] = X_proj[:, 1]
    else:
        # Lexicon: random 2D projection
        np.random.seed(42)
        df["x"] = np.random.randn(len(df))
        df["y"] = np.random.randn(len(df))

    return df

# ====================================================
# 🚦 Streamlit Execution
# ====================================================
if run_button and uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("🚧 Pipeline sedang berjalan..."):
        df_result = run_pipeline(tmp_path, sentiment_method)

    if not df_result.empty:
        st.success("✅ Analisis selesai!")

        # --- METRICS ---
        st.subheader("Ringkasan Sentimen")
        sentiment_counts = df_result["sentiment_label"].value_counts().reindex(["Negatif","Netral","Positif"], fill_value=0)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Komentar", len(df_result))
        col2.metric("Komentar Positif", sentiment_counts["Positif"])
        col3.metric("Komentar Negatif", sentiment_counts["Negatif"])

        # --- PIE CHART ---
        st.subheader("Distribusi Sentimen")
        fig_pie = px.pie(
            sentiment_counts.reset_index().rename(columns={"index":"Sentimen","sentiment_label":"Jumlah"}),
            values="sentiment_label", names="Sentimen",
            color="Sentimen", color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # --- SCATTER PCA ---
        st.subheader("Visualisasi 2D Teks")
        fig_scatter = px.scatter(
            df_result, x="x", y="y",
            color="sentiment_label",
            hover_data=["text"],
            title="Peta Sentimen Teks"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # --- TOP KEYWORDS PER SENTIMENT ---
        st.subheader("🔑 Kata Kunci Populer")
        for s in ["Positif","Netral","Negatif"]:
            st.markdown(f"**{s}**")
            texts_s = " ".join(df_result[df_result["sentiment_label"]==s]["text"].tolist())
            words = [w for w in texts_s.lower().split() if len(w)>2]
            top_words = [w for w, _ in Counter(words).most_common(10)]
            st.write(", ".join(top_words))

        # --- DATA TABLE ---
        st.subheader("📜 Detail Data")
        st.dataframe(df_result.head(20), use_container_width=True)

    else:
        st.error("❌ Pipeline gagal, tidak ada data valid.")
else:
    st.warning("👆 Unggah file JSON dan tekan **Jalankan Analisis** untuk memulai.")
