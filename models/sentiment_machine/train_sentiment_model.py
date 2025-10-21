# models/train_sentiment_model.py
"""
Skrip ini bertanggung jawab untuk melatih model sentimen (SentimentEngine)
menggunakan dataset berlabel dan menyimpannya sebagai file .pkl untuk digunakan nanti.
"""

import os
import joblib
import pandas as pd
import logging
from pathlib import Path

# Impor kelas yang diperlukan dari skrip Anda yang lain.
# Path relatif ini penting agar bisa dijalankan dari direktori utama.
try:
    # Coba impor seolah-olah dijalankan dari direktori utama
    from models.sentiment_mapper import SentimentEngine
    from pipeline.preprocess import clean_and_tokenize_text
except ImportError:
    # Fallback jika dijalankan langsung dari dalam folder models/
    # Ini membuat skrip lebih fleksibel.
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from models.sentiment_mapper import SentimentEngine
    from pipeline.preprocess import clean_and_tokenize_text


# Setup logging untuk melihat proses
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_save_model(training_csv_path: str, output_model_path: str):
    """
    Memuat data latih, melatih SentimentEngine, dan menyimpannya.
    """
    logging.info("Memulai proses pelatihan model sentimen...")

    # 1. Muat Dataset Latih Berlabel dari Fase 1
    try:
        df_train = pd.read_csv(training_csv_path)
        logging.info(f"Berhasil memuat {len(df_train)} baris data dari {training_csv_path}")
    except FileNotFoundError:
        logging.error(f"FATAL: File data latih tidak ditemukan di {training_csv_path}. Pastikan file sudah dibuat di Fase 1.")
        return
    except Exception as e:
        logging.error(f"Gagal memuat file CSV: {e}")
        return

    # Validasi kolom yang dibutuhkan
    if 'text' not in df_train.columns or 'label' not in df_train.columns:
        logging.error("FATAL: File CSV harus memiliki kolom 'text' dan 'label'.")
        return

    # Mengabaikan baris yang tidak memiliki teks
    df_train.dropna(subset=['text'], inplace=True)
    texts = df_train['text'].tolist()
    labels = df_train['label'].tolist()

    # 2. Preprocessing Teks Data Latih
    logging.info("Membersihkan teks data latih (preprocessing)...")

    # TfidfVectorizer butuh input string, jadi kita gabungkan token kembali setelah dibersihkan.
    # Ini memastikan data latih dan data prediksi nanti diproses dengan cara yang sama persis.
    # PERUBAHAN: Menggunakan nama fungsi yang benar 'clean_and_tokenize_text'
    clean_texts = [" ".join(clean_and_tokenize_text(text, lang='id')) for text in texts]

    # 3. Inisialisasi dan Latih SentimentEngine
    logging.info("Melatih SentimentEngine (TF-IDF + Logistic Regression)...")
    sentiment_engine = SentimentEngine()

    try:
        sentiment_engine.prepare_and_train(clean_texts, labels)
        logging.info("✅ Pelatihan model sentimen selesai.")
    except Exception as e:
        logging.error(f"Terjadi error saat pelatihan: {e}")
        return

    # 4. Simpan Model yang Sudah Dilatih
    try:
        Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(sentiment_engine, output_model_path)
        logging.info(f"💾 Model sentimen yang sudah dilatih berhasil disimpan ke: {output_model_path}")
    except Exception as e:
        logging.error(f"Gagal menyimpan model: {e}")

if __name__ == "__main__":
    # Tentukan path relatif dari direktori utama proyek
    TRAINING_DATA_PATH = 'data/training_data.csv'
    MODEL_OUTPUT_PATH = 'models/sentiment_model.pkl'

    # Jalankan fungsi utama
    train_and_save_model(TRAINING_DATA_PATH, MODEL_OUTPUT_PATH)

