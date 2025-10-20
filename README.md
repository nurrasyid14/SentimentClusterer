# 🧠 Sentiment Cluster Analyzer

Oleh:

- Muhammad Ikhwan Fitoriqillah (3324600002)
- Nicolaus Prima Dharma Nugroho (3324600016)
- Muhamad Nur Rasyid (3324600018)
- Kayla Nuansa Ceria (3324600023)
- Dave Benaya Walujo (3324600025)
- Zarah Berliana Erinawati (3324600027)
- Intan Azzuhra Permadani (3324600028)

Aplikasi **analisis sentimen dan klasterisasi teks** berbasis **Streamlit**, yang secara otomatis:
1. Memuat data hasil scraping (format `.json`)
2. Melakukan pembersihan dan tokenisasi teks
3. Membuat representasi vektor (TF-IDF)
4. Melatih model sentimen dummy (sementara)
5. Mengelompokkan komentar menggunakan **K-Means** atau **Fuzzy C-Means**
6. Menampilkan hasilnya secara **interaktif** dalam bentuk grafik dan tabel.

---

## 🚀 Fitur Utama

- 📥 **Upload file JSON** hasil scraping dari media sosial atau sumber lain.  
- 🧹 **Pembersihan teks otomatis**: lowercasing, tokenisasi, stopword removal.  
- 🔠 **Pembuatan embedding** menggunakan **TF-IDF Vectorizer**.  
- 🤖 **Analisis sentimen dummy (0=Negatif, 1=Netral, 2=Positif)**.  
- 🌀 **Clustering adaptif** dengan dua metode:
  - `K-Means`
  - `Fuzzy C-Means`
- 📊 **Dashboard interaktif Streamlit** berisi:
  - Ringkasan metrik utama
  - Distribusi cluster dalam bentuk pie chart
  - Peta sebaran 2D (scatter plot)
  - Tabel teks dan hasil klasifikasinya.

---

## 🧩 Arsitektur Pipeline
```
📦 SentimentClusterer
├── [.streamlit]
│   └── config.toml
├── [app]
│   ├── [utils]
│   │   ├── __init__.py
│   │   └── ui_helpers.py
│   ├── __init__.py
│   ├── Clustering_Visualizer.py
│   ├── Data_Explorer.py
│   ├── Home.py
│   ├── Sentiment_Analyzer.py
│   └── Settings.py
├── [data]
│   ├── [logs]
│   │   └── (EMPTY DIRECTORY)
│   ├── [processed]
│   │   ├── parsed_comments.pkl
│   │   └── tokens.pkl
│   ├── [raw]
│   │   └── (EMPTY DIRECTORY)
│   └── __init__.py
├── [documentations]
│   └── (EMPTY DIRECTORY)
├── [models]
│   ├── [__pycache__]
│   │   └── (EMPTY DIRECTORY)
│   ├── [stopwords]
│   │   ├── stopwords_en.txt
│   │   └── stopwords_id.txt
│   ├── __init__.py
│   ├── cluster_to_pkl.py
│   ├── clustering.py
│   ├── embeddings_builder.py
│   ├── sentiment_mapper.py
│   ├── vectorizer.py
│   └── visualizer.py
├── [pipeline]
│   ├── [__pycache__]
│   │   └── (EMPTY DIRECTORY)
│   ├── __init__.py
│   ├── main.py
│   ├── parser.py
│   ├── preprocess.py
│   ├── translator.py
│   └── utils.py
├── [visualizations]
│   └── (EMPTY DIRECTORY)
├── .env
├── app.py
├── requirements.txt
└── Timnas_Gagal_Piala_Dunia.json
```

---

## 🧠 Alur Proses (Pipeline)

1. **Upload JSON**  
   File berisi daftar komentar/tweet (kolom seperti `full_text`, `text`, `content`, dll).

2. **Parsing JSON → teks mentah**  
   Dilakukan oleh `JSONParser.parse()`, menyimpan hasil ke `parsed_comments.pkl`.

3. **Preprocessing**  
   Membersihkan teks dari karakter khusus, tokenisasi, dan stopword removal.

4. **Embedding dan Dummy Sentiment Training**  
   Menggunakan `TfidfVectorizer` + `LogisticRegression` untuk membuat representasi numerik teks.

5. **Clustering**  
   - `KMeansClustering`: pembagian cluster secara tegas  
   - `FuzzyCMeansClustering`: pembagian probabilistik antar cluster

6. **Visualisasi Hasil**  
   Ditampilkan langsung di Streamlit:
   - Metrik utama (jumlah komentar, jumlah cluster, cluster dominan)
   - Pie chart distribusi cluster
   - Scatter plot sebaran komentar per cluster
   - DataFrame hasil analisis

---

## ⚙️ App
https://sentimentclusterer7.streamlit.app/

📊 Contoh Tampilan
### Dashboard Utama: Upload JSON dan pilih metode klasterisasi

### Visualisasi Pie Chart: Distribusi cluster berdasarkan hasil sentimen

### Scatter Plot: Sebaran komentar dalam ruang vektor
#### 🧭 Interpretasi Kuadran pada Scatter Plot

| Kuadran                        | Posisi Titik | Interpretasi Umum                                                     | Kemungkinan Arti dalam Konteks Sentimen                                                                                |
| ------------------------------ | ------------ | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| *Kuadran I (x > 0, y > 0)*   | kanan–atas   | Nilai positif di kedua dimensi (misalnya optimisme + semangat tinggi) | 🟢 *Sentimen positif yang kuat*, misalnya komentar yang penuh dukungan atau pujian.                                  |
| *Kuadran II (x < 0, y > 0)*  | kiri–atas    | Dimensi pertama negatif tapi dimensi kedua positif (konflik makna)    | 🟡 *Netral cenderung positif*, bisa berupa komentar yang mengandung kritik tapi tetap sopan atau mendukung sebagian. |
| *Kuadran III (x < 0, y < 0)* | kiri–bawah   | Kedua dimensi negatif                                                 | 🔴 *Sentimen negatif kuat*, seperti komentar sinis, marah, atau kecewa.                                              |
| *Kuadran IV (x > 0, y < 0)*  | kanan–bawah  | Dimensi pertama positif tapi kedua negatif                            | 🟠 *Ambigu atau netral cenderung negatif*, misalnya sindiran halus atau ekspresi kecewa yang tidak frontal.          |


### Tabel Hasil: Menampilkan teks beserta cluster dan label sentimennya

