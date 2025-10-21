#Settings.py

import streamlit as st

st.title("Settings Dashboard")

st.header("Cluster & Model Configuration")

k_clusters = st.number_input("Jumlah cluster (k):", min_value=2, max_value=10, value=3)
cluster_method = st.selectbox("Metode clustering:", ["kmeans", "fuzzy", "kmodes"])

st.write(f"Cluster k={k_clusters}, metode={cluster_method}")
st.info("Pengaturan ini akan diterapkan saat pipeline dijalankan.")
