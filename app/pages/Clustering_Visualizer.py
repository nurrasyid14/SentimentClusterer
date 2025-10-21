#Clustering_Visualizer.py

import streamlit as st
import pandas as pd
from pathlib import Path
from models.visualizer import Visualizer

DATA_PROCESSED = Path("data/processed")
CLUSTER_FILE = DATA_PROCESSED / "cluster_results.pkl"

st.title("Cluster Visualizer")

if CLUSTER_FILE.exists():
    clusters = pd.read_pickle(CLUSTER_FILE)
    vis = Visualizer()
    fig = vis.scatter(clusters["vectors"].tolist(), clusters["labels"])
    st.plotly_chart(fig)
else:
    st.warning("File cluster_results.pkl belum tersedia. Jalankan pipeline dulu.")
