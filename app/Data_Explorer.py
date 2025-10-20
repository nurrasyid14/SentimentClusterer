import streamlit as st
import pandas as pd
from pathlib import Path

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")

st.set_page_config(page_title="Data Explorer", layout="wide")
st.title("Data Explorer")

option = st.radio("Pilih dataset:", ["Raw JSON", "Processed CSV/PKL"])

if option == "Raw JSON":
    files = list(DATA_RAW.glob("*.json"))
    file_selected = st.selectbox("Pilih file:", files)
    if file_selected:
        df = pd.read_json(file_selected)
        st.dataframe(df)

elif option == "Processed CSV/PKL":
    files = list(DATA_PROCESSED.glob("*.csv")) + list(DATA_PROCESSED.glob("*.pkl"))
    file_selected = st.selectbox("Pilih file:", files)
    if file_selected.suffix == ".csv":
        df = pd.read_csv(file_selected)
    else:
        df = pd.read_pickle(file_selected)
    st.dataframe(df)
