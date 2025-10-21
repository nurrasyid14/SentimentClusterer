#ui_helpers.py

import streamlit as st

def info_box(message: str):
    st.info(message)

def warning_box(message: str):
    st.warning(message)

def success_box(message: str):
    st.success(message)
