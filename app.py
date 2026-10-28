"""Streamlit front-end for the digit-classification assignment.

Loads the five saved scikit-learn pipelines from model/ and evaluates them on a test CSV
that the user uploads. Nothing is re-trained here.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

import utils

st.set_page_config(page_title="Digit Classifier Lab", page_icon="🔢", layout="wide")

SAMPLE_CSV = Path(__file__).resolve().parent / "test_data.csv"


@st.cache_resource(show_spinner=False)
def load_all_models():
    return {name: utils.load_model(name) for name in utils.MODEL_NAMES}


@st.cache_data(show_spinner=False)
def read_meta():
    return utils.load_metadata()


meta = read_meta()
models = load_all_models()
FEATURES, TARGET, CLASSES = meta["feature_names"], meta["target_name"], meta["classes"]

st.title("🔢 Digit Classifier Lab")
st.caption(f"{meta['dataset_name']}  |  {meta['n_rows']} images, {meta['n_features']} pixel "
           f"features, {len(CLASSES)} classes  |  five pre-trained models, no retraining here")

# ----------------------------------------------------------------- sidebar controls
with st.sidebar:
    st.header("Controls")
    chosen = st.selectbox("1. Pick a model", utils.MODEL_NAMES)
    upload = st.file_uploader("2. Upload a test CSV", type="csv")
    use_demo = st.checkbox("Use the bundled test_data.csv", value=upload is None)
    if SAMPLE_CSV.exists():
        st.download_button("Download sample CSV", SAMPLE_CSV.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")
    st.caption(f"The file needs {len(FEATURES)} feature columns plus the target column "
               f"'{TARGET}'.")

# ----------------------------------------------------------------- load + validate data
try:
    if upload is not None:
        frame, origin = pd.read_csv(upload), f"uploaded file {upload.name}"
    elif use_demo and SAMPLE_CSV.exists():
        frame, origin = pd.read_csv(SAMPLE_CSV), "bundled test_data.csv"
    else:
        st.info("👈 Upload a test CSV, or tick the bundled-sample box, to begin.")
        st.stop()
except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
    st.error("That file could not be read as a CSV.")
    st.stop()

issues = utils.check_upload(frame, FEATURES, TARGET)
if issues:
    for msg in issues:
        st.error(msg)
    st.stop()

X, y = frame[FEATURES], frame[TARGET]
unseen = sorted(set(y.unique()) - set(CLASSES))
if unseen:
    st.error(f"The target column has labels the models were never trained on: {unseen}")
    st.stop()
st.success(f"Loaded {origin}: {len(frame)} rows")