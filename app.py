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

model = models[chosen]
scores, y_pred, y_proba = utils.predict_and_score(model, X, y)

tab_scores, tab_matrix, tab_compare, tab_sample = st.tabs(
    ["📊 Scores", "🧩 Confusion matrix & report", "🏁 Model comparison", "🔍 Sample inspector"])

# ----------------------------------------------------------------- tab 1: scores
with tab_scores:
    st.subheader(f"{chosen} on {len(frame)} rows")
    for col, key in zip(st.columns(len(utils.METRIC_NAMES)), utils.METRIC_NAMES):
        value = scores[key]
        col.metric(key, "n/a" if np.isnan(value) else f"{value:.4f}")
    st.caption("Precision, Recall and F1 are macro-averaged over the classes; "
               "AUC is one-vs-rest, macro-averaged.")
    predictions = pd.DataFrame({"true_label": y, "predicted_label": y_pred,
                                "confidence": y_proba.max(axis=1).round(4)})
    st.download_button("Download predictions (CSV)",
                       predictions.to_csv(index=False).encode("utf-8"),
                       file_name="predictions.csv", mime="text/csv")

# ----------------------------------------------------------------- tab 2: matrix + report
with tab_matrix:
    cm, report = utils.confusion_and_report(y, y_pred, CLASSES)
    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        fig, ax = plt.subplots(figsize=(5.5, 4.6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
        ax.set_xlabel("Predicted digit")
        ax.set_ylabel("True digit")
        ax.set_title(chosen)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Classification report")
        st.dataframe(report.round(4))

# ----------------------------------------------------------------- tab 3: comparison
with tab_compare:
    st.subheader("All five models on the same data")
    board = pd.DataFrame({name: utils.predict_and_score(m, X, y)[0]
                          for name, m in models.items()}).T
    board.index.name = "ML Model Name"
    st.dataframe(board.round(4))
    metric_pick = st.selectbox("Metric to chart", utils.METRIC_NAMES)
    st.bar_chart(board[metric_pick])
    leader = board[metric_pick].idxmax()
    st.info(f"Highest {metric_pick} on this data: {leader} ({board.loc[leader, metric_pick]:.4f})")

# ----------------------------------------------------------------- tab 4: sample inspector
with tab_sample:
    st.subheader("Inspect one row")
    row = int(st.number_input("Row number", min_value=0, max_value=len(frame) - 1,
                              value=0, step=1))
    pic, info = st.columns([1, 2])
    side = int(round(len(FEATURES) ** 0.5))
    with pic:
        if side * side == len(FEATURES):          # square number of pixels -> draw the image
            fig, ax = plt.subplots(figsize=(2.6, 2.6))
            ax.imshow(X.iloc[row].to_numpy(dtype=float).reshape(side, side), cmap="gray_r")
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.dataframe(X.iloc[[row]].T)
    with info:
        st.write(f"True label: **{y.iloc[row]}**   |   Predicted: **{y_pred[row]}**")
        if y.iloc[row] == y_pred[row]:
            st.success("Correct prediction")
        else:
            st.warning("Misclassified")
        st.bar_chart(pd.Series(y_proba[row], index=[str(c) for c in model.classes_]))

with st.expander("About this app"):
    st.markdown(
        f"Each model is a scikit-learn `Pipeline` (median imputer, standard scaler, classifier) "
        f"saved with joblib. Hyper-parameters were tuned with 5-fold cross-validation on the "
        f"training split ({int((1 - meta['test_size']) * 100)}% of the data); this app only calls "
        f"`predict` and `predict_proba` on the file you provide.")