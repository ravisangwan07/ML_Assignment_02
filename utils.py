import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)


METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes",
               "Support Vector Machine"]

def model_path(name):
    """Path of the saved pipeline for a display name such as 'Logistic Regression'."""
    return MODEL_DIR / (name.lower().replace(" ", "_") + ".joblib")

def score_predictions(y_true, y_pred, y_proba, classes):
    """Compute the six required metrics and return them in a dict.

    Binary problem : precision/recall/F1 for the positive class (classes[1]).
    Multi-class    : precision/recall/F1 macro-averaged; AUC one-vs-rest, macro-averaged.
    """
    classes = list(classes)
    is_binary = len(classes) == 2
    opts = {"average": "binary" if is_binary else "macro", "zero_division": 0}
    if is_binary:
        opts["pos_label"] = classes[1]
    try:
        if is_binary:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr",
                                average="macro", labels=classes)
    except ValueError:            # happens when a class is absent from y_true
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, **opts),
        "Recall": recall_score(y_true, y_pred, **opts),
        "F1": f1_score(y_true, y_pred, **opts),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

def load_metadata():
    """Read model/metadata.json written by train_models.py."""
    with open(MODEL_DIR / "metadata.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_model(name):
    """Load a saved, already-trained pipeline (preprocessing + classifier)."""
    return joblib.load(model_path(name))


def predict_and_score(model, X, y):
    """Predict with a trained pipeline (NO re-fitting) and return (scores, y_pred, y_proba)."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    return score_predictions(y, y_pred, y_proba, model.classes_), y_pred, y_proba


def check_upload(frame, feature_names, target_name):
    """Return a list of problems with an uploaded test file (empty list = file is fine)."""
    problems = []
    if len(frame) == 0:
        problems.append("The file has no rows.")
    missing = [c for c in list(feature_names) + [target_name] if c not in frame.columns]
    if missing:
        shown = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        problems.append(f"Missing required column(s): {shown}")
    return problems


def confusion_and_report(y_true, y_pred, classes):
    """Confusion-matrix array and a classification-report DataFrame."""
    labels = list(classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels,
                                   output_dict=True, zero_division=0)
    return cm, pd.DataFrame(report).T