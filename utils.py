"""Helper functions shared by the training script and the Streamlit app."""
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from pathlib import Path

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
MODEL_DIR = Path(__file__).resolve().parent / "model"

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