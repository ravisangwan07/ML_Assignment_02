"""Train, tune, evaluate and save the five classifiers."""
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import json
import utils
from data_prep import RANDOM_STATE, TARGET, TEST_SIZE, load_data, split_data

def build_pipeline(estimator):
    """Same preprocessing for every model: fill gaps with the median, then standardise."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


def model_specs():
    """name -> (estimator, hyper-parameter grid). Grid keys start with 'model__'."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
            {"model__C": [0.01, 0.1, 1, 10]}),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"model__max_depth": [5, 10, 15, None],
             "model__min_samples_leaf": [1, 3, 5]}),
        "KNN": (
            KNeighborsClassifier(),
            {"model__n_neighbors": [3, 5, 7, 9],
             "model__weights": ["uniform", "distance"]}),
        "Naive Bayes": (
            GaussianNB(),
            {"model__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1, 1.0]}),
        "Support Vector Machine": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {"model__C": [1, 10, 100],
             "model__gamma": ["scale", 0.001, 0.01]}),
    }


def fit_and_evaluate(name, estimator, param_grid, X_train, y_train, X_test, y_test):
    """Tune with 5-fold CV on the TRAIN split, save the best pipeline, score on the TEST split."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(build_pipeline(estimator), param_grid, cv=cv,
                          scoring="accuracy", n_jobs=-1)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    joblib.dump(best, utils.model_path(name))

    scores = utils.score_predictions(y_test, best.predict(X_test),
                                     best.predict_proba(X_test), best.classes_)
    params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    print(f"{name:<24s} cv_acc={search.best_score_:.4f}  test_acc={scores['Accuracy']:.4f}  {params}")
    return {"name": name, "scores": scores, "best_params": params,
            "cv_accuracy": float(search.best_score_)}

def save_artifacts(table, results, data, X_train):
    """metrics.csv for the README + metadata.json that the Streamlit app reads."""
    table.to_csv(utils.MODEL_DIR / "metrics.csv", index=False)
    meta = {
        "dataset_name": "Optical Recognition of Handwritten Digits (UCI)",
        "target_name": TARGET,
        "feature_names": X_train.columns.tolist(),
        "classes": [int(c) for c in sorted(data[TARGET].unique())],
        "n_rows": int(len(data)),
        "n_features": int(X_train.shape[1]),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "best_params": {r["name"]: r["best_params"] for r in results},
        "cv_accuracy": {r["name"]: r["cv_accuracy"] for r in results},
    }
    with open(utils.MODEL_DIR / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

def main():
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)
    utils.MODEL_DIR.mkdir(exist_ok=True)

    results = [fit_and_evaluate(name, est, grid, X_train, y_train, X_test, y_test)
               for name, (est, grid) in model_specs().items()]

    table = pd.DataFrame([{"ML Model Name": r["name"], **r["scores"]} for r in results])
    save_artifacts(table, results, data, X_train)
    print("\n=== TEST-SET COMPARISON ===")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()