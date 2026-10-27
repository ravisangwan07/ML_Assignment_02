"""Train, tune, evaluate and save the five classifiers."""
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import utils
from data_prep import RANDOM_STATE, load_data, split_data


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


def main():
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)
    utils.MODEL_DIR.mkdir(exist_ok=True)

    results = [fit_and_evaluate(name, est, grid, X_train, y_train, X_test, y_test)
               for name, (est, grid) in model_specs().items()]

    table = pd.DataFrame([{"ML Model Name": r["name"], **r["scores"]} for r in results])
    print("\n=== TEST-SET COMPARISON ===")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()