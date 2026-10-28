# Handwritten Digit Classification with 5 ML Models and a Streamlit App

**Course:** Machine Learning, M.Tech (AIML), BITS Pilani WILP | **Assignment 2**

## a. Problem statement

Handwritten digits are scanned and reduced to 8x8 grey-level images. The task is to predict which digit (0-9)
an image shows from its 64 pixel-intensity values, which makes this a **10-class classification problem**.
Five classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes and a
Support Vector Machine) are trained on the same data and compared with six metrics: Accuracy, AUC, Precision,
Recall, F1 and MCC. The saved models are served through a Streamlit web app that evaluates any uploaded
test CSV **without re-training**.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Optical Recognition of Handwritten Digits |
| Source | UCI Machine Learning Repository, loaded through `sklearn.datasets.load_digits` |
| Instances | 1797 (minimum required: 500) |
| Features | 64 numeric pixel intensities with integer values 0-16 (minimum required: 12) |
| Target | `digit`, 10 classes (0-9), about 180 samples per class |
| Missing values | None |
| Split | 80 % train (1437) / 20 % test (360), stratified, `random_state = 42` |
| Test file | `test_data.csv` (360 rows: 64 feature columns plus `digit`) |

**Pre-processing:** median imputer followed by `StandardScaler`, fitted on the training split only and stored
inside every saved `Pipeline`, so the app applies the identical transformation to uploaded data.
**Tuning:** hyper-parameters were chosen by 5-fold stratified cross-validated grid search on the training split
only; the test split was used once, for the final metrics.

## c. GitHub repository link

- Repository: https://github.com/ravisangwan07/ML_Assignment_02
- Live Streamlit app: https://mlassignment02-2026aj05407.streamlit.app

```
ML_Assignment_02/
|-- app.py               Streamlit application
|-- utils.py             shared helpers: metrics, model loading, validation
|-- data_prep.py         dataset loading, stratified split, test_data.csv
|-- train_models.py      training, tuning, evaluation and model saving
|-- requirements.txt
|-- README.md
|-- test_data.csv        held-out test data (features and target)
|-- model/               saved pipelines (.joblib), metadata.json, metrics.csv
```

**Run locally:** `pip install -r requirements.txt`, then `python train_models.py` (optional, regenerates the
models), then `streamlit run app.py`.

## d. Models used

All five models use the same training split and are evaluated on the same 360-row test split. Precision, Recall
and F1 are macro-averaged over the 10 classes; AUC is one-vs-rest, macro-averaged.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9722 | 0.9991 | 0.9721 | 0.9719 | 0.9719 | 0.9692 |
| Decision Tree | 0.8028 | 0.9223 | 0.8039 | 0.8019 | 0.8014 | 0.7812 |
| KNN | 0.9667 | 0.9907 | 0.9678 | 0.9664 | 0.9663 | 0.9631 |
| Naive Bayes | 0.8917 | 0.9798 | 0.8947 | 0.8908 | 0.8918 | 0.8798 |
| Support Vector Machine | 0.9750 | 0.9995 | 0.9760 | 0.9748 | 0.9748 | 0.9723 |

Hyper-parameters chosen by cross-validation: Logistic Regression {'C': 1}; Decision Tree {'max_depth': 15, 'min_samples_leaf': 3};
KNN {'n_neighbors': 3, 'weights': 'uniform'}; Naive Bayes {'var_smoothing': 0.1}; SVM {'C': 1, 'gamma': 'scale'}.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | 97.22 % accuracy, AUC 0.9991, MCC 0.9692, with 10 of 360 test images wrong. A linear model does very well, so the digit classes are close to linearly separable once the pixels are standardised. The most frequent error is a true 8 predicted as 1. It trains in seconds and the saved model is under 10 KB. |
| Decision Tree | Weakest model: 80.28 % accuracy, MCC 0.7812, 71 errors. Splits on single pixels cannot describe stroke shapes well, and one tree has high variance. Its AUC (0.9223) is clearly lower than the rest because leaf nodes give almost hard probabilities. Only about 66 % of true 8s are found. |
| KNN | 96.67 % accuracy, AUC 0.9907 (k = 3, uniform weights, 12 errors). Images of the same digit sit close together after scaling, so a neighbour vote works well. The cost is that the model keeps the whole training set (the largest saved file, about 750 KB) and predicts more slowly. |
| Naive Bayes | 89.17 % accuracy and MCC 0.8798 (39 errors). The independence assumption is clearly broken because neighbouring pixels are strongly correlated. The AUC of 0.9798 is still high, so it ranks the classes well even when its top choice is wrong, but its probabilities are over-confident. A large `var_smoothing` (0.1) worked best. |
| Support Vector Machine | 97.50 % accuracy, AUC 0.9995, F1 0.9748, MCC 0.9723, with only 9 errors. The RBF kernel captures non-linear stroke patterns while the margin keeps variance low. It also had the best cross-validation accuracy (98.12 %), so the result is not just a lucky test split. |
| **Overall winner for this dataset** | **Support Vector Machine (RBF)**: best on all six metrics and on cross-validation accuracy. Its lead over Logistic Regression on the test set is one image, so Logistic Regression is a strong, much smaller alternative. Digit 8 is the hardest class for every model and is usually mistaken for 1. |

## Streamlit app features

1. Upload a test CSV, or use the bundled `test_data.csv`; a sample file can be downloaded from the sidebar.
2. Drop-down to choose one of the five models.
3. Six evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC) for the chosen model.
4. Confusion matrix and full classification report.
5. Extras: comparison of all five models on the uploaded data, a row inspector that draws the digit with the
   predicted class probabilities, and a download of the predictions as CSV.
