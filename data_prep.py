"""Data loading and splitting for the handwritten-digits classification assignment."""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "digit"

def split_data(data):
    """Stratified train/test split so every digit appears in both parts in the same proportion."""
    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                            random_state=RANDOM_STATE)

def load_data():
    """Return the UCI digits data as one DataFrame (64 pixel columns + target)."""
    frame = load_digits(as_frame=True).frame
    return frame.rename(columns={"target": TARGET})


if __name__ == "__main__":
    data = load_data()
    n_rows, n_cols = data.shape
    print(f"rows={n_rows}, features={n_cols - 1}")
    print(data[TARGET].value_counts().sort_index().to_dict())
    assert n_rows >= 500 and n_cols - 1 >= 12, "dataset too small for the assignment"
    X_train, X_test, y_train, y_test = split_data(data)
    print(f"train={len(X_train)}, test={len(X_test)}")