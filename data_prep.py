"""Data loading and splitting for the handwritten-digits classification assignment."""
from sklearn.datasets import load_digits

TARGET = "digit"


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