import numpy as np


def as_column(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1, 1)
