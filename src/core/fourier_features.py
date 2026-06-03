from typing import Self

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FourierFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, n_terms: int = 1, frequency: float = 0.5) -> None:
        self.n_terms = n_terms
        self.frequency = frequency

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> Self:
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        values = x[:, 0]
        harmonics = np.arange(1, self.n_terms + 1, dtype=float)
        angles = np.outer(values, harmonics * self.frequency)
        return np.hstack((np.sin(angles), np.cos(angles)))
