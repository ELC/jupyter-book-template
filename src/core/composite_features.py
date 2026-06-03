from collections.abc import Sequence
from typing import Self

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from core._feature_columns import as_column


class CompositeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, transformers: Sequence[TransformerMixin] | None = None) -> None:
        self.transformers = transformers or []

    def __sklearn_is_fitted__(self) -> bool:  # noqa: PLW3201
        if not self.transformers:
            return True
        for transformer in self.transformers:
            try:
                check_is_fitted(transformer)
            except NotFittedError:
                return False
        return True

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> Self:
        column = as_column(x)
        for transformer in self.transformers:
            transformer.fit(column, y)
        return self

    def fit_transform(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        column = as_column(x)
        if not self.transformers:
            return column
        blocks = [transformer.fit_transform(column, y) for transformer in self.transformers]
        return np.hstack(blocks)

    def transform(self, x: np.ndarray) -> np.ndarray:
        column = as_column(x)
        if not self.transformers:
            return column
        blocks = [transformer.transform(column) for transformer in self.transformers]
        return np.hstack(blocks)
