from collections.abc import Sequence
from typing import Self

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.utils._repr_html.estimator import _VisualBlock  # noqa: PLC2701


class MultiRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
    """Bundle several named regressors into a single sklearn-native composite.

    `.transform(X)` returns a `(n_samples, n_estimators)` matrix of per-base
    predictions; `.predict(X)` averages them. Unfitted estimator specs remain
    available via `.estimators` for callers that need to `clone()` them
    (e.g. bootstrap, conformal calibration).
    """

    named_estimators_: dict[str, BaseEstimator]
    estimators_: list[BaseEstimator]

    def __init__(self, estimators: Sequence[tuple[str, BaseEstimator]]) -> None:
        self.estimators = estimators
        self.named_estimators_ = {}
        self.estimators_ = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> Self:
        self.named_estimators_ = {
            name: clone(estimator).fit(x, y) for name, estimator in self.estimators
        }
        self.estimators_ = list(self.named_estimators_.values())
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack([estimator.predict(x) for estimator in self.estimators_])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.transform(x).mean(axis=1)

    @property
    def names_(self) -> list[str]:
        return [name for name, _ in self.estimators]

    def _sk_visual_block_(self) -> _VisualBlock:  # noqa: PLW3201
        names = [name for name, _ in self.estimators]
        estimators = [estimator for _, estimator in self.estimators]
        return _VisualBlock("parallel", estimators, names=names)
