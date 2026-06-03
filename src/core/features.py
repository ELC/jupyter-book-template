from typing import NamedTuple

import numpy as np
import pandas as pd
from pandera.typing import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_union
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures

from core.fourier_features import FourierFeatures
from core.schemas import SplitDatasetBase, SplitKind, TrainingData
from core.settings import Settings
from core.splits import select_split


class PreparedSplit(NamedTuple):
    x: np.ndarray
    y: pd.Series


def _polynomial_transformer(settings: Settings) -> PolynomialFeatures | None:
    if settings.polynomial_degree <= 0:
        return None
    return PolynomialFeatures(
        degree=settings.polynomial_degree,
        include_bias=False,
    )


def _fourier_transformer(settings: Settings) -> FourierFeatures | None:
    if settings.fourier_terms <= 0:
        return None
    return FourierFeatures(
        n_terms=settings.fourier_terms,
        frequency=settings.seasonality_frequency,
    )


def _transformers_from_settings(settings: Settings) -> list[TransformerMixin]:
    transformers: list[TransformerMixin] = []
    polynomial = _polynomial_transformer(settings)
    if polynomial is not None:
        transformers.append(polynomial)
    fourier = _fourier_transformer(settings)
    if fourier is not None:
        transformers.append(fourier)
    return transformers


def expand_features(settings: Settings) -> BaseEstimator:
    transformers = _transformers_from_settings(settings)
    if not transformers:
        return FunctionTransformer(validate=False)
    return make_union(*transformers)


def prepare_split(
    data: DataFrame[SplitDatasetBase],
    split: SplitKind,
) -> PreparedSplit:
    subset = select_split(data, split)
    return PreparedSplit(
        x=subset[[TrainingData.x]].to_numpy(),
        y=subset[TrainingData.y],
    )
