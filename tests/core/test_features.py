import numpy as np
import pytest
from pandera.typing import DataFrame
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted

from core import (
    FourierFeatures,
    Settings,
    SplitDatasetBase,
    SplitKind,
    TrainingData,
    expand_features,
    prepare_split,
    select_split,
)


def _column(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1, 1)


def test_expand_features_transform_requires_fit() -> None:
    transformer = expand_features(Settings(polynomial_degree=2, fourier_terms=0))
    with pytest.raises(NotFittedError):
        transformer.transform(_column(np.array([1.0, 2.0])))


def test_expand_features_raw_x_only() -> None:
    transformer = expand_features(Settings(polynomial_degree=0, fourier_terms=0))
    x = _column(np.array([1.0, 2.0]))
    fitted = transformer.fit_transform(x)
    check_is_fitted(transformer)
    np.testing.assert_allclose(transformer.transform(x), fitted)
    assert isinstance(transformer, FunctionTransformer)


def test_prepare_split_returns_two_dimensional_x(
    split_dataset: DataFrame[SplitDatasetBase],
) -> None:
    prepared = prepare_split(split_dataset, SplitKind.TRAINING)
    expected = select_split(split_dataset, SplitKind.TRAINING)
    np.testing.assert_allclose(prepared.x[:, 0], expected[TrainingData.x].to_numpy())
    assert prepared.x.ndim == 2
    assert prepared.y.tolist() == expected[TrainingData.y].tolist()


def test_expand_features_reuses_polynomial_state() -> None:
    settings = Settings(polynomial_degree=2, fourier_terms=0)
    transformer = expand_features(settings)
    train_x = _column(np.array([1.0, 2.0, 3.0]))
    eval_x = _column(np.array([4.0]))
    transformer.fit_transform(train_x)
    eval_features = transformer.transform(eval_x)
    np.testing.assert_allclose(eval_features, np.array([[4.0, 16.0]]))


def test_expand_features_column_count(
    expand_features_settings: Settings,
    expected_feature_columns: int,
    expand_features_column_count_x: np.ndarray,
) -> None:
    x = _column(expand_features_column_count_x)
    features = expand_features(expand_features_settings).fit_transform(x)
    assert features.shape == (len(expand_features_column_count_x), expected_feature_columns)


def test_expand_features_polynomial_values(
    polynomial_expand_features_settings: Settings,
    polynomial_expand_features_x: np.ndarray,
) -> None:
    features = expand_features(polynomial_expand_features_settings).fit_transform(
        _column(polynomial_expand_features_x),
    )
    np.testing.assert_allclose(features[:, 0], polynomial_expand_features_x)
    np.testing.assert_allclose(features[:, 1], polynomial_expand_features_x**2)


def test_expand_features_polynomial_and_fourier_values(
    polynomial_and_fourier_expand_features_settings: Settings,
    expand_features_column_count_x: np.ndarray,
) -> None:
    settings = polynomial_and_fourier_expand_features_settings
    features = expand_features(settings).fit_transform(_column(expand_features_column_count_x))
    x = expand_features_column_count_x
    frequency = settings.seasonality_frequency
    np.testing.assert_allclose(features[:, 0], x)
    np.testing.assert_allclose(features[:, 1], x**2)
    np.testing.assert_allclose(features[:, 2], x**3)
    np.testing.assert_allclose(features[:, 3], np.sin(frequency * x))
    np.testing.assert_allclose(features[:, 4], np.sin(2 * frequency * x))
    np.testing.assert_allclose(features[:, 5], np.cos(frequency * x))
    np.testing.assert_allclose(features[:, 6], np.cos(2 * frequency * x))


def test_expand_features_fourier_values(
    fourier_expand_features_settings: Settings,
    fourier_expand_features_x: np.ndarray,
) -> None:
    features = expand_features(fourier_expand_features_settings).fit_transform(
        _column(fourier_expand_features_x),
    )
    frequency = fourier_expand_features_settings.seasonality_frequency
    np.testing.assert_allclose(features[:, 0], np.sin(frequency * fourier_expand_features_x))
    np.testing.assert_allclose(features[:, 1], np.cos(frequency * fourier_expand_features_x))


def test_fourier_features_transform_values(fourier_expand_features_settings: Settings) -> None:
    x = np.array([0.0, 1.0])
    transformer = FourierFeatures(
        n_terms=fourier_expand_features_settings.fourier_terms,
        frequency=fourier_expand_features_settings.seasonality_frequency,
    )
    features = transformer.fit_transform(_column(x))
    frequency = fourier_expand_features_settings.seasonality_frequency
    np.testing.assert_allclose(features[:, 0], np.sin(frequency * x))
    np.testing.assert_allclose(features[:, 1], np.cos(frequency * x))


def test_expand_features_returns_feature_union() -> None:
    settings = Settings(polynomial_degree=2, fourier_terms=1, seasonality_frequency=0.5)
    transformer = expand_features(settings)
    assert isinstance(transformer, FeatureUnion)
    assert len(transformer.transformer_list) == 2
