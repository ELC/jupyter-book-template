import numpy as np
import pytest
from pandera.typing import DataFrame
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_is_fitted

from core import CompositeFeatures, FourierFeatures, Settings, SplitKind, expand_features, select_split
from core.features import prepare_split
from core.schemas import SplitDatasetBase, TrainingData


def test_expand_features_transform_requires_fit() -> None:
    transformer = expand_features(Settings(polynomial_degree=2, fourier_terms=0))
    with pytest.raises(NotFittedError):
        transformer.transform(np.array([1.0, 2.0]))


def test_expand_features_raw_x_only() -> None:
    transformer = expand_features(Settings(polynomial_degree=0, fourier_terms=0))
    x = np.array([1.0, 2.0])
    fitted = transformer.fit_transform(x)
    check_is_fitted(transformer)
    np.testing.assert_allclose(transformer.transform(x), fitted)


def test_prepare_split_returns_raw_x_and_y(
    split_dataset: DataFrame[SplitDatasetBase],
) -> None:
    prepared = prepare_split(split_dataset, SplitKind.TRAINING)
    expected = select_split(split_dataset, SplitKind.TRAINING)
    np.testing.assert_allclose(prepared.x, expected[TrainingData.x].to_numpy())
    assert prepared.y.tolist() == expected[TrainingData.y].tolist()


def test_expand_features_reuses_polynomial_state() -> None:
    settings = Settings(polynomial_degree=2, fourier_terms=0)
    transformer = expand_features(settings)
    train_x = np.array([1.0, 2.0, 3.0])
    eval_x = np.array([4.0])
    transformer.fit_transform(train_x)
    eval_features = transformer.transform(eval_x)
    np.testing.assert_allclose(eval_features, np.array([[4.0, 16.0]]))


def test_expand_features_column_count(
    expand_features_settings: Settings,
    expected_feature_columns: int,
    expand_features_column_count_x: np.ndarray,
) -> None:
    features = expand_features(expand_features_settings).fit_transform(expand_features_column_count_x)
    assert features.shape == (len(expand_features_column_count_x), expected_feature_columns)


def test_expand_features_polynomial_values(
    polynomial_expand_features_settings: Settings,
    polynomial_expand_features_x: np.ndarray,
) -> None:
    features = expand_features(polynomial_expand_features_settings).fit_transform(
        polynomial_expand_features_x,
    )
    np.testing.assert_allclose(features[:, 0], polynomial_expand_features_x)
    np.testing.assert_allclose(features[:, 1], polynomial_expand_features_x**2)


def test_expand_features_polynomial_and_fourier_values(
    polynomial_and_fourier_expand_features_settings: Settings,
    expand_features_column_count_x: np.ndarray,
) -> None:
    settings = polynomial_and_fourier_expand_features_settings
    features = expand_features(settings).fit_transform(expand_features_column_count_x)
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
        fourier_expand_features_x,
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
    features = transformer.fit_transform(x)
    frequency = fourier_expand_features_settings.seasonality_frequency
    np.testing.assert_allclose(features[:, 0], np.sin(frequency * x))
    np.testing.assert_allclose(features[:, 1], np.cos(frequency * x))


def test_composite_features_hstacks_polynomial_and_fourier() -> None:
    settings = Settings(polynomial_degree=2, fourier_terms=1, seasonality_frequency=0.5)
    transformer = CompositeFeatures(
        transformers=[
            PolynomialFeatures(degree=settings.polynomial_degree, include_bias=False),
            FourierFeatures(n_terms=settings.fourier_terms, frequency=settings.seasonality_frequency),
        ],
    )
    x = np.array([1.0, 2.0])
    features = transformer.fit_transform(x)
    assert features.shape == (2, 4)
    np.testing.assert_allclose(features[:, 0], x)
    np.testing.assert_allclose(features[:, 1], x**2)
    np.testing.assert_allclose(features[:, 2], np.sin(settings.seasonality_frequency * x))
    np.testing.assert_allclose(features[:, 3], np.cos(settings.seasonality_frequency * x))


def test_composite_features_is_fitted_reports_state_per_transformer() -> None:
    transformer = CompositeFeatures(transformers=[FourierFeatures(n_terms=1, frequency=0.5)])
    assert transformer.__sklearn_is_fitted__() is False
    transformer.fit(np.array([1.0, 2.0]))
    assert transformer.__sklearn_is_fitted__() is True


def test_composite_features_fit_then_transform() -> None:
    transformer = CompositeFeatures(transformers=[FourierFeatures(n_terms=1, frequency=0.5)])
    x = np.array([1.0, 2.0])
    transformer.fit(x)
    features = transformer.transform(x)
    assert features.shape == (2, 2)
