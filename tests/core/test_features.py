import numpy as np

from core import Settings, expand_features


def test_expand_features_column_count(
    expand_features_settings: Settings,
    expected_feature_columns: int,
    expand_features_column_count_x: np.ndarray,
) -> None:
    features = expand_features(expand_features_column_count_x, expand_features_settings)
    assert features.shape == (len(expand_features_column_count_x), expected_feature_columns)


def test_expand_features_polynomial_values(
    polynomial_expand_features_settings: Settings,
    polynomial_expand_features_x: np.ndarray,
) -> None:
    features = expand_features(polynomial_expand_features_x, polynomial_expand_features_settings)
    np.testing.assert_allclose(features[:, 0], polynomial_expand_features_x)
    np.testing.assert_allclose(features[:, 1], polynomial_expand_features_x**2)


def test_expand_features_polynomial_and_fourier_values(
    polynomial_and_fourier_expand_features_settings: Settings,
    expand_features_column_count_x: np.ndarray,
) -> None:
    settings = polynomial_and_fourier_expand_features_settings
    features = expand_features(expand_features_column_count_x, settings)
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
    features = expand_features(fourier_expand_features_x, fourier_expand_features_settings)
    frequency = fourier_expand_features_settings.seasonality_frequency
    np.testing.assert_allclose(features[:, 0], np.sin(frequency * fourier_expand_features_x))
    np.testing.assert_allclose(features[:, 1], np.cos(frequency * fourier_expand_features_x))
