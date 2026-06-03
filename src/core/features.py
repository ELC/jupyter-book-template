import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from core.settings import Settings


def _fourier_features(x: np.ndarray, settings: Settings) -> np.ndarray:
    harmonics = np.arange(1, settings.fourier_terms + 1, dtype=float)
    angles = np.outer(x, harmonics * settings.seasonality_frequency)
    return np.hstack((np.sin(angles), np.cos(angles)))


def expand_features(x: np.ndarray, settings: Settings) -> np.ndarray:
    x_column = x.reshape(-1, 1)
    blocks: list[np.ndarray] = []
    if settings.polynomial_degree > 0:
        polynomial = PolynomialFeatures(
            degree=settings.polynomial_degree,
            include_bias=False,
        )
        blocks.append(polynomial.fit_transform(x_column))
    if settings.fourier_terms > 0:
        blocks.append(_fourier_features(x, settings))
    if not blocks:
        return x_column
    return np.hstack(blocks)
