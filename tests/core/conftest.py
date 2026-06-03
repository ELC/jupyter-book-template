from typing import cast

import numpy as np
import pytest
from pandera.typing import DataFrame

from core import Settings, SplitKind, TrainingData


@pytest.fixture(
    params=[
        pytest.param(
            Settings(polynomial_degree=0, fourier_terms=0, seasonality_frequency=0.5),
            id="raw-x-only",
        ),
        pytest.param(
            Settings(polynomial_degree=2, fourier_terms=0, seasonality_frequency=0.5),
            id="polynomial-only",
        ),
        pytest.param(
            Settings(polynomial_degree=0, fourier_terms=2, seasonality_frequency=0.5),
            id="fourier-only",
        ),
        pytest.param(
            Settings(polynomial_degree=3, fourier_terms=2, seasonality_frequency=0.5),
            id="polynomial-and-fourier",
        ),
    ],
)
def expand_features_settings(request: pytest.FixtureRequest) -> Settings:
    return cast("Settings", request.param)


@pytest.fixture
def expected_feature_columns(expand_features_settings: Settings) -> int:
    columns = 0
    if expand_features_settings.polynomial_degree > 0:
        columns += expand_features_settings.polynomial_degree
    if expand_features_settings.fourier_terms > 0:
        columns += 2 * expand_features_settings.fourier_terms
    if columns == 0:
        return 1
    return columns


@pytest.fixture
def expand_features_column_count_x() -> np.ndarray:
    return np.array([-1.0, 0.0, 2.0])


@pytest.fixture
def polynomial_expand_features_settings() -> Settings:
    return Settings(polynomial_degree=2, fourier_terms=0)


@pytest.fixture
def polynomial_expand_features_x() -> np.ndarray:
    return np.array([2.0, 3.0])


@pytest.fixture
def fourier_expand_features_settings() -> Settings:
    return Settings(
        polynomial_degree=0,
        fourier_terms=1,
        seasonality_frequency=0.5,
    )


@pytest.fixture
def fourier_expand_features_x() -> np.ndarray:
    return np.array([0.0, np.pi])


@pytest.fixture
def polynomial_and_fourier_expand_features_settings() -> Settings:
    return Settings(polynomial_degree=3, fourier_terms=2, seasonality_frequency=0.5)


@pytest.fixture
def expected_fold(
    selected_split_kind: SplitKind,
    train_fold: DataFrame[TrainingData],
    calibration_fold: DataFrame[TrainingData],
    evaluation_fold: DataFrame[TrainingData],
) -> DataFrame[TrainingData]:
    folds = {
        SplitKind.TRAINING: train_fold,
        SplitKind.CALIBRATION: calibration_fold,
        SplitKind.EVALUATION: evaluation_fold,
    }
    return folds[selected_split_kind]
