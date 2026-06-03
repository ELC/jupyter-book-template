import pytest

from core import ModelKind, Settings
from prediction import (
    MultiRegressor,
    random_forest_regressor,
    regression_pipeline,
    svm_regressor,
)


@pytest.fixture
def two_model_regressors(settings: Settings) -> MultiRegressor:
    return MultiRegressor(
        estimators=[
            (
                ModelKind.RANDOM_FOREST.value,
                regression_pipeline(random_forest_regressor(settings), settings),
            ),
            (
                ModelKind.SVM.value,
                regression_pipeline(svm_regressor(settings), settings),
            ),
        ],
    )


@pytest.fixture
def deterministic_bootstrap_settings() -> Settings:
    return Settings(n_resamples=20, confidence_level=0.90, seed=0)


@pytest.fixture
def positive_width_bootstrap_settings() -> Settings:
    return Settings(n_resamples=30, confidence_level=0.90, seed=1)


@pytest.fixture
def varying_width_bootstrap_settings() -> Settings:
    return Settings(n_resamples=50, confidence_level=0.95, seed=0)


@pytest.fixture
def minimal_resample_bootstrap_settings() -> Settings:
    return Settings(n_resamples=5)


@pytest.fixture
def ten_resample_bootstrap_settings(settings: Settings) -> Settings:
    return Settings(**{**settings.model_dump(), "n_resamples": 10})


@pytest.fixture
def empirical_coverage_bootstrap_settings(settings: Settings) -> Settings:
    return Settings(**{**settings.model_dump(), "n_resamples": 80, "confidence_level": 0.90, "seed": 0})
