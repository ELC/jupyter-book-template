import pytest
from sklearn.ensemble import VotingRegressor
from sklearn.pipeline import Pipeline

from core import ModelKind, Settings, expand_features
from prediction import (
    ENSEMBLE_STEP,
    FEATURES_STEP,
    random_forest_regressor,
    svm_regressor,
)


@pytest.fixture
def two_model_regressors(settings: Settings) -> Pipeline:
    return Pipeline(
        steps=[
            (FEATURES_STEP, expand_features(settings)),
            (
                ENSEMBLE_STEP,
                VotingRegressor(
                    estimators=[
                        (ModelKind.RANDOM_FOREST.value, random_forest_regressor(settings)),
                        (ModelKind.SVM.value, svm_regressor(settings)),
                    ],
                ),
            ),
        ],
    )


@pytest.fixture
def deterministic_bootstrap_settings() -> Settings:
    return Settings(n_resamples=20, confidence_level=0.90, seed=0)


@pytest.fixture
def positive_width_bootstrap_settings() -> Settings:
    return Settings(n_resamples=20, confidence_level=0.90, seed=1)


@pytest.fixture
def varying_width_bootstrap_settings() -> Settings:
    return Settings(n_resamples=30, confidence_level=0.95, seed=0)


@pytest.fixture
def minimal_bootstrap_settings() -> Settings:
    return Settings(n_resamples=5)


@pytest.fixture
def small_bootstrap_settings(settings: Settings) -> Settings:
    return Settings(**{**settings.model_dump(), "n_resamples": 10})


@pytest.fixture
def empirical_coverage_bootstrap_settings(settings: Settings) -> Settings:
    return Settings(
        **{
            **settings.model_dump(),
            "n_resamples": 30,
            "confidence_level": 0.90,
            "seed": 0,
        },
    )
