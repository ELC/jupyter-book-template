import pytest
from mapie.regression import SplitConformalRegressor
from pandera.typing import DataFrame
from sklearn.pipeline import Pipeline

from analysis import confidence_intervals
from core import (
    ConfidenceInterval,
    PredictionInterval,
    PredictionsWithGroundTruth,
    Settings,
    SplitDatasetBase,
)
from prediction import conformal_intervals, fit_conformal, predict


@pytest.fixture
def predictions(
    fitted_pipeline: Pipeline,
    split_dataset: DataFrame[SplitDatasetBase],
) -> DataFrame[PredictionsWithGroundTruth]:
    return predict(fitted_pipeline, split_dataset)


@pytest.fixture
def regression_metrics_settings() -> Settings:
    return Settings(n_resamples=20, seed=0)


@pytest.fixture
def custom_regression_metrics_settings() -> Settings:
    return Settings(n_resamples=5, seed=0)


@pytest.fixture
def cv_confidence(
    unfitted_pipeline: Pipeline,
    split_dataset: DataFrame[SplitDatasetBase],
    settings: Settings,
) -> DataFrame[ConfidenceInterval]:
    return confidence_intervals(unfitted_pipeline, split_dataset, settings)


@pytest.fixture
def conformal_model(
    split_dataset: DataFrame[SplitDatasetBase],
    settings: Settings,
    unfitted_pipeline: Pipeline,
) -> SplitConformalRegressor:
    return fit_conformal(split_dataset, unfitted_pipeline, settings)


@pytest.fixture
def prediction_intervals(
    conformal_model: SplitConformalRegressor,
    split_dataset: DataFrame[SplitDatasetBase],
) -> DataFrame[PredictionInterval]:
    return conformal_intervals(conformal_model, split_dataset)
