import pytest
from mapie.regression import SplitConformalRegressor
from pandera.typing import DataFrame
from sklearn.ensemble import RandomForestRegressor

from analysis import bootstrap_confidence_intervals
from core import ConfidenceInterval, PredictionInterval, PredictionsWithGroundTruth, Settings
from core.schemas import SplitDatasetBase
from prediction import conformal_intervals, fit_conformal, predict


@pytest.fixture
def predictions(
    fitted_model: RandomForestRegressor,
    split_dataset: DataFrame[SplitDatasetBase],
    settings: Settings,
) -> DataFrame[PredictionsWithGroundTruth]:
    return predict(fitted_model, split_dataset, settings)


@pytest.fixture
def regression_metrics_settings() -> Settings:
    return Settings(n_resamples=20, seed=0)


@pytest.fixture
def custom_regression_metrics_settings() -> Settings:
    return Settings(n_resamples=5, seed=0)


@pytest.fixture
def bootstrap_confidence(
    unfitted_regressor: RandomForestRegressor,
    split_dataset: DataFrame[SplitDatasetBase],
    settings: Settings,
) -> DataFrame[ConfidenceInterval]:
    return bootstrap_confidence_intervals(unfitted_regressor, split_dataset, settings)


@pytest.fixture
def conformal_model(
    split_dataset: DataFrame[SplitDatasetBase],
    settings: Settings,
    unfitted_regressor: RandomForestRegressor,
) -> SplitConformalRegressor:
    return fit_conformal(split_dataset, unfitted_regressor, settings)


@pytest.fixture
def prediction_intervals(
    conformal_model: SplitConformalRegressor,
    split_dataset: DataFrame[SplitDatasetBase],
    settings: Settings,
) -> DataFrame[PredictionInterval]:
    return conformal_intervals(conformal_model, split_dataset, settings)
