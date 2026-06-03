from mapie.regression import SplitConformalRegressor
from pandera.typing import DataFrame
from sklearn.ensemble import RandomForestRegressor

from core import IntervalKind, PredictionInterval, Settings, SplitKind, select_split
from core.schemas import SplitDatasetBase, TrainingData
from prediction import conformal_intervals, fit_conformal


def test_fit_conformal_returns_split_conformal_regressor(
    split_dataset: DataFrame[SplitDatasetBase],
    settings: Settings,
    unfitted_regressor: RandomForestRegressor,
) -> None:
    model = fit_conformal(split_dataset, unfitted_regressor, settings)
    assert isinstance(model, SplitConformalRegressor)


def test_fit_conformal_accepts_fitted_template(
    split_dataset: DataFrame[SplitDatasetBase],
    settings: Settings,
    fitted_model: RandomForestRegressor,
) -> None:
    model = fit_conformal(split_dataset, fitted_model, settings)
    assert isinstance(model, SplitConformalRegressor)


def test_conformal_intervals_aligns_with_evaluation_split(
    split_dataset: DataFrame[SplitDatasetBase],
    settings: Settings,
    unfitted_regressor: RandomForestRegressor,
) -> None:
    model = fit_conformal(split_dataset, unfitted_regressor, settings)
    intervals = conformal_intervals(model, split_dataset, settings)
    evaluation = select_split(split_dataset, SplitKind.EVALUATION)
    assert len(intervals) == len(evaluation)
    assert intervals[PredictionInterval.x].tolist() == evaluation[TrainingData.x].tolist()


def test_conformal_intervals_schema_and_width(
    split_dataset: DataFrame[SplitDatasetBase],
    settings: Settings,
    unfitted_regressor: RandomForestRegressor,
) -> None:
    model = fit_conformal(split_dataset, unfitted_regressor, settings)
    intervals = conformal_intervals(model, split_dataset, settings)
    widths = intervals[PredictionInterval.upper] - intervals[PredictionInterval.lower]
    assert (widths > 0).all()
    assert (intervals[PredictionInterval.kind] == IntervalKind.PREDICTION).all()
