import numpy as np
from mapie.metrics.regression import regression_coverage_score
from mapie.regression import SplitConformalRegressor
from pandera.typing import DataFrame
from sklearn.pipeline import Pipeline

from core import IntervalKind, IntervalMetricKind, PredictionInterval, Settings, SplitKind, select_split
from core.schemas import SplitDatasetBase, TrainingData
from evaluation import interval_metrics
from prediction import conformal_intervals, fit_conformal, predict


def test_fit_conformal_returns_split_conformal_regressor(
    split_dataset: DataFrame[SplitDatasetBase],
    settings: Settings,
    unfitted_pipeline: Pipeline,
) -> None:
    model = fit_conformal(split_dataset, unfitted_pipeline, settings)
    assert isinstance(model, SplitConformalRegressor)


def test_fit_conformal_accepts_fitted_template(
    split_dataset: DataFrame[SplitDatasetBase],
    settings: Settings,
    fitted_pipeline: Pipeline,
) -> None:
    model = fit_conformal(split_dataset, fitted_pipeline, settings)
    assert isinstance(model, SplitConformalRegressor)


def test_conformal_intervals_aligns_with_evaluation_split(
    split_dataset: DataFrame[SplitDatasetBase],
    settings: Settings,
    unfitted_pipeline: Pipeline,
) -> None:
    model = fit_conformal(split_dataset, unfitted_pipeline, settings)
    intervals = conformal_intervals(model, split_dataset)
    evaluation = select_split(split_dataset, SplitKind.EVALUATION)
    assert len(intervals) == len(evaluation)
    assert intervals[PredictionInterval.x].tolist() == evaluation[TrainingData.x].tolist()


def test_conformal_intervals_schema_and_width(
    split_dataset: DataFrame[SplitDatasetBase],
    settings: Settings,
    unfitted_pipeline: Pipeline,
) -> None:
    model = fit_conformal(split_dataset, unfitted_pipeline, settings)
    intervals = conformal_intervals(model, split_dataset)
    widths = intervals[PredictionInterval.upper] - intervals[PredictionInterval.lower]
    assert (widths > 0).all()
    assert (intervals[PredictionInterval.kind] == IntervalKind.PREDICTION).all()


def test_conformal_empirical_coverage_near_confidence_level(
    split_dataset: DataFrame[SplitDatasetBase],
    settings: Settings,
    unfitted_pipeline: Pipeline,
    fitted_pipeline: Pipeline,
) -> None:
    model = fit_conformal(split_dataset, unfitted_pipeline, settings)
    intervals = conformal_intervals(model, split_dataset)
    predictions = predict(fitted_pipeline, split_dataset)
    y_true = predictions["y_true"].to_numpy()
    y_intervals = np.stack(
        [intervals["lower"].to_numpy(), intervals["upper"].to_numpy()],
        axis=1,
    )[:, :, np.newaxis]
    coverage = float(regression_coverage_score(y_true, y_intervals)[0])
    assert 0.70 <= coverage <= 1.0
    report = interval_metrics(intervals, predictions, settings=settings)
    assert set(report["metric"].tolist()) == {
        IntervalMetricKind.WIDTH.value,
        IntervalMetricKind.MWI.value,
    }
