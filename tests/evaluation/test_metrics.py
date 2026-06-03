from pandera.typing import DataFrame

from core import (
    ConfidenceInterval,
    IntervalKind,
    IntervalMetricKind,
    PredictionInterval,
    PredictionsWithGroundTruth,
    RegressionMetricKind,
    Settings,
)
from evaluation import (
    DEFAULT_REGRESSION_METRICS,
    RegressionMetric,
    interval_metrics,
    regression_metrics,
)


def test_regression_metrics_returns_bootstrap_intervals(
    predictions: DataFrame[PredictionsWithGroundTruth],
    regression_metrics_settings: Settings,
) -> None:
    metrics = regression_metrics(predictions, settings=regression_metrics_settings)
    assert set(metrics["metric"].tolist()) == {member.value for member in RegressionMetricKind}
    assert (metrics["lower"] <= metrics["upper"]).all()
    rmse_row = metrics.loc[metrics["metric"] == RegressionMetricKind.RMSE.value].iloc[0]
    assert rmse_row["lower"] >= 0.0


def test_regression_metrics_accepts_custom_metric_list(
    predictions: DataFrame[PredictionsWithGroundTruth],
    custom_regression_metrics_settings: Settings,
) -> None:
    metrics = regression_metrics(
        predictions,
        metrics=[RegressionMetric(RegressionMetricKind.RMSE, lambda _y_true, _y_pred: 1.0)],
        settings=custom_regression_metrics_settings,
    )
    assert metrics["metric"].tolist() == [RegressionMetricKind.RMSE.value]
    assert metrics["lower"].iloc[0] == 1.0
    assert metrics["upper"].iloc[0] == 1.0


def test_default_regression_metrics_are_regression_metric_instances() -> None:
    for metric in DEFAULT_REGRESSION_METRICS:
        assert isinstance(metric, RegressionMetric)


def test_interval_metrics_for_confidence_intervals_reports_width_only(
    predictions: DataFrame[PredictionsWithGroundTruth],
    bootstrap_confidence: DataFrame[ConfidenceInterval],
    settings: Settings,
) -> None:
    report = interval_metrics(bootstrap_confidence, predictions, settings=settings)
    assert report["metric"].tolist() == [IntervalMetricKind.WIDTH.value]
    assert report["kind"].iloc[0] == IntervalKind.CONFIDENCE.value
    assert report["value"].iloc[0] > 0.0


def test_interval_metrics_reports_mapie_metrics(
    predictions: DataFrame[PredictionsWithGroundTruth],
    prediction_intervals: DataFrame[PredictionInterval],
    settings: Settings,
) -> None:
    report = interval_metrics(prediction_intervals, predictions, settings=settings)
    assert set(report["metric"].tolist()) == {
        IntervalMetricKind.WIDTH.value,
        IntervalMetricKind.MWI.value,
    }
    assert report["kind"].eq(IntervalKind.PREDICTION.value).all()
    assert (report["value"] > 0).all()
