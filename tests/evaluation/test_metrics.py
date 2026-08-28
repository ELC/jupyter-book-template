from pandera.typing import DataFrame

from core import (
    ConfidenceInterval,
    IntervalKind,
    IntervalMetricKind,
    IntervalMetricReport,
    MetricReport,
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
    assert set(metrics[MetricReport.metric].tolist()) == {member.value for member in RegressionMetricKind}
    assert (metrics[MetricReport.lower] <= metrics[MetricReport.upper]).all()
    rmse_row = metrics.loc[metrics[MetricReport.metric] == RegressionMetricKind.RMSE.value].iloc[0]
    assert rmse_row[MetricReport.lower] >= 0.0


def test_regression_metrics_accepts_custom_metric_list(
    predictions: DataFrame[PredictionsWithGroundTruth],
    custom_regression_metrics_settings: Settings,
) -> None:
    metrics = regression_metrics(
        predictions,
        metrics=[RegressionMetric(RegressionMetricKind.RMSE, lambda _y_true, _y_pred: 1.0)],
        settings=custom_regression_metrics_settings,
    )
    assert metrics[MetricReport.metric].tolist() == [RegressionMetricKind.RMSE.value]
    assert metrics[MetricReport.lower].iloc[0] == 1.0
    assert metrics[MetricReport.upper].iloc[0] == 1.0


def test_default_regression_metrics_are_regression_metric_instances() -> None:
    for metric in DEFAULT_REGRESSION_METRICS:
        assert isinstance(metric, RegressionMetric)


def test_interval_metrics_for_confidence_intervals_reports_width_and_coverage(
    predictions: DataFrame[PredictionsWithGroundTruth],
    cv_confidence: DataFrame[ConfidenceInterval],
    settings: Settings,
) -> None:
    report = interval_metrics(cv_confidence, predictions, settings=settings)
    assert set(report[IntervalMetricReport.metric].tolist()) == {
        IntervalMetricKind.WIDTH.value,
        IntervalMetricKind.COVERAGE.value,
    }
    assert report[IntervalMetricReport.kind].eq(IntervalKind.CONFIDENCE.value).all()
    assert (report[IntervalMetricReport.lower] <= report[IntervalMetricReport.upper]).all()
    width_row = report.loc[report[IntervalMetricReport.metric] == IntervalMetricKind.WIDTH.value].iloc[0]
    coverage_row = report.loc[report[IntervalMetricReport.metric] == IntervalMetricKind.COVERAGE.value].iloc[0]
    assert width_row[IntervalMetricReport.lower] >= 0.0
    assert 0.0 <= coverage_row[IntervalMetricReport.lower] <= 1.0
    assert 0.0 <= coverage_row[IntervalMetricReport.upper] <= 1.0


def test_interval_metrics_reports_mapie_metrics(
    predictions: DataFrame[PredictionsWithGroundTruth],
    prediction_intervals: DataFrame[PredictionInterval],
    settings: Settings,
) -> None:
    report = interval_metrics(prediction_intervals, predictions, settings=settings)
    assert set(report[IntervalMetricReport.metric].tolist()) == {
        IntervalMetricKind.WIDTH.value,
        IntervalMetricKind.MWI.value,
        IntervalMetricKind.COVERAGE.value,
    }
    assert report[IntervalMetricReport.kind].eq(IntervalKind.PREDICTION.value).all()
    assert (report[IntervalMetricReport.lower] <= report[IntervalMetricReport.upper]).all()
    width_row = report.loc[report[IntervalMetricReport.metric] == IntervalMetricKind.WIDTH.value].iloc[0]
    mwi_row = report.loc[report[IntervalMetricReport.metric] == IntervalMetricKind.MWI.value].iloc[0]
    coverage_row = report.loc[report[IntervalMetricReport.metric] == IntervalMetricKind.COVERAGE.value].iloc[0]
    assert width_row[IntervalMetricReport.lower] >= 0.0
    assert mwi_row[IntervalMetricReport.lower] >= 0.0
    assert 0.0 <= coverage_row[IntervalMetricReport.lower] <= 1.0
    assert 0.0 <= coverage_row[IntervalMetricReport.upper] <= 1.0
