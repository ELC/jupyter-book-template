import altair as alt
import pandas as pd
import pytest
from pandera.typing import DataFrame

from analysis.model_comparison import ModelComparisonReport
from core import (
    ConfidenceIntervalByModel,
    IntervalKind,
    IntervalMetricKind,
    IntervalMetricReportByModel,
    MetricReportByModel,
    ModelKind,
    PredictionIntervalByModel,
    PredictionsByModel,
    RegressionMetricKind,
)
from visualization import plot_interval_metrics, plot_regression_metrics


def _two_model_regression_metrics() -> DataFrame[MetricReportByModel]:
    rows = []
    for model in (ModelKind.RANDOM_FOREST, ModelKind.SVM):
        rows.extend({
                    MetricReportByModel.metric: metric.value,
                    MetricReportByModel.lower: 0.1,
                    MetricReportByModel.upper: 0.5,
                    MetricReportByModel.model: model.value,
                } for metric in RegressionMetricKind)
    return pd.DataFrame(rows).pipe(DataFrame[MetricReportByModel])


def _two_model_interval_metrics(kind: IntervalKind) -> DataFrame[IntervalMetricReportByModel]:
    rows = []
    metrics_for_kind = (
        (IntervalMetricKind.WIDTH,)
        if kind is IntervalKind.CONFIDENCE
        else (IntervalMetricKind.WIDTH, IntervalMetricKind.MWI)
    )
    for model in (ModelKind.RANDOM_FOREST, ModelKind.SVM):
        rows.extend({
                    IntervalMetricReportByModel.kind: kind.value,
                    IntervalMetricReportByModel.metric: metric.value,
                    IntervalMetricReportByModel.value: 1.5,
                    IntervalMetricReportByModel.model: model.value,
                } for metric in metrics_for_kind)
    return pd.DataFrame(rows).pipe(DataFrame[IntervalMetricReportByModel])


@pytest.fixture
def synthetic_metric_report() -> ModelComparisonReport:
    empty_predictions = pd.DataFrame(
        {
            PredictionsByModel.x: pd.Series(dtype=float),
            PredictionsByModel.y_pred: pd.Series(dtype=float),
            PredictionsByModel.y_true: pd.Series(dtype=float),
            PredictionsByModel.model: pd.Series(dtype=str),
        },
    ).pipe(DataFrame[PredictionsByModel])
    empty_ci = pd.DataFrame(
        {
            ConfidenceIntervalByModel.x: pd.Series(dtype=float),
            ConfidenceIntervalByModel.lower: pd.Series(dtype=float),
            ConfidenceIntervalByModel.upper: pd.Series(dtype=float),
            ConfidenceIntervalByModel.kind: pd.Series(dtype=str),
            ConfidenceIntervalByModel.model: pd.Series(dtype=str),
        },
    ).pipe(DataFrame[ConfidenceIntervalByModel])
    empty_pi = pd.DataFrame(
        {
            PredictionIntervalByModel.x: pd.Series(dtype=float),
            PredictionIntervalByModel.lower: pd.Series(dtype=float),
            PredictionIntervalByModel.upper: pd.Series(dtype=float),
            PredictionIntervalByModel.kind: pd.Series(dtype=str),
            PredictionIntervalByModel.model: pd.Series(dtype=str),
        },
    ).pipe(DataFrame[PredictionIntervalByModel])
    return ModelComparisonReport(
        predictions=empty_predictions,
        confidence=empty_ci,
        prediction=empty_pi,
        regression_metrics=_two_model_regression_metrics(),
        confidence_metrics=_two_model_interval_metrics(IntervalKind.CONFIDENCE),
        prediction_metrics=_two_model_interval_metrics(IntervalKind.PREDICTION),
    )


def test_plot_regression_metrics_horizontal_facets_by_metric(
    synthetic_metric_report: ModelComparisonReport,
) -> None:
    chart = plot_regression_metrics(synthetic_metric_report)
    assert isinstance(chart, alt.FacetChart)
    assert chart.facet.column.shorthand == f"{MetricReportByModel.metric}:N"
    layered = chart.spec
    assert layered.layer is not None
    assert len(layered.layer) == 3
    range_rule, lower_tick, upper_tick = layered.layer
    assert range_rule.mark.type == "rule"
    assert lower_tick.mark.type == "tick"
    assert upper_tick.mark.type == "tick"
    assert range_rule.encoding.y.shorthand == f"{MetricReportByModel.model}:N"
    assert range_rule.encoding.x.shorthand == f"{MetricReportByModel.lower}:Q"


def test_plot_interval_metrics_vconcats_kind_facets(
    synthetic_metric_report: ModelComparisonReport,
) -> None:
    chart = plot_interval_metrics(synthetic_metric_report)
    assert isinstance(chart, alt.VConcatChart)
    assert len(chart.vconcat) == 2
    confidence_chart, prediction_chart = chart.vconcat
    assert confidence_chart.title == "Confidence intervals"
    assert prediction_chart.title == "Prediction intervals"
    for facet_chart in chart.vconcat:
        assert isinstance(facet_chart, alt.FacetChart)
        assert facet_chart.facet.column.shorthand == f"{IntervalMetricReportByModel.metric}:N"
        spec = facet_chart.spec
        assert spec.mark.type == "tick"
        assert spec.mark.orient == "vertical"
        assert spec.encoding.y.shorthand == f"{IntervalMetricReportByModel.model}:N"
        assert spec.encoding.x.shorthand == f"{IntervalMetricReportByModel.value}:Q"
