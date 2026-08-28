import altair as alt
import pandas as pd
import pytest
from pandera.typing import DataFrame

from analysis import ModelComparisonReport
from core import (
    ConfidenceIntervalByModel,
    IntervalKind,
    IntervalMetricReportByModel,
    MetricReportByModel,
    ModelKind,
    PredictionIntervalByModel,
    PredictionsByModel,
    TrainingData,
)
from visualization import plot_confidence_intervals, plot_prediction_intervals


def _two_model_predictions() -> DataFrame[PredictionsByModel]:
    rows: list[dict[str, float | str]] = []
    for model in (ModelKind.RANDOM_FOREST, ModelKind.SVM):
        rows.extend(
            {
                PredictionsByModel.x: x,
                PredictionsByModel.y_pred: x * 0.5,
                PredictionsByModel.y_true: x,
                PredictionsByModel.mu_true: x * 0.4,
                PredictionsByModel.model: model.value,
            }
            for x in (-1.0, 0.0, 1.0)
        )
    return pd.DataFrame(rows).pipe(DataFrame[PredictionsByModel])


def _two_model_intervals(kind: IntervalKind) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for model in (ModelKind.RANDOM_FOREST, ModelKind.SVM):
        rows.extend(
            {
                "x": x,
                "lower": x - 0.5,
                "upper": x + 0.5,
                "kind": kind.value,
                "model": model.value,
            }
            for x in (-1.0, 0.0, 1.0)
        )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_report() -> ModelComparisonReport:
    confidence = _two_model_intervals(IntervalKind.CONFIDENCE).pipe(
        DataFrame[ConfidenceIntervalByModel],
    )
    prediction = _two_model_intervals(IntervalKind.PREDICTION).pipe(
        DataFrame[PredictionIntervalByModel],
    )
    empty_metrics = pd.DataFrame(
        {
            MetricReportByModel.metric: pd.Series(dtype=str),
            MetricReportByModel.lower: pd.Series(dtype=float),
            MetricReportByModel.upper: pd.Series(dtype=float),
            MetricReportByModel.model: pd.Series(dtype=str),
        },
    ).pipe(DataFrame[MetricReportByModel])
    empty_interval_metrics = pd.DataFrame(
        {
            IntervalMetricReportByModel.kind: pd.Series(dtype=str),
            IntervalMetricReportByModel.metric: pd.Series(dtype=str),
            IntervalMetricReportByModel.lower: pd.Series(dtype=float),
            IntervalMetricReportByModel.upper: pd.Series(dtype=float),
            IntervalMetricReportByModel.model: pd.Series(dtype=str),
        },
    ).pipe(DataFrame[IntervalMetricReportByModel])
    return ModelComparisonReport(
        predictions=_two_model_predictions(),
        confidence=confidence,
        prediction=prediction,
        regression_metrics=empty_metrics,
        confidence_metrics=empty_interval_metrics,
        prediction_metrics=empty_interval_metrics,
    )


def test_plot_confidence_intervals_hconcats_one_panel_per_model(
    synthetic_report: ModelComparisonReport,
) -> None:
    chart = plot_confidence_intervals(synthetic_report)
    assert isinstance(chart, alt.HConcatChart)
    assert len(chart.hconcat) == 2


def test_plot_confidence_intervals_each_panel_layers_mu_band_and_line(
    synthetic_report: ModelComparisonReport,
) -> None:
    chart = plot_confidence_intervals(synthetic_report)
    for panel in chart.hconcat:
        assert panel.layer is not None
        assert len(panel.layer) == 3
        mu_line, band, prediction_line = panel.layer
        assert mu_line.mark.type == "line"
        assert band.mark.type == "errorband"
        assert prediction_line.mark.type == "line"


def test_plot_prediction_intervals_hconcats_one_panel_per_model(
    dataset: DataFrame[TrainingData],
    synthetic_report: ModelComparisonReport,
) -> None:
    chart = plot_prediction_intervals(dataset, synthetic_report)
    assert isinstance(chart, alt.HConcatChart)
    assert len(chart.hconcat) == 2


def test_plot_prediction_intervals_each_panel_layers_scatter_band_and_line(
    dataset: DataFrame[TrainingData],
    synthetic_report: ModelComparisonReport,
) -> None:
    chart = plot_prediction_intervals(dataset, synthetic_report)
    for panel in chart.hconcat:
        assert panel.layer is not None
        assert len(panel.layer) == 3
        scatter, band, line = panel.layer
        assert scatter.mark.type == "circle"
        assert band.mark.type == "errorband"
        assert line.mark.type == "line"
