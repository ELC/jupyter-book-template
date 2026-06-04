import altair as alt
import pandas as pd
import pytest
from pandera.typing import DataFrame

from analysis.model_comparison import ModelComparisonReport
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
from visualization import plot_intervals


def _two_model_predictions() -> DataFrame[PredictionsByModel]:
    rows: list[dict[str, float | str]] = []
    for model in (ModelKind.RANDOM_FOREST, ModelKind.SVM):
        rows.extend(
            {
                PredictionsByModel.x: x,
                PredictionsByModel.y_pred: x * 0.5,
                PredictionsByModel.y_true: x,
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
            IntervalMetricReportByModel.value: pd.Series(dtype=float),
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


def test_plot_intervals_hconcats_one_panel_per_model(
    dataset: DataFrame[TrainingData],
    synthetic_report: ModelComparisonReport,
) -> None:
    chart = plot_intervals(dataset, synthetic_report)
    assert isinstance(chart, alt.HConcatChart)
    assert len(chart.hconcat) == 2


def test_plot_intervals_each_panel_layers_four_marks(
    dataset: DataFrame[TrainingData],
    synthetic_report: ModelComparisonReport,
) -> None:
    chart = plot_intervals(dataset, synthetic_report)
    for panel in chart.hconcat:
        assert panel.layer is not None
        assert len(panel.layer) == 4
        scatter, prediction_band, confidence_band, line = panel.layer
        assert scatter.mark.type == "circle"
        assert prediction_band.mark.type == "errorband"
        assert confidence_band.mark.type == "errorband"
        assert line.mark.type == "line"
