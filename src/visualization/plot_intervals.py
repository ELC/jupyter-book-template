import altair as alt
from pandera.typing import DataFrame

from analysis.model_comparison import ModelComparisonReport
from core.schemas import (
    ConfidenceIntervalByModel,
    PredictionIntervalByModel,
    Predictions,
    PredictionsByModel,
    TrainingData,
)
from visualization._common import configure_altair
from visualization.theme import (
    CHART_HEIGHT,
    CHART_WIDTH,
    INTERVAL_BAND_OPACITY,
    INTERVAL_COLOR_DOMAIN,
    INTERVAL_COLOR_RANGE,
    SCATTER_OPACITY,
)


def _data_scatter(data: DataFrame[TrainingData]) -> alt.Chart:
    return (
        alt
        .Chart(data)
        .mark_circle(opacity=SCATTER_OPACITY, size=30)
        .encode(
            x=alt.X(f"{TrainingData.x}:Q", title="x"),
            y=alt.Y(f"{TrainingData.y}:Q", title="y"),
            tooltip=[f"{TrainingData.x}:Q", f"{TrainingData.y}:Q"],
        )
    )


def _prediction_line(predictions: DataFrame[PredictionsByModel]) -> alt.Chart:
    return (
        alt
        .Chart(predictions.sort_values(Predictions.x))
        .mark_line(color="black", strokeWidth=2)
        .encode(
            x=f"{Predictions.x}:Q",
            y=f"{Predictions.y_pred}:Q",
            tooltip=[f"{Predictions.x}:Q", f"{Predictions.y_pred}:Q"],
        )
    )


def _interval_band(
    intervals: DataFrame[ConfidenceIntervalByModel] | DataFrame[PredictionIntervalByModel],
) -> alt.Chart:
    return (
        alt
        .Chart(intervals)
        .mark_errorband(opacity=INTERVAL_BAND_OPACITY)
        .encode(
            x=f"{ConfidenceIntervalByModel.x}:Q",
            y=f"{ConfidenceIntervalByModel.lower}:Q",
            y2=f"{ConfidenceIntervalByModel.upper}:Q",
            color=alt.Color(
                f"{ConfidenceIntervalByModel.kind}:N",
                title="Interval",
                scale=alt.Scale(
                    domain=INTERVAL_COLOR_DOMAIN,
                    range=INTERVAL_COLOR_RANGE,
                ),
            ),
        )
    )


def _model_panel(
    model: str,
    data: DataFrame[TrainingData],
    report: ModelComparisonReport,
) -> alt.LayerChart | alt.FacetChart:
    model_predictions = report.predictions[report.predictions[PredictionsByModel.model] == model]
    model_confidence = report.confidence[report.confidence[ConfidenceIntervalByModel.model] == model]
    model_prediction = report.prediction[report.prediction[PredictionIntervalByModel.model] == model]
    return alt.layer(
        _data_scatter(data),
        _interval_band(model_prediction),
        _interval_band(model_confidence),
        _prediction_line(model_predictions),
    ).properties(
        width=CHART_WIDTH,
        height=CHART_HEIGHT,
        title=model,
    )


def plot_intervals(
    data: DataFrame[TrainingData],
    report: ModelComparisonReport,
) -> alt.HConcatChart:
    configure_altair()
    models = list(report.predictions[PredictionsByModel.model].drop_duplicates())
    panels = [_model_panel(model, data, report) for model in models]
    return alt.hconcat(*panels).properties(
        title="Predictions with bootstrap CI and conformal PI",
    )
