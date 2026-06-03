import altair as alt
from pandera.typing import DataFrame

from core.schemas import (
    ConfidenceInterval,
    PredictionInterval,
    Predictions,
    PredictionsWithGroundTruth,
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


def _prediction_line(
    predictions: DataFrame[Predictions] | DataFrame[PredictionsWithGroundTruth],
) -> alt.Chart:
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
    intervals: DataFrame[ConfidenceInterval] | DataFrame[PredictionInterval],
) -> alt.Chart:
    return (
        alt
        .Chart(intervals)
        .mark_errorband(opacity=INTERVAL_BAND_OPACITY)
        .encode(
            x=f"{ConfidenceInterval.x}:Q",
            y=f"{ConfidenceInterval.lower}:Q",
            y2=f"{ConfidenceInterval.upper}:Q",
            color=alt.Color(
                f"{ConfidenceInterval.kind}:N",
                title="Interval",
                scale=alt.Scale(
                    domain=INTERVAL_COLOR_DOMAIN,
                    range=INTERVAL_COLOR_RANGE,
                ),
            ),
        )
    )


def plot_intervals(
    data: DataFrame[TrainingData],
    predictions: DataFrame[Predictions] | DataFrame[PredictionsWithGroundTruth],
    confidence: DataFrame[ConfidenceInterval],
    prediction: DataFrame[PredictionInterval],
) -> alt.FacetChart | alt.LayerChart:
    configure_altair()
    return alt.layer(
        _data_scatter(data),
        _interval_band(prediction),
        _interval_band(confidence),
        _prediction_line(predictions),
    ).properties(
        width=CHART_WIDTH,
        height=CHART_HEIGHT,
        title="Random forest predictions with bootstrap CI and conformal PI",
    )
