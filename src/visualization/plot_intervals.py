import altair as alt
import pandas as pd
from pandera.typing import DataFrame

from core.schemas import (
    ConfidenceInterval,
    IntervalKind,
    PredictionInterval,
    Predictions,
    PredictionsWithGroundTruth,
    TrainingData,
)
from visualization._common import configure_altair


def _data_scatter(data: pd.DataFrame) -> alt.Chart:
    return (
        alt
        .Chart(data)
        .mark_circle(opacity=0.35, size=30)
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


def _interval_bands(intervals: pd.DataFrame) -> alt.Chart:
    return (
        alt
        .Chart(intervals)
        .mark_errorband(opacity=0.35)
        .encode(
            x=f"{ConfidenceInterval.x}:Q",
            y=f"{ConfidenceInterval.lower}:Q",
            y2=f"{ConfidenceInterval.upper}:Q",
            color=alt.Color(
                "kind:N",
                title="Interval",
                scale=alt.Scale(
                    domain=[member.value for member in IntervalKind],
                    range=["#1f77b4", "#ff7f0e"],
                ),
            ),
        )
    )


def plot_intervals(
    data: pd.DataFrame,
    predictions: DataFrame[Predictions] | DataFrame[PredictionsWithGroundTruth],
    confidence: DataFrame[ConfidenceInterval],
    prediction: DataFrame[PredictionInterval],
) -> alt.FacetChart | alt.LayerChart:
    configure_altair()
    intervals = pd.concat([confidence, prediction], ignore_index=True)
    return alt.layer(
        _data_scatter(data),
        _interval_bands(intervals),
        _prediction_line(predictions),
    ).properties(
        width=640,
        height=400,
        title="Random forest predictions with bootstrap CI and conformal PI",
    )
