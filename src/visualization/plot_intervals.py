from typing import cast

import altair as alt
from pandera.typing import DataFrame

from analysis import ModelComparisonReport
from core import (
    ConfidenceIntervalByModel,
    IntervalKind,
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
    INTERVAL_COLORS,
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


def _mu_true_line(predictions: DataFrame[PredictionsByModel]) -> alt.Chart:
    unique = predictions.drop_duplicates(subset=[Predictions.x]).sort_values(Predictions.x)
    return (
        alt
        .Chart(unique)
        .mark_line(color="gray", strokeDash=[6, 4], strokeWidth=2)
        .encode(
            x=f"{Predictions.x}:Q",
            y=alt.Y(f"{PredictionsByModel.mu_true}:Q", title="y"),
            tooltip=[f"{Predictions.x}:Q", f"{PredictionsByModel.mu_true}:Q"],
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
    kind: IntervalKind,
) -> alt.Chart:
    return (
        alt
        .Chart(intervals)
        .mark_errorband(opacity=INTERVAL_BAND_OPACITY, color=INTERVAL_COLORS[kind])
        .encode(
            x=f"{ConfidenceIntervalByModel.x}:Q",
            y=f"{ConfidenceIntervalByModel.lower}:Q",
            y2=f"{ConfidenceIntervalByModel.upper}:Q",
        )
    )


def _confidence_panel(
    model: str,
    report: ModelComparisonReport,
) -> alt.LayerChart:
    model_predictions = report.predictions[report.predictions[PredictionsByModel.model] == model]
    model_confidence = report.confidence[report.confidence[ConfidenceIntervalByModel.model] == model]
    layered = alt.layer(
        _mu_true_line(model_predictions),
        _interval_band(model_confidence, IntervalKind.CONFIDENCE),
        _prediction_line(model_predictions),
    )
    return cast("alt.LayerChart", layered).properties(
        width=CHART_WIDTH,
        height=CHART_HEIGHT,
        title=alt.TitleParams(
            text=model,
            subtitle="Evaluation split: mu(x) dashed, f_hat(x) solid, bootstrap CI shaded",
        ),
    )


def _prediction_panel(
    model: str,
    data: DataFrame[TrainingData],
    report: ModelComparisonReport,
) -> alt.LayerChart:
    model_predictions = report.predictions[report.predictions[PredictionsByModel.model] == model]
    model_prediction = report.prediction[report.prediction[PredictionIntervalByModel.model] == model]
    layered = alt.layer(
        _data_scatter(data),
        _interval_band(model_prediction, IntervalKind.PREDICTION),
        _prediction_line(model_predictions),
    )
    return cast("alt.LayerChart", layered).properties(
        width=CHART_WIDTH,
        height=CHART_HEIGHT,
        title=alt.TitleParams(
            text=model,
            subtitle="All observations behind evaluation predictions and conformal PI",
        ),
    )


def plot_confidence_intervals(report: ModelComparisonReport) -> alt.HConcatChart:
    """Bootstrap CI for the predictor's conditional mean against the true mean function.

    Each panel layers: the data-generating mean mu(x) (dashed grey), the model's
    prediction mean f_hat(x), and the bootstrap confidence band. This visualises
    *epistemic* (predictor) uncertainty only.

    Returns:
        Horizontal concatenation with one confidence panel per model.
    """
    configure_altair()
    models = list(report.predictions[PredictionsByModel.model].drop_duplicates())
    panels = [_confidence_panel(model, report) for model in models]
    return alt.hconcat(*panels).properties(
        title="Predictor uncertainty: bootstrap CI vs true mean mu(x)",
    )


def plot_prediction_intervals(
    data: DataFrame[TrainingData],
    report: ModelComparisonReport,
) -> alt.HConcatChart:
    """Conformal prediction intervals against observed y.

    Each panel layers: the data scatter, the model's prediction mean, and the
    split-conformal prediction band. This visualises the *full predictive*
    (epistemic + aleatoric) uncertainty.

    Returns:
        Horizontal concatenation with one prediction panel per model.
    """
    configure_altair()
    models = list(report.predictions[PredictionsByModel.model].drop_duplicates())
    panels = [_prediction_panel(model, data, report) for model in models]
    return alt.hconcat(*panels).properties(
        title="Predictive uncertainty: conformal PI vs observed y",
    )
