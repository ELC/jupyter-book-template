import altair as alt
from pandera.typing import DataFrame
from sklearn.ensemble import RandomForestRegressor

from analysis import bootstrap_confidence_intervals
from core import Settings, TrainingData
from core.schemas import SplitDatasetBase
from prediction import conformal_intervals, fit_conformal, predict, random_forest_regressor
from visualization import CHART_HEIGHT, CHART_WIDTH, plot_intervals
from visualization.theme import SCATTER_OPACITY


def test_plot_intervals_returns_layer_chart(
    dataset: DataFrame[TrainingData],
    settings: Settings,
    split_dataset: DataFrame[SplitDatasetBase],
    fitted_model: RandomForestRegressor,
    unfitted_regressor: RandomForestRegressor,
) -> None:
    predictions = predict(fitted_model, split_dataset, settings)
    confidence = bootstrap_confidence_intervals(
        unfitted_regressor,
        split_dataset,
        Settings(n_resamples=10, confidence_level=0.90, seed=0),
    )
    conformal_regressor = random_forest_regressor(settings)
    conformal_model = fit_conformal(split_dataset, conformal_regressor, settings)
    prediction = conformal_intervals(conformal_model, split_dataset, settings)
    chart = plot_intervals(dataset, predictions, confidence, prediction)
    assert isinstance(chart, alt.LayerChart)
    assert chart.layer is not None
    assert len(chart.layer) == 4
    assert chart.width == CHART_WIDTH
    assert chart.height == CHART_HEIGHT


def test_plot_intervals_layer_encodings(
    dataset: DataFrame[TrainingData],
    settings: Settings,
    split_dataset: DataFrame[SplitDatasetBase],
    fitted_model: RandomForestRegressor,
    unfitted_regressor: RandomForestRegressor,
) -> None:
    predictions = predict(fitted_model, split_dataset, settings)
    confidence = bootstrap_confidence_intervals(
        unfitted_regressor,
        split_dataset,
        Settings(n_resamples=5, confidence_level=0.90, seed=0),
    )
    conformal_model = fit_conformal(split_dataset, random_forest_regressor(settings), settings)
    prediction = conformal_intervals(conformal_model, split_dataset, settings)
    chart = plot_intervals(dataset, predictions, confidence, prediction)
    assert chart.layer is not None
    scatter, prediction_band, confidence_band, line = chart.layer
    assert scatter.mark.type == "circle"
    assert scatter.mark.opacity == SCATTER_OPACITY
    assert prediction_band.mark.type == "errorband"
    assert confidence_band.mark.type == "errorband"
    assert line.mark.type == "line"
