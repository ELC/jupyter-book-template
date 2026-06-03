import altair as alt
from pandera.typing import DataFrame
from sklearn.ensemble import RandomForestRegressor

from analysis import bootstrap_confidence_intervals
from core import Settings
from core.schemas import SplitDatasetBase
from prediction import conformal_intervals, fit_conformal, predict, random_forest_regressor
from visualization import plot_intervals


def test_plot_intervals_returns_layer_chart(
    settings: Settings,
    split_dataset: DataFrame[SplitDatasetBase],
    fitted_model: RandomForestRegressor,
    unfitted_regressor: RandomForestRegressor,
) -> None:
    predictions = predict(fitted_model, split_dataset, settings)
    confidence = bootstrap_confidence_intervals(
        unfitted_regressor,
        split_dataset,
        Settings(n_resamples=10, confidence_level=0.90, rng=0),
    )
    conformal_regressor = random_forest_regressor(settings)
    conformal_model = fit_conformal(split_dataset, conformal_regressor, settings)
    prediction = conformal_intervals(conformal_model, split_dataset, settings)
    chart = plot_intervals(split_dataset, predictions, confidence, prediction)
    assert isinstance(chart, alt.LayerChart)
    assert chart.layer is not None
    assert len(chart.layer) == 3
