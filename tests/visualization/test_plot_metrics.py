import altair as alt
from pandera.typing import DataFrame
from sklearn.pipeline import Pipeline

from analysis import bootstrap_confidence_intervals
from core import Settings
from core.schemas import SplitDatasetBase
from evaluation import interval_metrics, regression_metrics
from prediction import (
    conformal_intervals,
    fit_conformal,
    predict,
    random_forest_regressor,
    regression_pipeline,
)
from visualization import (
    CHART_HEIGHT,
    CHART_WIDTH,
    plot_interval_metrics,
    plot_regression_metrics,
)


def test_plot_regression_metrics_returns_errorbar_chart(
    split_dataset: DataFrame[SplitDatasetBase],
    fitted_pipeline: Pipeline,
) -> None:
    predictions = predict(fitted_pipeline, split_dataset)
    metrics = regression_metrics(predictions, settings=Settings(n_resamples=10, seed=0))
    chart = plot_regression_metrics(metrics)
    assert isinstance(chart, alt.Chart)
    assert chart.mark.type == "errorbar"
    assert chart.width == CHART_WIDTH
    assert chart.height == CHART_HEIGHT


def test_plot_interval_metrics_groups_shared_metrics_by_kind(
    split_dataset: DataFrame[SplitDatasetBase],
    settings: Settings,
    fitted_pipeline: Pipeline,
    unfitted_pipeline: Pipeline,
) -> None:
    predictions = predict(fitted_pipeline, split_dataset)
    confidence = bootstrap_confidence_intervals(
        unfitted_pipeline,
        split_dataset,
        Settings(n_resamples=5, confidence_level=0.90, seed=0),
    )
    conformal_pipeline = regression_pipeline(random_forest_regressor(settings), settings)
    conformal_model = fit_conformal(split_dataset, conformal_pipeline, settings)
    prediction = conformal_intervals(conformal_model, split_dataset)
    ci_report = interval_metrics(confidence, predictions, settings=settings)
    pi_report = interval_metrics(prediction, predictions, settings=settings)
    chart = plot_interval_metrics(ci_report, pi_report)
    assert isinstance(chart, alt.Chart)
    assert chart.mark.type == "bar"
    encoding = chart.encoding
    assert encoding.color is not None
    assert encoding.color.shorthand == "kind:N"
    assert encoding.xOffset is not None
    assert encoding.xOffset.shorthand == "kind:N"
