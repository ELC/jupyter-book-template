from pandera.typing import DataFrame
from sklearn.pipeline import Pipeline

from analysis import compare_models
from core import (
    ConfidenceIntervalByModel,
    IntervalMetricReportByModel,
    MetricReportByModel,
    ModelKind,
    PredictionIntervalByModel,
    PredictionsByModel,
    Settings,
)
from core.schemas import SplitDatasetBase


def test_compare_models_predictions_validate_per_model(
    split_dataset: DataFrame[SplitDatasetBase],
    two_model_regressors: Pipeline,
    settings: Settings,
) -> None:
    report = compare_models(split_dataset, two_model_regressors, settings)
    PredictionsByModel.validate(report.predictions)
    assert set(report.predictions[PredictionsByModel.model].unique()) == {
        ModelKind.RANDOM_FOREST.value,
        ModelKind.SVM.value,
    }


def test_compare_models_intervals_validate_per_model(
    split_dataset: DataFrame[SplitDatasetBase],
    two_model_regressors: Pipeline,
    settings: Settings,
) -> None:
    report = compare_models(split_dataset, two_model_regressors, settings)
    ConfidenceIntervalByModel.validate(report.confidence)
    PredictionIntervalByModel.validate(report.prediction)
    assert set(report.confidence[ConfidenceIntervalByModel.model].unique()) == {
        ModelKind.RANDOM_FOREST.value,
        ModelKind.SVM.value,
    }
    assert set(report.prediction[PredictionIntervalByModel.model].unique()) == {
        ModelKind.RANDOM_FOREST.value,
        ModelKind.SVM.value,
    }


def test_compare_models_metrics_validate_per_model(
    split_dataset: DataFrame[SplitDatasetBase],
    two_model_regressors: Pipeline,
    settings: Settings,
) -> None:
    report = compare_models(split_dataset, two_model_regressors, settings)
    MetricReportByModel.validate(report.regression_metrics)
    IntervalMetricReportByModel.validate(report.confidence_metrics)
    IntervalMetricReportByModel.validate(report.prediction_metrics)
    assert set(report.regression_metrics[MetricReportByModel.model].unique()) == {
        ModelKind.RANDOM_FOREST.value,
        ModelKind.SVM.value,
    }
