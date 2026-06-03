from dataclasses import dataclass

import pandas as pd
from pandera.typing import DataFrame

from analysis.bootstrap import bootstrap_confidence_intervals
from core.schemas import (
    ConfidenceIntervalByModel,
    IntervalKind,
    IntervalMetricKind,
    IntervalMetricReportByModel,
    MetricReportByModel,
    PredictionIntervalByModel,
    PredictionsByModel,
    RegressionMetricKind,
    SplitDatasetBase,
)
from core.settings import Settings
from evaluation.interval_metrics import interval_metrics
from evaluation.regression_metrics import regression_metrics
from prediction.conformal import conformal_intervals, fit_conformal
from prediction.multi_regressor import MultiRegressor
from prediction.regression import fit_pipeline, predict

MODEL_COLUMN = "model"


@dataclass(frozen=True, slots=True)
class ModelComparisonReport:
    predictions: DataFrame[PredictionsByModel]
    confidence: DataFrame[ConfidenceIntervalByModel]
    prediction: DataFrame[PredictionIntervalByModel]
    regression_metrics: DataFrame[MetricReportByModel]
    confidence_metrics: DataFrame[IntervalMetricReportByModel]
    prediction_metrics: DataFrame[IntervalMetricReportByModel]


def _tag(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    return frame.assign(**{MODEL_COLUMN: name})


def _sorted_by(frame: pd.DataFrame, orders: dict[str, list[str]]) -> pd.DataFrame:
    ranks = {column: {value: index for index, value in enumerate(order)} for column, order in orders.items()}
    return frame.sort_values(
        by=list(orders),
        key=lambda series: series.map(ranks[series.name]),
    ).reset_index(drop=True)


def compare_models(
    data: DataFrame[SplitDatasetBase],
    regressors: MultiRegressor,
    settings: Settings,
) -> ModelComparisonReport:
    predictions_parts: list[pd.DataFrame] = []
    confidence_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    regression_metrics_parts: list[pd.DataFrame] = []
    confidence_metrics_parts: list[pd.DataFrame] = []
    prediction_metrics_parts: list[pd.DataFrame] = []

    for name, pipeline in regressors.estimators:
        fitted = fit_pipeline(data, pipeline)
        preds = predict(fitted, data)
        ci = bootstrap_confidence_intervals(pipeline, data, settings)
        pi = conformal_intervals(fit_conformal(data, pipeline, settings), data)
        predictions_parts.append(_tag(preds, name))
        confidence_parts.append(_tag(ci, name))
        prediction_parts.append(_tag(pi, name))
        regression_metrics_parts.append(_tag(regression_metrics(preds, settings=settings), name))
        confidence_metrics_parts.append(_tag(interval_metrics(ci, preds, settings=settings), name))
        prediction_metrics_parts.append(_tag(interval_metrics(pi, preds, settings=settings), name))

    model_order = [name for name, _ in regressors.estimators]
    regression_orders = {
        MetricReportByModel.metric: [m.value for m in RegressionMetricKind],
        MODEL_COLUMN: model_order,
    }
    interval_orders = {
        IntervalMetricReportByModel.kind: [m.value for m in IntervalKind],
        IntervalMetricReportByModel.metric: [m.value for m in IntervalMetricKind],
        MODEL_COLUMN: model_order,
    }

    return ModelComparisonReport(
        predictions=pd.concat(predictions_parts, ignore_index=True).pipe(
            DataFrame[PredictionsByModel],
        ),
        confidence=pd.concat(confidence_parts, ignore_index=True).pipe(
            DataFrame[ConfidenceIntervalByModel],
        ),
        prediction=pd.concat(prediction_parts, ignore_index=True).pipe(
            DataFrame[PredictionIntervalByModel],
        ),
        regression_metrics=_sorted_by(
            pd.concat(regression_metrics_parts, ignore_index=True),
            regression_orders,
        ).pipe(DataFrame[MetricReportByModel]),
        confidence_metrics=_sorted_by(
            pd.concat(confidence_metrics_parts, ignore_index=True),
            interval_orders,
        ).pipe(DataFrame[IntervalMetricReportByModel]),
        prediction_metrics=_sorted_by(
            pd.concat(prediction_metrics_parts, ignore_index=True),
            interval_orders,
        ).pipe(DataFrame[IntervalMetricReportByModel]),
    )
