from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
from pandera.typing import DataFrame
from sklearn.pipeline import Pipeline

from analysis.cross_validation import confidence_intervals
from core.schemas import (
    ConfidenceIntervalByModel,
    IntervalKind,
    IntervalMetricKind,
    IntervalMetricReportByModel,
    MetricReportByModel,
    PredictionIntervalByModel,
    Predictions,
    PredictionsByModel,
    PredictionsWithGroundTruth,
    RegressionMetricKind,
    SplitDatasetBase,
)
from core.settings import Settings
from evaluation.interval_metrics import interval_metrics
from evaluation.regression_metrics import regression_metrics
from prediction.conformal import conformal_intervals, fit_conformal
from prediction.regression import ENSEMBLE_STEP, FEATURES_STEP, fit_pipeline, predict, regression_pipeline
from simulation.generator import mean_function

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    from sklearn.ensemble import VotingRegressor

MODEL_COLUMN = "model"
MU_TRUE_COLUMN = "mu_true"


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


def _sort_key(series: pd.Series, ranks: dict[str, dict[str, int]]) -> pd.Series:
    column = str(series.name)
    return series.map(ranks[column])


def _sorted_by(frame: pd.DataFrame, orders: dict[str, list[str]]) -> pd.DataFrame:
    ranks = {column: {value: index for index, value in enumerate(order)} for column, order in orders.items()}
    return frame.sort_values(
        by=list(orders),
        key=lambda series: _sort_key(series, ranks),
    ).reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class _PerModelFrames:
    predictions: pd.DataFrame
    confidence: pd.DataFrame
    prediction: pd.DataFrame
    regression_metrics: pd.DataFrame
    confidence_metrics: pd.DataFrame
    prediction_metrics: pd.DataFrame


def _score_model(
    name: str,
    pipeline: Pipeline,
    data: DataFrame[SplitDatasetBase],
    settings: Settings,
) -> _PerModelFrames:
    fitted = fit_pipeline(data, pipeline)
    preds = predict(fitted, data)
    ci = confidence_intervals(pipeline, data, settings)
    pi = conformal_intervals(fit_conformal(data, pipeline, settings), data)
    preds_with_mu = preds.assign(
        **{MU_TRUE_COLUMN: mean_function(preds[Predictions.x].to_numpy(), settings)},
    )
    return _PerModelFrames(
        predictions=_tag(preds, name),
        confidence=_tag(ci, name),
        prediction=_tag(pi, name),
        regression_metrics=_tag(regression_metrics(preds, settings=settings), name),
        confidence_metrics=_tag(
            interval_metrics(ci, preds_with_mu, settings=settings, target_column=MU_TRUE_COLUMN),
            name,
        ),
        prediction_metrics=_tag(
            interval_metrics(
                pi,
                preds_with_mu,
                settings=settings,
                target_column=PredictionsWithGroundTruth.y_true,
            ),
            name,
        ),
    )


def _per_model_pipelines(regressors: Pipeline) -> list[tuple[str, Pipeline]]:
    features: BaseEstimator = regressors.named_steps[FEATURES_STEP]
    ensemble: VotingRegressor = regressors.named_steps[ENSEMBLE_STEP]
    return [(name, regression_pipeline(estimator, features)) for name, estimator in ensemble.estimators]


def compare_models(
    data: DataFrame[SplitDatasetBase],
    regressors: Pipeline,
    settings: Settings,
) -> ModelComparisonReport:
    pairs = _per_model_pipelines(regressors)
    per_model = [_score_model(name, pipeline, data, settings) for name, pipeline in pairs]
    model_order = [name for name, _ in pairs]
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
        predictions=pd.concat([frames.predictions for frames in per_model], ignore_index=True).pipe(
            DataFrame[PredictionsByModel],
        ),
        confidence=pd.concat([frames.confidence for frames in per_model], ignore_index=True).pipe(
            DataFrame[ConfidenceIntervalByModel],
        ),
        prediction=pd.concat([frames.prediction for frames in per_model], ignore_index=True).pipe(
            DataFrame[PredictionIntervalByModel],
        ),
        regression_metrics=_sorted_by(
            pd.concat([frames.regression_metrics for frames in per_model], ignore_index=True),
            regression_orders,
        ).pipe(DataFrame[MetricReportByModel]),
        confidence_metrics=_sorted_by(
            pd.concat([frames.confidence_metrics for frames in per_model], ignore_index=True),
            interval_orders,
        ).pipe(DataFrame[IntervalMetricReportByModel]),
        prediction_metrics=_sorted_by(
            pd.concat([frames.prediction_metrics for frames in per_model], ignore_index=True),
            interval_orders,
        ).pipe(DataFrame[IntervalMetricReportByModel]),
    )
