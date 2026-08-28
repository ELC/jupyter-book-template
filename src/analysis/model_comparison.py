from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np
import pandas as pd
from mapie.regression import SplitConformalRegressor
from pandera.typing import DataFrame
from scipy.stats import norm
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from analysis import confidence_intervals
from core import (
    ConfidenceIntervalByModel,
    IntervalKind,
    IntervalMetricKind,
    IntervalMetricReport,
    IntervalMetricReportByModel,
    MetricReportByModel,
    PredictionIntervalByModel,
    Predictions,
    PredictionsByModel,
    PredictionsWithGroundTruth,
    RegressionMetricKind,
    Settings,
    SplitDatasetBase,
    SplitKind,
    prepare_split,
)
from evaluation import PREDICTION_INTERVAL_METRICS, interval_metrics, regression_metrics
from prediction import (
    ENSEMBLE_STEP,
    FEATURES_STEP,
    build_conformity_score,
    conformal_intervals,
    fit_conformal,
    fit_pipeline,
    predict,
    regression_pipeline,
)
from simulation import mean_function

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    from sklearn.ensemble import VotingRegressor

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


def _attach_mu_true(
    preds: DataFrame[PredictionsWithGroundTruth],
    settings: Settings,
) -> pd.DataFrame:
    mu_true = mean_function(preds[Predictions.x].to_numpy(), settings)
    return preds.assign(**{PredictionsByModel.mu_true: mu_true})


def _sort_key(series: pd.Series, ranks: dict[str, dict[str, int]]) -> pd.Series:
    column = str(series.name)
    return series.map(ranks[column])


def _sorted_by(frame: pd.DataFrame, orders: dict[str, list[str]]) -> pd.DataFrame:
    ranks = {column: {value: index for index, value in enumerate(order)} for column, order in orders.items()}
    return frame.sort_values(
        by=list(orders),
        key=lambda series: _sort_key(series, ranks),
    ).reset_index(drop=True)


def _conformalize_and_intervals(
    estimator: Pipeline,
    pipeline: Pipeline,
    arrays: "_ConformalArrays",
    settings: Settings,
) -> tuple[np.ndarray, np.ndarray]:
    conformal = SplitConformalRegressor(
        estimator=estimator,
        confidence_level=settings.confidence_level,
        conformity_score=build_conformity_score(pipeline, settings),
        prefit=True,
    )
    conformal.conformalize(arrays.cal_x, arrays.cal_y)
    _, intervals = conformal.predict_interval(arrays.eval_x)
    lower, upper = intervals[..., 0].T
    return lower, upper


def _prediction_interval_metric_values(
    lower: np.ndarray,
    upper: np.ndarray,
    y_true: np.ndarray,
    settings: Settings,
) -> dict[IntervalMetricKind, float]:
    y_intervals = np.stack([lower, upper], axis=1)[:, :, np.newaxis]
    return {metric.name: metric(y_true, y_intervals, settings) for metric in PREDICTION_INTERVAL_METRICS}


class _ConformalArrays(NamedTuple):
    cal_x: np.ndarray
    cal_y: np.ndarray
    eval_x: np.ndarray
    eval_y: np.ndarray


def _jackknife_metric_replicates(
    fitted_estimator: Pipeline,
    pipeline: Pipeline,
    arrays: _ConformalArrays,
    settings: Settings,
) -> dict[IntervalMetricKind, np.ndarray]:
    n_cal = len(arrays.cal_y)
    samples: dict[IntervalMetricKind, list[float]] = {metric.name: [] for metric in PREDICTION_INTERVAL_METRICS}
    for left_out in range(n_cal):
        keep = np.ones(n_cal, dtype=bool)
        keep[left_out] = False
        loo_arrays = _ConformalArrays(
            cal_x=arrays.cal_x[keep],
            cal_y=arrays.cal_y[keep],
            eval_x=arrays.eval_x,
            eval_y=arrays.eval_y,
        )
        lower_loo, upper_loo = _conformalize_and_intervals(
            fitted_estimator,
            pipeline,
            loo_arrays,
            settings,
        )
        values = _prediction_interval_metric_values(lower_loo, upper_loo, arrays.eval_y, settings)
        for key, value in values.items():
            samples[key].append(value)
    return {key: np.asarray(values) for key, values in samples.items()}


def _jackknife_confidence_interval(
    theta_hat: float,
    replicates: np.ndarray,
    settings: Settings,
) -> tuple[float, float]:
    n_replicates = len(replicates)
    mean_loo = float(replicates.mean())
    variance = (n_replicates - 1.0) / n_replicates * float(np.sum((replicates - mean_loo) ** 2))
    standard_error = float(np.sqrt(variance))
    alpha = 1.0 - settings.confidence_level
    z_quantile = float(norm.ppf(1.0 - alpha / 2.0))
    return theta_hat - z_quantile * standard_error, theta_hat + z_quantile * standard_error


# Jackknife-on-calibration CI for split-conformal interval metrics.
#
# The previous bootstrap variant resampled the calibration set with
# replacement, which mechanically inflates the empirical quantile of
# nonconformity scores (duplicates of small residuals appear multiple times)
# and, more importantly, breaks the exchangeability requirement of split
# conformal coverage on every replicate (Vovk, Gammerman & Shafer 2005;
# Lei et al. 2018).
#
# Leave-one-out replicates over the calibration set instead preserve
# exchangeability of the remaining n_cal - 1 points, so each LOO interval is
# itself a valid split-conformal interval at the same nominal level. The
# metric variability across LOO replicates is a clean estimate of the
# conformal-calibration contribution to the metric's sampling distribution,
# reduced to a Wald CI via the standard jackknife SE (Efron & Tibshirani
# 1993, §11.2):
#     SE_jack^2 = (n - 1) / n * sum_i (theta_(i) - mean(theta_(.)))^2
# centred on the full-calibration estimate theta_hat.
#
# Note: variability over the evaluation set is not separately accounted for;
# the CI captures conformal-threshold uncertainty given a fixed evaluation
# grid, not test-population sampling variability.
def _jackknife_prediction_interval_metrics(
    pipeline: Pipeline,
    data: DataFrame[SplitDatasetBase],
    settings: Settings,
) -> DataFrame[IntervalMetricReport]:
    fitted_estimator = fit_pipeline(data, clone(pipeline))
    calib = prepare_split(data, SplitKind.CALIBRATION)
    eval_split = prepare_split(data, SplitKind.EVALUATION)
    arrays = _ConformalArrays(
        cal_x=calib.x,
        cal_y=calib.y.to_numpy(),
        eval_x=eval_split.x,
        eval_y=eval_split.y.to_numpy(),
    )

    original_lower, original_upper = _conformalize_and_intervals(
        fitted_estimator,
        pipeline,
        arrays,
        settings,
    )
    theta_hat = _prediction_interval_metric_values(
        original_lower,
        original_upper,
        arrays.eval_y,
        settings,
    )
    replicates = _jackknife_metric_replicates(fitted_estimator, pipeline, arrays, settings)

    rows = []
    for metric in PREDICTION_INTERVAL_METRICS:
        ci_low, ci_high = _jackknife_confidence_interval(
            theta_hat[metric.name],
            replicates[metric.name],
            settings,
        )
        rows.append(
            {
                IntervalMetricReport.kind: IntervalKind.PREDICTION.value,
                IntervalMetricReport.metric: metric.name.value,
                IntervalMetricReport.lower: ci_low,
                IntervalMetricReport.upper: ci_high,
            },
        )
    return pd.DataFrame(rows).pipe(DataFrame[IntervalMetricReport])


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
    preds_with_mu = _attach_mu_true(preds, settings)
    ci = confidence_intervals(pipeline, data, settings)
    pi = conformal_intervals(fit_conformal(data, pipeline, settings), data)
    return _PerModelFrames(
        predictions=_tag(preds_with_mu, name),
        confidence=_tag(ci, name),
        prediction=_tag(pi, name),
        regression_metrics=_tag(regression_metrics(preds, settings=settings), name),
        confidence_metrics=_tag(
            interval_metrics(
                ci,
                cast("DataFrame[PredictionsWithGroundTruth]", preds_with_mu),
                settings=settings,
                target_column=PredictionsByModel.mu_true,
            ),
            name,
        ),
        prediction_metrics=_tag(
            _jackknife_prediction_interval_metrics(pipeline, data, settings),
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
