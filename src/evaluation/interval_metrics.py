from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from mapie.metrics.regression import regression_mean_width_score, regression_mwi_score
from pandera.typing import DataFrame
from scipy.stats import bootstrap

from core import (
    ConfidenceInterval,
    IntervalBase,
    IntervalKind,
    IntervalMetricKind,
    IntervalMetricReport,
    IntervalWithGroundTruth,
    PredictionInterval,
    Predictions,
    PredictionsWithGroundTruth,
    Settings,
)

type IntervalMetricFunction = Callable[[np.ndarray, np.ndarray, Settings], float]


@dataclass(frozen=True, slots=True)
class IntervalMetric:
    name: IntervalMetricKind
    metric_function: IntervalMetricFunction

    def __call__(self, y_true: np.ndarray, y_intervals: np.ndarray, settings: Settings) -> float:
        return self.metric_function(y_true, y_intervals, settings)


def _width_score(_y_true: np.ndarray, y_intervals: np.ndarray, _settings: Settings) -> float:
    return float(regression_mean_width_score(y_intervals)[0])


def _mwi_score(y_true: np.ndarray, y_intervals: np.ndarray, settings: Settings) -> float:
    return regression_mwi_score(y_true, y_intervals, settings.confidence_level)


def _coverage_score(y_true: np.ndarray, y_intervals: np.ndarray, _settings: Settings) -> float:
    lower = y_intervals[:, 0, 0]
    upper = y_intervals[:, 1, 0]
    return float(((lower <= y_true) & (y_true <= upper)).mean())


CONFIDENCE_INTERVAL_METRICS: tuple[IntervalMetric, ...] = (
    IntervalMetric(IntervalMetricKind.WIDTH, _width_score),
    IntervalMetric(IntervalMetricKind.COVERAGE, _coverage_score),
)
PREDICTION_INTERVAL_METRICS: tuple[IntervalMetric, ...] = (
    IntervalMetric(IntervalMetricKind.WIDTH, _width_score),
    IntervalMetric(IntervalMetricKind.MWI, _mwi_score),
    IntervalMetric(IntervalMetricKind.COVERAGE, _coverage_score),
)
_INTERVAL_METRICS: dict[IntervalKind, tuple[IntervalMetric, ...]] = {
    IntervalKind.CONFIDENCE: CONFIDENCE_INTERVAL_METRICS,
    IntervalKind.PREDICTION: PREDICTION_INTERVAL_METRICS,
}


def _stack_intervals(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.stack([lower, upper], axis=1)[:, :, np.newaxis]


def _bootstrap_interval_metric_ci(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    metric: IntervalMetric,
    settings: Settings,
) -> tuple[float, float]:
    def statistic(y_true_sample: np.ndarray, lower_sample: np.ndarray, upper_sample: np.ndarray) -> float:
        return metric(y_true_sample, _stack_intervals(lower_sample, upper_sample), settings)

    rng = np.random.default_rng(settings.bootstrap_seed)
    result = bootstrap(
        (y_true, lower, upper),
        statistic=statistic,
        n_resamples=settings.n_resamples,
        confidence_level=settings.confidence_level,
        paired=True,
        method="bca",
        rng=rng,
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def _interval_metric_report(
    merged: DataFrame[IntervalWithGroundTruth],
    settings: Settings,
) -> DataFrame[IntervalMetricReport]:
    kind = IntervalKind(merged[IntervalWithGroundTruth.kind].iloc[0])
    y_true = merged[IntervalWithGroundTruth.y_true].to_numpy()
    lower = merged[IntervalBase.lower].to_numpy()
    upper = merged[IntervalBase.upper].to_numpy()
    rows: list[dict[str, object]] = []
    for metric in _INTERVAL_METRICS[kind]:
        ci_low, ci_high = _bootstrap_interval_metric_ci(y_true, lower, upper, metric, settings)
        rows.append(
            {
                IntervalMetricReport.kind: kind.value,
                IntervalMetricReport.metric: metric.name.value,
                IntervalMetricReport.lower: ci_low,
                IntervalMetricReport.upper: ci_high,
            },
        )
    return pd.DataFrame(rows).pipe(DataFrame[IntervalMetricReport])


def interval_metrics(
    intervals: DataFrame[ConfidenceInterval] | DataFrame[PredictionInterval],
    predictions: DataFrame[PredictionsWithGroundTruth],
    *,
    settings: Settings,
    target_column: str = PredictionsWithGroundTruth.y_true,
) -> DataFrame[IntervalMetricReport]:
    target_frame = predictions[[Predictions.x, target_column]].rename(
        columns={target_column: IntervalWithGroundTruth.y_true},
    )
    merged = intervals.merge(
        target_frame,
        left_on=IntervalBase.x,
        right_on=Predictions.x,
        how="inner",
    ).pipe(DataFrame[IntervalWithGroundTruth])
    return _interval_metric_report(merged, settings)
