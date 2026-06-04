from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from mapie.metrics.regression import regression_mean_width_score, regression_mwi_score
from pandera.typing import DataFrame

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


CONFIDENCE_INTERVAL_METRICS: tuple[IntervalMetric, ...] = (IntervalMetric(IntervalMetricKind.WIDTH, _width_score),)
PREDICTION_INTERVAL_METRICS: tuple[IntervalMetric, ...] = (IntervalMetric(IntervalMetricKind.MWI, _mwi_score),)
_INTERVAL_METRICS: dict[IntervalKind, tuple[IntervalMetric, ...]] = {
    IntervalKind.CONFIDENCE: CONFIDENCE_INTERVAL_METRICS,
    IntervalKind.PREDICTION: CONFIDENCE_INTERVAL_METRICS + PREDICTION_INTERVAL_METRICS,
}


def _mapie_interval_array(merged: DataFrame[IntervalWithGroundTruth]) -> np.ndarray:
    lower = merged[IntervalBase.lower].to_numpy()
    upper = merged[IntervalBase.upper].to_numpy()
    return np.stack([lower, upper], axis=1)[:, :, np.newaxis]


def _interval_metric_report(
    merged: DataFrame[IntervalWithGroundTruth],
    settings: Settings,
) -> DataFrame[IntervalMetricReport]:
    kind = IntervalKind(merged[IntervalWithGroundTruth.kind].iloc[0])
    y_true = merged[IntervalWithGroundTruth.y_true].to_numpy()
    y_intervals = _mapie_interval_array(merged)
    rows = [
        {
            IntervalMetricReport.kind: kind.value,
            IntervalMetricReport.metric: metric.name.value,
            IntervalMetricReport.value: metric(y_true, y_intervals, settings),
        }
        for metric in _INTERVAL_METRICS[kind]
    ]
    return pd.DataFrame(rows).pipe(DataFrame[IntervalMetricReport])


def interval_metrics(
    intervals: DataFrame[ConfidenceInterval] | DataFrame[PredictionInterval],
    predictions: DataFrame[PredictionsWithGroundTruth],
    *,
    settings: Settings,
) -> DataFrame[IntervalMetricReport]:
    merged = intervals.merge(
        predictions[[Predictions.x, PredictionsWithGroundTruth.y_true]],
        left_on=IntervalBase.x,
        right_on=Predictions.x,
        how="inner",
    ).pipe(DataFrame[IntervalWithGroundTruth])
    return _interval_metric_report(merged, settings)
