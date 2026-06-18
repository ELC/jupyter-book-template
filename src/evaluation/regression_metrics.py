from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandera.typing import DataFrame
from scipy.stats import bootstrap
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from core import (
    MetricReport,
    Predictions,
    PredictionsWithGroundTruth,
    RegressionMetricKind,
    Settings,
)

type RegressionMetricFunction = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True, slots=True)
class RegressionMetric:
    name: RegressionMetricKind
    metric_function: RegressionMetricFunction

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.metric_function(y_true, y_pred)


DEFAULT_REGRESSION_METRICS = (
    RegressionMetric(RegressionMetricKind.RMSE, root_mean_squared_error),
    RegressionMetric(RegressionMetricKind.MAE, mean_absolute_error),
    RegressionMetric(RegressionMetricKind.R2, r2_score),
)


def _bootstrap_metric_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Sequence[RegressionMetric],
    settings: Settings,
) -> DataFrame[MetricReport]:
    rng = np.random.default_rng(settings.bootstrap_seed)
    rows: list[dict[str, object]] = []
    for metric in metrics:
        result = bootstrap(
            (y_true, y_pred),
            statistic=metric,
            n_resamples=settings.n_resamples,
            confidence_level=settings.confidence_level,
            paired=True,
            method="bca",
            rng=rng,
        )
        lower, upper = result.confidence_interval
        rows.append(
            {
                MetricReport.metric: metric.name.value,
                MetricReport.lower: float(lower),
                MetricReport.upper: float(upper),
            },
        )
    return pd.DataFrame(rows).pipe(DataFrame[MetricReport])


def regression_metrics(
    predictions: DataFrame[PredictionsWithGroundTruth],
    metrics: Sequence[RegressionMetric] | None = None,
    *,
    settings: Settings,
) -> DataFrame[MetricReport]:
    selected_metrics = metrics or DEFAULT_REGRESSION_METRICS
    y_true = predictions[PredictionsWithGroundTruth.y_true].to_numpy()
    y_pred = predictions[Predictions.y_pred].to_numpy()
    return _bootstrap_metric_intervals(y_true, y_pred, selected_metrics, settings)
