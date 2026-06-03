from typing import NamedTuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pandera.typing import DataFrame
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from core.schemas import ConfidenceInterval, SplitDatasetBase, SplitKind, TrainingData
from core.settings import Settings
from core.splits import select_split


class BootstrapFitResult(NamedTuple):
    bootstrap_array: np.ndarray
    theta_hat: np.ndarray
    eval_data: DataFrame[TrainingData]

    @property
    def eval_x(self) -> np.ndarray:
        return self.eval_data[TrainingData.x].to_numpy()


def _fit_and_predict(
    pipeline: Pipeline,
    train: DataFrame[TrainingData],
    eval_data: DataFrame[TrainingData],
) -> np.ndarray:
    train_x = train[TrainingData.x].to_numpy()
    train_y = train[TrainingData.y].to_numpy()
    eval_x = eval_data[TrainingData.x].to_numpy()
    fitted_pipeline = clone(pipeline)
    fitted_pipeline.fit(train_x, train_y)
    return fitted_pipeline.predict(eval_x)


_delayed_fit_and_predict = delayed(_fit_and_predict)


def _collect_bootstrap_predictions(
    pipeline: Pipeline,
    train: DataFrame[TrainingData],
    eval_data: DataFrame[TrainingData],
    settings: Settings,
) -> np.ndarray:
    rng = np.random.default_rng(settings.bootstrap_seed)
    resample_indices = rng.integers(0, len(train), size=(settings.n_resamples, len(train)))
    bootstrap_tasks = (
        _delayed_fit_and_predict(
            pipeline,
            train.iloc[resample_idx],
            eval_data,
        )
        for resample_idx in resample_indices
    )
    parallel_executor = Parallel(n_jobs=settings.bootstrap_n_jobs, return_as="generator")
    parallel_resamples = parallel_executor(bootstrap_tasks)
    boot_preds = list(tqdm(parallel_resamples, total=settings.n_resamples, desc="Bootstrap"))
    return np.asarray(boot_preds)


def _run_bootstrap_resamples(
    pipeline: Pipeline,
    train: DataFrame[TrainingData],
    eval_data: DataFrame[TrainingData],
    settings: Settings,
) -> BootstrapFitResult:
    bootstrap_array = _collect_bootstrap_predictions(
        pipeline,
        train,
        eval_data,
        settings,
    )
    theta_hat = _fit_and_predict(pipeline, train, eval_data)
    return BootstrapFitResult(bootstrap_array, theta_hat, eval_data)


def _basic_confidence_interval(
    fit: BootstrapFitResult,
    settings: Settings,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = 1.0 - settings.confidence_level
    low_q = np.quantile(fit.bootstrap_array, alpha / 2, axis=0)
    high_q = np.quantile(fit.bootstrap_array, 1.0 - alpha / 2, axis=0)
    return 2.0 * fit.theta_hat - high_q, 2.0 * fit.theta_hat - low_q


def bootstrap_confidence_intervals(
    pipeline: Pipeline,
    data: DataFrame[SplitDatasetBase],
    settings: Settings,
) -> DataFrame[ConfidenceInterval]:
    train = select_split(data, SplitKind.TRAINING)
    eval_data = select_split(data, SplitKind.EVALUATION)
    fit = _run_bootstrap_resamples(pipeline, train, eval_data, settings)
    lower, upper = _basic_confidence_interval(fit, settings)
    return pd.DataFrame(
        {
            ConfidenceInterval.x: fit.eval_x,
            ConfidenceInterval.lower: lower,
            ConfidenceInterval.upper: upper,
        },
    ).pipe(DataFrame[ConfidenceInterval])
