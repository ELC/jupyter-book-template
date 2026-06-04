from typing import NamedTuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pandera.typing import DataFrame
from sklearn.base import clone
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline

from core.schemas import ConfidenceInterval, SplitDatasetBase, SplitKind, TrainingData
from core.settings import Settings
from core.splits import select_split

_RANDOM_STATE_SUFFIX = "random_state"


class CrossValidationFitResult(NamedTuple):
    fold_predictions: np.ndarray
    theta_hat: np.ndarray
    eval_data: DataFrame[TrainingData]

    @property
    def eval_x(self) -> np.ndarray:
        return self.eval_data[TrainingData.x].to_numpy()


def _seed_overrides(pipeline: Pipeline, seed: int) -> dict[str, int]:
    return {
        param: seed
        for param in pipeline.get_params(deep=True)
        if param == _RANDOM_STATE_SUFFIX or param.endswith(f"__{_RANDOM_STATE_SUFFIX}")
    }


def _clone_with_seed(pipeline: Pipeline, seed: int | None) -> Pipeline:
    fresh = clone(pipeline)
    if seed is None:
        return fresh
    overrides = _seed_overrides(fresh, seed)
    if overrides:
        fresh.set_params(**overrides)
    return fresh


def _fit_and_predict(
    pipeline: Pipeline,
    train: DataFrame[TrainingData],
    eval_data: DataFrame[TrainingData],
    seed: int | None,
) -> np.ndarray:
    train_x = train[[TrainingData.x]].to_numpy()
    train_y = train[TrainingData.y].to_numpy()
    eval_x = eval_data[[TrainingData.x]].to_numpy()
    fitted_pipeline = _clone_with_seed(pipeline, seed)
    fitted_pipeline.fit(train_x, train_y)
    return fitted_pipeline.predict(eval_x)


_delayed_fit_and_predict = delayed(_fit_and_predict)


def _spawn_seeds(parent_seed: int, count: int) -> list[int]:
    children = np.random.SeedSequence(parent_seed).spawn(count)
    return [int(child.generate_state(1)[0]) for child in children]


def _collect_fold_predictions(
    pipeline: Pipeline,
    train: DataFrame[TrainingData],
    eval_data: DataFrame[TrainingData],
    settings: Settings,
) -> np.ndarray:
    splitter = RepeatedKFold(
        n_splits=settings.cv_n_splits,
        n_repeats=settings.cv_n_repeats,
        random_state=settings.bootstrap_seed,
    )
    n_folds = settings.cv_n_splits * settings.cv_n_repeats
    fold_seeds = _spawn_seeds(settings.bootstrap_seed, n_folds)
    fold_indices = list(splitter.split(train))
    tasks = (
        _delayed_fit_and_predict(
            pipeline,
            train.iloc[train_idx],
            eval_data,
            fold_seeds[fold_position],
        )
        for fold_position, (train_idx, _) in enumerate(fold_indices)
    )
    parallel_executor = Parallel(n_jobs=settings.bootstrap_n_jobs)
    fold_preds = parallel_executor(tasks)
    return np.asarray(fold_preds)


def _run_cross_validation(
    pipeline: Pipeline,
    train: DataFrame[TrainingData],
    eval_data: DataFrame[TrainingData],
    settings: Settings,
) -> CrossValidationFitResult:
    fold_predictions = _collect_fold_predictions(pipeline, train, eval_data, settings)
    theta_hat = _fit_and_predict(pipeline, train, eval_data, seed=None)
    return CrossValidationFitResult(fold_predictions, theta_hat, eval_data)


def _basic_confidence_interval(
    fit: CrossValidationFitResult,
    settings: Settings,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = 1.0 - settings.confidence_level
    low_q = np.quantile(fit.fold_predictions, alpha / 2, axis=0)
    high_q = np.quantile(fit.fold_predictions, 1.0 - alpha / 2, axis=0)
    return 2.0 * fit.theta_hat - high_q, 2.0 * fit.theta_hat - low_q


def confidence_intervals(
    pipeline: Pipeline,
    data: DataFrame[SplitDatasetBase],
    settings: Settings,
) -> DataFrame[ConfidenceInterval]:
    train = select_split(data, SplitKind.TRAINING)
    eval_data = select_split(data, SplitKind.EVALUATION)
    fit = _run_cross_validation(pipeline, train, eval_data, settings)
    lower, upper = _basic_confidence_interval(fit, settings)
    return pd.DataFrame(
        {
            ConfidenceInterval.x: fit.eval_x,
            ConfidenceInterval.lower: lower,
            ConfidenceInterval.upper: upper,
        },
    ).pipe(DataFrame[ConfidenceInterval])
