from typing import NamedTuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pandera.typing import DataFrame
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from core import ConfidenceInterval, Settings, SplitDatasetBase, SplitKind, TrainingData, select_split

_RANDOM_STATE_SUFFIX = "random_state"


class BootstrapFit(NamedTuple):
    bootstrap_predictions: np.ndarray
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


def fit_and_predict(
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


_delayed_fit_and_predict = delayed(fit_and_predict)


def _spawn_seeds(parent_seed: int, count: int) -> list[int]:
    children = np.random.SeedSequence(parent_seed).spawn(count)
    return [int(child.generate_state(1)[0]) for child in children]


def case_resample(
    train: DataFrame[TrainingData],
    rng: np.random.Generator,
) -> DataFrame[TrainingData]:
    indices = rng.integers(0, len(train), size=len(train))
    resampled = train.iloc[indices].reset_index(drop=True)
    return resampled.pipe(DataFrame[TrainingData])


def _collect_bootstrap_predictions(
    pipeline: Pipeline,
    train: DataFrame[TrainingData],
    eval_data: DataFrame[TrainingData],
    settings: Settings,
) -> np.ndarray:
    resample_rng = np.random.default_rng(settings.bootstrap_seed)
    fit_seeds = _spawn_seeds(settings.bootstrap_seed, settings.n_resamples)
    resamples = [case_resample(train, resample_rng) for _ in range(settings.n_resamples)]
    tasks = (
        _delayed_fit_and_predict(pipeline, resample, eval_data, fit_seeds[index])
        for index, resample in enumerate(resamples)
    )
    parallel_executor = Parallel(n_jobs=settings.bootstrap_n_jobs)
    bootstrap_predictions = parallel_executor(tasks)
    return np.asarray(bootstrap_predictions)


def _run_bootstrap(
    pipeline: Pipeline,
    train: DataFrame[TrainingData],
    eval_data: DataFrame[TrainingData],
    settings: Settings,
) -> BootstrapFit:
    bootstrap_predictions = _collect_bootstrap_predictions(pipeline, train, eval_data, settings)
    theta_hat = fit_and_predict(pipeline, train, eval_data, seed=None)
    return BootstrapFit(bootstrap_predictions, theta_hat, eval_data)


def _basic_bootstrap_interval(
    fit: BootstrapFit,
    settings: Settings,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = 1.0 - settings.confidence_level
    low_q = np.quantile(fit.bootstrap_predictions, alpha / 2, axis=0)
    high_q = np.quantile(fit.bootstrap_predictions, 1.0 - alpha / 2, axis=0)
    return 2.0 * fit.theta_hat - high_q, 2.0 * fit.theta_hat - low_q


# Case (non-parametric) bootstrap confidence intervals on the evaluation grid.
#
# For each resample b = 1..n_resamples:
#   1. Draw a training set of size n_train with replacement (case / pairs
#      bootstrap, which is valid under i.i.d. rows without requiring residual
#      symmetry).
#   2. Refit a fresh clone of `pipeline` on that resample (with a
#      deterministically spawned random_state so the procedure stays
#      reproducible across calls).
#   3. Predict on the evaluation split.
#
# The resulting (n_resamples, n_eval) matrix of bootstrap predictions is
# reduced to a per-point basic-bootstrap pivot CI
#     [ 2 * theta_hat(x) - q_{1-a/2}(x), 2 * theta_hat(x) - q_{a/2}(x) ]
# where theta_hat(x) is the prediction from the full-training-set fit. The
# pivot is valid here (unlike the previous CV-fold variant) because the
# replicates are genuine bootstrap draws, satisfying the assumptions of
# Davison & Hinkley (1997, §5.2.1).
#
# Interpretation: this is a CI for the conditional mean of the *predictor*,
# E[ f_hat(x) | D ]. It coincides with a CI for the data-generating mean
# mu(x) only if the predictor class is well-specified.
def confidence_intervals(
    pipeline: Pipeline,
    data: DataFrame[SplitDatasetBase],
    settings: Settings,
) -> DataFrame[ConfidenceInterval]:
    train = select_split(data, SplitKind.TRAINING)
    eval_data = select_split(data, SplitKind.EVALUATION)
    fit = _run_bootstrap(pipeline, train, eval_data, settings)
    lower, upper = _basic_bootstrap_interval(fit, settings)
    return pd.DataFrame(
        {
            ConfidenceInterval.x: fit.eval_x,
            ConfidenceInterval.lower: lower,
            ConfidenceInterval.upper: upper,
        },
    ).pipe(DataFrame[ConfidenceInterval])
