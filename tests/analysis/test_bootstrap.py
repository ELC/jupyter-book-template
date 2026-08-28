import numpy as np
from pandera.typing import DataFrame
from sklearn.pipeline import Pipeline

from analysis import case_resample, confidence_intervals, fit_and_predict
from core import (
    ConfidenceInterval,
    IntervalKind,
    Settings,
    SplitDatasetBase,
    SplitKind,
    TrainingData,
    select_split,
)
from simulation import mean_function


def test_confidence_intervals_is_deterministic(
    split_dataset: DataFrame[SplitDatasetBase],
    unfitted_pipeline: Pipeline,
    deterministic_bootstrap_settings: Settings,
) -> None:
    first = confidence_intervals(unfitted_pipeline, split_dataset, deterministic_bootstrap_settings)
    second = confidence_intervals(unfitted_pipeline, split_dataset, deterministic_bootstrap_settings)
    assert first.equals(second)


def test_confidence_intervals_interval_width_positive(
    split_dataset: DataFrame[SplitDatasetBase],
    unfitted_pipeline: Pipeline,
    positive_width_bootstrap_settings: Settings,
) -> None:
    intervals = confidence_intervals(unfitted_pipeline, split_dataset, positive_width_bootstrap_settings)
    widths = intervals[ConfidenceInterval.upper] - intervals[ConfidenceInterval.lower]
    assert (widths > 0).all()
    assert (intervals[ConfidenceInterval.kind] == IntervalKind.CONFIDENCE).all()


def test_confidence_intervals_uses_settings_defaults(
    split_dataset: DataFrame[SplitDatasetBase],
    unfitted_pipeline: Pipeline,
    small_bootstrap_settings: Settings,
) -> None:
    intervals = confidence_intervals(unfitted_pipeline, split_dataset, small_bootstrap_settings)
    assert (intervals[ConfidenceInterval.upper] > intervals[ConfidenceInterval.lower]).all()


def test_confidence_intervals_width_varies_with_x(
    split_dataset: DataFrame[SplitDatasetBase],
    unfitted_pipeline: Pipeline,
    varying_width_bootstrap_settings: Settings,
) -> None:
    intervals = confidence_intervals(unfitted_pipeline, split_dataset, varying_width_bootstrap_settings)
    widths = intervals[ConfidenceInterval.upper] - intervals[ConfidenceInterval.lower]
    assert np.unique(np.round(widths, 4)).size > 1


def test_confidence_intervals_accepts_fitted_template(
    split_dataset: DataFrame[SplitDatasetBase],
    fitted_pipeline: Pipeline,
    minimal_bootstrap_settings: Settings,
) -> None:
    intervals = confidence_intervals(fitted_pipeline, split_dataset, minimal_bootstrap_settings)
    assert len(intervals) > 0


def test_confidence_intervals_empirical_coverage_against_mu(
    split_dataset: DataFrame[SplitDatasetBase],
    unfitted_pipeline: Pipeline,
    empirical_coverage_bootstrap_settings: Settings,
) -> None:
    intervals = confidence_intervals(
        unfitted_pipeline,
        split_dataset,
        empirical_coverage_bootstrap_settings,
    )
    eval_split = select_split(split_dataset, SplitKind.EVALUATION)
    mu_true = mean_function(eval_split[TrainingData.x].to_numpy(), empirical_coverage_bootstrap_settings)
    mu_lookup = dict(zip(eval_split[TrainingData.x], mu_true, strict=True))
    targets = intervals[ConfidenceInterval.x].map(mu_lookup).to_numpy()
    covered = (
        (targets >= intervals[ConfidenceInterval.lower].to_numpy())
        & (targets <= intervals[ConfidenceInterval.upper].to_numpy())
    ).mean()
    assert 0.0 < covered <= 1.0


def test_per_resample_seed_changes_predictions(
    split_dataset: DataFrame[SplitDatasetBase],
    unfitted_pipeline: Pipeline,
) -> None:
    train = select_split(split_dataset, SplitKind.TRAINING)
    eval_data = select_split(split_dataset, SplitKind.EVALUATION)
    preds_seed_a = fit_and_predict(unfitted_pipeline, train, eval_data, seed=1)
    preds_seed_b = fit_and_predict(unfitted_pipeline, train, eval_data, seed=99)
    assert not np.array_equal(preds_seed_a, preds_seed_b)


def test_case_resample_preserves_row_count_and_resamples_with_replacement(
    split_dataset: DataFrame[SplitDatasetBase],
) -> None:
    train = select_split(split_dataset, SplitKind.TRAINING)
    rng = np.random.default_rng(0)
    resample = case_resample(train, rng)
    assert len(resample) == len(train)
    duplicates = len(resample) - resample.drop_duplicates().shape[0]
    assert duplicates > 0
