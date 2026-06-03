import numpy as np
from pandera.typing import DataFrame
from sklearn.pipeline import Pipeline

from analysis import bootstrap_confidence_intervals
from core import ConfidenceInterval, IntervalKind, Settings
from core.schemas import SplitDatasetBase
from prediction import predict


def test_bootstrap_confidence_intervals_is_deterministic(
    split_dataset: DataFrame[SplitDatasetBase],
    unfitted_pipeline: Pipeline,
    deterministic_bootstrap_settings: Settings,
) -> None:
    first = bootstrap_confidence_intervals(unfitted_pipeline, split_dataset, deterministic_bootstrap_settings)
    second = bootstrap_confidence_intervals(unfitted_pipeline, split_dataset, deterministic_bootstrap_settings)
    assert first.equals(second)


def test_bootstrap_confidence_intervals_interval_width_positive(
    split_dataset: DataFrame[SplitDatasetBase],
    unfitted_pipeline: Pipeline,
    positive_width_bootstrap_settings: Settings,
) -> None:
    intervals = bootstrap_confidence_intervals(unfitted_pipeline, split_dataset, positive_width_bootstrap_settings)
    widths = intervals[ConfidenceInterval.upper] - intervals[ConfidenceInterval.lower]
    assert (widths > 0).all()
    assert (intervals[ConfidenceInterval.kind] == IntervalKind.CONFIDENCE).all()


def test_bootstrap_confidence_intervals_uses_settings_defaults(
    split_dataset: DataFrame[SplitDatasetBase],
    unfitted_pipeline: Pipeline,
    ten_resample_bootstrap_settings: Settings,
) -> None:
    intervals = bootstrap_confidence_intervals(
        unfitted_pipeline,
        split_dataset,
        ten_resample_bootstrap_settings,
    )
    assert (intervals[ConfidenceInterval.upper] > intervals[ConfidenceInterval.lower]).all()


def test_bootstrap_confidence_intervals_width_varies_with_x(
    split_dataset: DataFrame[SplitDatasetBase],
    unfitted_pipeline: Pipeline,
    varying_width_bootstrap_settings: Settings,
) -> None:
    intervals = bootstrap_confidence_intervals(unfitted_pipeline, split_dataset, varying_width_bootstrap_settings)
    widths = intervals[ConfidenceInterval.upper] - intervals[ConfidenceInterval.lower]
    assert np.unique(np.round(widths, 4)).size > 1


def test_bootstrap_confidence_intervals_accepts_fitted_template(
    split_dataset: DataFrame[SplitDatasetBase],
    fitted_pipeline: Pipeline,
    minimal_resample_bootstrap_settings: Settings,
) -> None:
    intervals = bootstrap_confidence_intervals(fitted_pipeline, split_dataset, minimal_resample_bootstrap_settings)
    assert len(intervals) > 0


def test_bootstrap_empirical_coverage_near_confidence_level(
    split_dataset: DataFrame[SplitDatasetBase],
    unfitted_pipeline: Pipeline,
    fitted_pipeline: Pipeline,
    empirical_coverage_bootstrap_settings: Settings,
) -> None:
    intervals = bootstrap_confidence_intervals(
        unfitted_pipeline,
        split_dataset,
        empirical_coverage_bootstrap_settings,
    )
    predictions = predict(fitted_pipeline, split_dataset)
    merged = intervals.merge(
        predictions,
        left_on=ConfidenceInterval.x,
        right_on="x",
        how="inner",
    )
    covered = (
        (merged["y_true"] >= merged[ConfidenceInterval.lower]) & (merged["y_true"] <= merged[ConfidenceInterval.upper])
    ).mean()
    assert 0.0 < covered <= 1.0
