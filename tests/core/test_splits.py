from pandera.typing import DataFrame

from core import SplitDatasetBase, SplitKind, TrainingData, build_split_dataset, select_split


def test_build_split_dataset_partitions_into_three_folds(
    dataset: DataFrame[TrainingData],
) -> None:
    split = build_split_dataset(dataset)
    assert len(split) == len(dataset)
    counts = split[SplitDatasetBase.split].value_counts()
    assert counts[SplitKind.TRAINING.value] > 0
    assert counts[SplitKind.CALIBRATION.value] > 0
    assert counts[SplitKind.EVALUATION.value] > 0
    assert counts[SplitKind.TRAINING.value] + counts[SplitKind.CALIBRATION.value] + counts[
        SplitKind.EVALUATION.value
    ] == len(dataset)


def test_build_split_dataset_honors_proportions(
    dataset: DataFrame[TrainingData],
) -> None:
    split = build_split_dataset(
        dataset,
        train_size=0.6,
        conformalize_size=0.2,
        test_size=0.2,
    )
    counts = split[SplitDatasetBase.split].value_counts()
    assert counts[SplitKind.TRAINING.value] == int(len(dataset) * 0.6)
    assert counts[SplitKind.CALIBRATION.value] == int(len(dataset) * 0.2)
    assert counts[SplitKind.EVALUATION.value] == int(len(dataset) * 0.2)


def test_build_split_dataset_is_deterministic(
    dataset: DataFrame[TrainingData],
) -> None:
    first = build_split_dataset(dataset, random_state=42)
    second = build_split_dataset(dataset, random_state=42)
    assert first.equals(second)


def test_select_split_returns_training_data_subset(
    dataset: DataFrame[TrainingData],
    selected_split_kind: SplitKind,
) -> None:
    split = build_split_dataset(dataset)
    selected = select_split(split, selected_split_kind)
    expected_count = (split[SplitDatasetBase.split] == selected_split_kind.value).sum()
    assert len(selected) == expected_count
    assert list(selected.columns) == [TrainingData.x, TrainingData.y]
