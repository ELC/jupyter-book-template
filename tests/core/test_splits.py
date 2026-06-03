from pandera.typing import DataFrame

from core import SplitKind, TrainingData, build_split_dataset, select_split
from core.schemas import SplitDatasetBase


def test_build_split_dataset_assigns_splits(
    train_fold: DataFrame[TrainingData],
    calibration_fold: DataFrame[TrainingData],
    evaluation_fold: DataFrame[TrainingData],
) -> None:
    dataset = build_split_dataset(train_fold, calibration_fold, evaluation_fold)
    assert len(dataset) == len(train_fold) + len(calibration_fold) + len(evaluation_fold)
    assert dataset[SplitDatasetBase.split].tolist() == (
        [SplitKind.TRAINING.value] * len(train_fold)
        + [SplitKind.CALIBRATION.value] * len(calibration_fold)
        + [SplitKind.EVALUATION.value] * len(evaluation_fold)
    )
    assert select_split(dataset, SplitKind.TRAINING).equals(train_fold.reset_index(drop=True))
    assert select_split(dataset, SplitKind.CALIBRATION).equals(calibration_fold.reset_index(drop=True))
    assert select_split(dataset, SplitKind.EVALUATION).equals(evaluation_fold.reset_index(drop=True))


def test_select_split_returns_training_data_subset(
    train_fold: DataFrame[TrainingData],
    calibration_fold: DataFrame[TrainingData],
    evaluation_fold: DataFrame[TrainingData],
    selected_split_kind: SplitKind,
    expected_fold: DataFrame[TrainingData],
) -> None:
    dataset = build_split_dataset(train_fold, calibration_fold, evaluation_fold)
    selected = select_split(dataset, selected_split_kind)
    assert selected.equals(expected_fold.reset_index(drop=True))
