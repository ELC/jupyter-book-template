import pandas as pd
from pandera.typing import DataFrame

from core.schemas import (
    CalibrationSplit,
    EvaluationSplit,
    SplitDatasetBase,
    SplitKind,
    TrainingData,
    TrainingSplit,
)


def build_split_dataset(
    train: DataFrame[TrainingData],
    calibration: DataFrame[TrainingData],
    evaluation: DataFrame[TrainingData],
) -> DataFrame[SplitDatasetBase]:
    parts = [
        train.pipe(DataFrame[TrainingSplit]),
        calibration.pipe(DataFrame[CalibrationSplit]),
        evaluation.pipe(DataFrame[EvaluationSplit]),
    ]
    return pd.concat(parts, ignore_index=True).pipe(DataFrame[SplitDatasetBase])


def select_split(
    data: DataFrame[SplitDatasetBase],
    split: SplitKind,
) -> DataFrame[TrainingData]:
    mask = data[SplitDatasetBase.split] == split
    subset = data.loc[mask, [TrainingData.x, TrainingData.y]].reset_index(drop=True)
    return subset.pipe(DataFrame[TrainingData])
