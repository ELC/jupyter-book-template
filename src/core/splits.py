import pandas as pd
from mapie.utils import train_conformalize_test_split
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
    data: DataFrame[TrainingData],
    *,
    train_size: float = 0.7,
    conformalize_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 0,
) -> DataFrame[SplitDatasetBase]:
    train, calibration, evaluation, *_ = train_conformalize_test_split(
        data,
        data[TrainingData.y],
        train_size=train_size,
        conformalize_size=conformalize_size,
        test_size=test_size,
        random_state=random_state,
    )
    parts: list[pd.DataFrame] = [
        TrainingData.validate(train).pipe(DataFrame[TrainingSplit]),
        TrainingData.validate(calibration).pipe(DataFrame[CalibrationSplit]),
        TrainingData.validate(evaluation).pipe(DataFrame[EvaluationSplit]),
    ]
    return pd.concat(parts, ignore_index=True).pipe(DataFrame[SplitDatasetBase])


def select_split(
    data: DataFrame[SplitDatasetBase],
    split: SplitKind,
) -> DataFrame[TrainingData]:
    mask = data[SplitDatasetBase.split] == split
    subset = data.loc[mask, [TrainingData.x, TrainingData.y]].reset_index(drop=True)
    return subset.pipe(DataFrame[TrainingData])
