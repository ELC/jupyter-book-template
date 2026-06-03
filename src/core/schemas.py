from enum import StrEnum, auto

import pandera.pandas as pa
from pandera.typing import Series


class IntervalKind(StrEnum):
    CONFIDENCE = auto()
    PREDICTION = auto()


class SplitKind(StrEnum):
    TRAINING = auto()
    CALIBRATION = auto()
    EVALUATION = auto()


class TrainingData(pa.DataFrameModel):
    x: Series[float] = pa.Field(ge=-10.0, le=10.0)
    y: Series[float]


class SplitDatasetBase(pa.DataFrameModel):
    x: Series[float] = pa.Field(ge=-10.0, le=10.0)
    y: Series[float]
    split: Series[str]


class TrainingSplit(SplitDatasetBase):
    split: Series[str] = pa.Field(
        default=SplitKind.TRAINING.value,
        eq=SplitKind.TRAINING.value,
    )

    class Config:
        add_missing_columns = True


class CalibrationSplit(SplitDatasetBase):
    split: Series[str] = pa.Field(
        default=SplitKind.CALIBRATION.value,
        eq=SplitKind.CALIBRATION.value,
    )

    class Config:
        add_missing_columns = True


class EvaluationSplit(SplitDatasetBase):
    split: Series[str] = pa.Field(
        default=SplitKind.EVALUATION.value,
        eq=SplitKind.EVALUATION.value,
    )

    class Config:
        add_missing_columns = True


class Predictions(pa.DataFrameModel):
    x: Series[float]
    y_pred: Series[float]


class PredictionsWithGroundTruth(Predictions):
    y_true: Series[float]


class IntervalBase(pa.DataFrameModel):
    x: Series[float]
    lower: Series[float]
    upper: Series[float]


class ConfidenceInterval(IntervalBase):
    kind: Series[str] = pa.Field(
        default=IntervalKind.CONFIDENCE.value,
        eq=IntervalKind.CONFIDENCE.value,
    )

    class Config:
        add_missing_columns = True


class PredictionInterval(IntervalBase):
    kind: Series[str] = pa.Field(
        default=IntervalKind.PREDICTION.value,
        eq=IntervalKind.PREDICTION.value,
    )

    class Config:
        add_missing_columns = True
