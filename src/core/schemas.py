from enum import StrEnum, auto

import pandera.pandas as pa
from pandera.typing import Series

DEFAULT_X_MIN = -10.0
DEFAULT_X_MAX = 10.0
_X_FIELD = pa.Field(ge=DEFAULT_X_MIN, le=DEFAULT_X_MAX)


class IntervalKind(StrEnum):
    CONFIDENCE = auto()
    PREDICTION = auto()


class RegressionMetricKind(StrEnum):
    RMSE = auto()
    MAE = auto()
    R2 = auto()


class IntervalMetricKind(StrEnum):
    WIDTH = auto()
    MWI = auto()
    COVERAGE = auto()


class SplitKind(StrEnum):
    TRAINING = auto()
    CALIBRATION = auto()
    EVALUATION = auto()


class ModelKind(StrEnum):
    RANDOM_FOREST = "random_forest"
    SVM = "svm"


class ConformityScoreKind(StrEnum):
    ABSOLUTE = "absolute"
    GAMMA = "gamma"
    RESIDUAL_NORMALIZED = "residual_normalized"


_MODEL_FIELD = pa.Field(isin=[member.value for member in ModelKind])


class TrainingData(pa.DataFrameModel):
    x: Series[float] = _X_FIELD
    y: Series[float]


class SplitDatasetBase(pa.DataFrameModel):
    x: Series[float] = _X_FIELD
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


class IntervalWithGroundTruth(IntervalBase):
    kind: Series[str] = pa.Field(isin=[member.value for member in IntervalKind])
    y_true: Series[float]


class MetricReport(pa.DataFrameModel):
    metric: Series[str] = pa.Field(isin=[member.value for member in RegressionMetricKind])
    lower: Series[float]
    upper: Series[float]


class IntervalMetricReport(pa.DataFrameModel):
    kind: Series[str] = pa.Field(isin=[member.value for member in IntervalKind])
    metric: Series[str] = pa.Field(isin=[member.value for member in IntervalMetricKind])
    lower: Series[float]
    upper: Series[float]


class PredictionsByModel(PredictionsWithGroundTruth):
    mu_true: Series[float]
    model: Series[str] = _MODEL_FIELD


class ConfidenceIntervalByModel(IntervalBase):
    kind: Series[str] = pa.Field(
        default=IntervalKind.CONFIDENCE.value,
        eq=IntervalKind.CONFIDENCE.value,
    )
    model: Series[str] = _MODEL_FIELD

    class Config:
        add_missing_columns = True


class PredictionIntervalByModel(IntervalBase):
    kind: Series[str] = pa.Field(
        default=IntervalKind.PREDICTION.value,
        eq=IntervalKind.PREDICTION.value,
    )
    model: Series[str] = _MODEL_FIELD

    class Config:
        add_missing_columns = True


class MetricReportByModel(MetricReport):
    model: Series[str] = _MODEL_FIELD


class IntervalMetricReportByModel(IntervalMetricReport):
    model: Series[str] = _MODEL_FIELD
