import pandas as pd
from mapie.regression import SplitConformalRegressor
from pandera.typing import DataFrame
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from core.features import prepare_split
from core.schemas import PredictionInterval, SplitDatasetBase, SplitKind
from core.settings import Settings


def fit_conformal(
    data: DataFrame[SplitDatasetBase],
    pipeline: Pipeline,
    settings: Settings,
) -> SplitConformalRegressor:
    estimator = clone(pipeline)
    train = prepare_split(data, SplitKind.TRAINING)
    calib = prepare_split(data, SplitKind.CALIBRATION)
    conformal = SplitConformalRegressor(
        estimator=estimator,
        confidence_level=settings.confidence_level,
        prefit=False,
    )
    conformal.fit(train.x, train.y)
    conformal.conformalize(calib.x, calib.y)
    return conformal


def conformal_intervals(
    model: SplitConformalRegressor,
    data: DataFrame[SplitDatasetBase],
) -> DataFrame[PredictionInterval]:
    eval_split = prepare_split(data, SplitKind.EVALUATION)
    _, intervals = model.predict_interval(eval_split.x)
    lower, upper = intervals[..., 0].T
    return pd.DataFrame(
        {
            PredictionInterval.x: eval_split.x[:, 0],
            PredictionInterval.lower: lower,
            PredictionInterval.upper: upper,
        },
    ).pipe(DataFrame[PredictionInterval])
