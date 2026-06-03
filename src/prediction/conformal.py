import pandas as pd
from mapie.regression import SplitConformalRegressor
from pandera.typing import DataFrame
from sklearn.base import RegressorMixin, clone

from core.features import expand_features, prepare_split
from core.schemas import PredictionInterval, SplitDatasetBase, SplitKind
from core.settings import Settings


def fit_conformal(
    data: DataFrame[SplitDatasetBase],
    model: RegressorMixin,
    settings: Settings,
) -> SplitConformalRegressor:
    estimator = clone(model)
    transformer = expand_features(settings)
    train = prepare_split(data, SplitKind.TRAINING, settings, transformer=transformer)
    calib = prepare_split(data, SplitKind.CALIBRATION, settings, transformer=transformer)
    conformal = SplitConformalRegressor(
        estimator=estimator,
        confidence_level=settings.confidence_level,
        prefit=False,
    )
    conformal.fit(train.features, train.y)
    conformal.conformalize(calib.features, calib.y)
    return conformal


def conformal_intervals(
    model: SplitConformalRegressor,
    data: DataFrame[SplitDatasetBase],
    settings: Settings,
) -> DataFrame[PredictionInterval]:
    eval_split = prepare_split(data, SplitKind.EVALUATION, settings)
    _, intervals = model.predict_interval(eval_split.features)
    lower, upper = intervals[..., 0].T
    return pd.DataFrame(
        {
            PredictionInterval.x: eval_split.x,
            PredictionInterval.lower: lower,
            PredictionInterval.upper: upper,
        },
    ).pipe(DataFrame[PredictionInterval])
