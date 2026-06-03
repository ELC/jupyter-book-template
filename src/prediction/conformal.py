import pandas as pd
from mapie.regression import SplitConformalRegressor
from pandera.typing import DataFrame
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor

from core.features import expand_features
from core.schemas import PredictionInterval, SplitDatasetBase, SplitKind, TrainingData
from core.settings import Settings
from core.splits import select_split


def fit_conformal(
    data: DataFrame[SplitDatasetBase],
    model: RandomForestRegressor,
    settings: Settings,
) -> SplitConformalRegressor:
    estimator = clone(model)
    train = select_split(data, SplitKind.TRAINING)
    calib = select_split(data, SplitKind.CALIBRATION)
    train_x = expand_features(train[TrainingData.x].to_numpy(), settings)
    calib_x = expand_features(calib[TrainingData.x].to_numpy(), settings)
    train_y = train[TrainingData.y]
    calib_y = calib[TrainingData.y]
    conformal = SplitConformalRegressor(
        estimator=estimator,
        confidence_level=settings.confidence_level,
        prefit=False,
    )
    conformal.fit(train_x, train_y)
    conformal.conformalize(calib_x, calib_y)
    return conformal


def conformal_intervals(
    model: SplitConformalRegressor,
    data: DataFrame[SplitDatasetBase],
    _settings: Settings,
) -> DataFrame[PredictionInterval]:
    eval_data = select_split(data, SplitKind.EVALUATION)
    eval_x = expand_features(eval_data[TrainingData.x].to_numpy(), _settings)
    _, intervals = model.predict_interval(eval_x)
    lower, upper = intervals[..., 0].T
    return pd.DataFrame(
        {
            PredictionInterval.x: eval_data[TrainingData.x].to_numpy(),
            PredictionInterval.lower: lower,
            PredictionInterval.upper: upper,
        },
    ).pipe(DataFrame[PredictionInterval])
