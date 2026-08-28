import pandas as pd
from pandera.typing import DataFrame
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline

from core import Predictions, PredictionsWithGroundTruth, SplitDatasetBase, SplitKind, prepare_split

FEATURES_STEP = "features"
REGRESSOR_STEP = "regressor"
ENSEMBLE_STEP = "ensemble"


def regression_pipeline(regressor: BaseEstimator, features: BaseEstimator) -> Pipeline:
    return Pipeline(
        steps=[
            (FEATURES_STEP, features),
            (REGRESSOR_STEP, regressor),
        ],
    )


def fit_pipeline(
    data: DataFrame[SplitDatasetBase],
    estimator: BaseEstimator,
) -> BaseEstimator:
    fitted = clone(estimator)
    train = prepare_split(data, SplitKind.TRAINING)
    fitted.fit(train.x, train.y)
    return fitted


def predict(
    estimator: BaseEstimator,
    data: DataFrame[SplitDatasetBase],
    *,
    split: SplitKind = SplitKind.EVALUATION,
) -> DataFrame[PredictionsWithGroundTruth]:
    split_data = prepare_split(data, split)
    y_pred = estimator.predict(split_data.x)
    y_true = split_data.y.to_numpy()
    return pd.DataFrame(
        {
            Predictions.x: split_data.x[:, 0],
            Predictions.y_pred: y_pred,
            PredictionsWithGroundTruth.y_true: y_true,
        },
    ).pipe(DataFrame[PredictionsWithGroundTruth])
