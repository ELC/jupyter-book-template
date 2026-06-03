import pandas as pd
from pandera.typing import DataFrame
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor

from core.features import expand_features
from core.schemas import Predictions, PredictionsWithGroundTruth, SplitDatasetBase, SplitKind, TrainingData
from core.settings import Settings
from core.splits import select_split


def random_forest_regressor(settings: Settings) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=settings.n_estimators,
        max_depth=settings.max_depth,
        random_state=settings.seed,
        n_jobs=1,
    )


def fit_random_forest(
    data: DataFrame[SplitDatasetBase],
    model: RandomForestRegressor,
    settings: Settings,
) -> RandomForestRegressor:
    fitted = clone(model)
    train = select_split(data, SplitKind.TRAINING)
    train_x = expand_features(train[TrainingData.x].to_numpy(), settings)
    train_y = train[TrainingData.y]
    fitted.fit(train_x, train_y)
    return fitted


def predict(
    model: RandomForestRegressor,
    data: DataFrame[SplitDatasetBase],
    settings: Settings,
    *,
    split: SplitKind = SplitKind.EVALUATION,
) -> DataFrame[PredictionsWithGroundTruth]:
    eval_data = select_split(data, split)
    eval_x = expand_features(eval_data[TrainingData.x].to_numpy(), settings)
    return pd.DataFrame(
        {
            Predictions.x: eval_data[TrainingData.x].to_numpy(),
            Predictions.y_pred: model.predict(eval_x),
            PredictionsWithGroundTruth.y_true: eval_data[TrainingData.y].to_numpy(),
        },
    ).pipe(DataFrame[PredictionsWithGroundTruth])
