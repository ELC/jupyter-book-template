import pandas as pd
from pandera.typing import DataFrame
from sklearn.base import RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor

from core.features import prepare_split
from core.schemas import Predictions, PredictionsWithGroundTruth, SplitDatasetBase, SplitKind
from core.settings import Settings


def random_forest_regressor(settings: Settings) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=settings.n_estimators,
        max_depth=settings.max_depth,
        random_state=settings.seed,
        n_jobs=1,
    )


def fit_random_forest(
    data: DataFrame[SplitDatasetBase],
    model: RegressorMixin,
    settings: Settings,
) -> RandomForestRegressor:
    fitted = clone(model)
    train = prepare_split(data, SplitKind.TRAINING, settings)
    fitted.fit(train.features, train.y)
    return fitted


def predict(
    model: RegressorMixin,
    data: DataFrame[SplitDatasetBase],
    settings: Settings,
    *,
    split: SplitKind = SplitKind.EVALUATION,
) -> DataFrame[PredictionsWithGroundTruth]:
    split_data = prepare_split(data, split, settings)
    y_pred = model.predict(split_data.features)
    y_true = split_data.y.to_numpy()
    return pd.DataFrame(
        {
            Predictions.x: split_data.x,
            Predictions.y_pred: y_pred,
            PredictionsWithGroundTruth.y_true: y_true,
        },
    ).pipe(DataFrame[PredictionsWithGroundTruth])
