import pandas as pd
from pandera.typing import DataFrame
from sklearn.base import RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from core.features import expand_features, prepare_split
from core.schemas import Predictions, PredictionsWithGroundTruth, SplitDatasetBase, SplitKind
from core.settings import Settings

FEATURES_STEP = "features"
REGRESSOR_STEP = "regressor"


def random_forest_regressor(settings: Settings) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=settings.n_estimators,
        max_depth=settings.max_depth,
        random_state=settings.seed,
        n_jobs=1,
    )


def regression_pipeline(regressor: RegressorMixin, settings: Settings) -> Pipeline:
    return Pipeline(
        steps=[
            (FEATURES_STEP, expand_features(settings)),
            (REGRESSOR_STEP, regressor),
        ],
    )


def fit_pipeline(
    data: DataFrame[SplitDatasetBase],
    pipeline: Pipeline,
) -> Pipeline:
    fitted = clone(pipeline)
    train = prepare_split(data, SplitKind.TRAINING)
    fitted.fit(train.x, train.y)
    return fitted


def predict(
    pipeline: Pipeline,
    data: DataFrame[SplitDatasetBase],
    *,
    split: SplitKind = SplitKind.EVALUATION,
) -> DataFrame[PredictionsWithGroundTruth]:
    split_data = prepare_split(data, split)
    y_pred = pipeline.predict(split_data.x)
    y_true = split_data.y.to_numpy()
    return pd.DataFrame(
        {
            Predictions.x: split_data.x,
            Predictions.y_pred: y_pred,
            PredictionsWithGroundTruth.y_true: y_true,
        },
    ).pipe(DataFrame[PredictionsWithGroundTruth])
