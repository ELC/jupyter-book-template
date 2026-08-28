from pandera.typing import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from core import (
    Predictions,
    PredictionsWithGroundTruth,
    Settings,
    SplitDatasetBase,
    SplitKind,
    TrainingData,
    expand_features,
    select_split,
)
from prediction import (
    FEATURES_STEP,
    REGRESSOR_STEP,
    fit_pipeline,
    predict,
    random_forest_regressor,
    regression_pipeline,
)


def test_regression_pipeline_uses_supplied_regressor(settings: Settings) -> None:
    regressor = random_forest_regressor(settings)
    features = expand_features(settings)
    pipeline = regression_pipeline(regressor, features)
    assert pipeline.named_steps[REGRESSOR_STEP] is regressor
    features_step = pipeline.named_steps[FEATURES_STEP]
    assert features_step is features
    assert isinstance(features_step, FeatureUnion | FunctionTransformer)


def test_regression_pipeline_accepts_alternative_regressor(settings: Settings) -> None:
    regressor = LinearRegression()
    pipeline = regression_pipeline(regressor, expand_features(settings))
    assert pipeline.named_steps[REGRESSOR_STEP] is regressor


def test_regression_pipeline_shares_features_across_regressors(settings: Settings) -> None:
    features = expand_features(settings)
    rf_pipeline = regression_pipeline(random_forest_regressor(settings), features)
    lr_pipeline = regression_pipeline(LinearRegression(), features)
    assert rf_pipeline.named_steps[FEATURES_STEP] is lr_pipeline.named_steps[FEATURES_STEP]


def test_fit_pipeline_fits_training_features(
    split_dataset: DataFrame[SplitDatasetBase],
    settings: Settings,
    unfitted_pipeline: Pipeline,
) -> None:
    fitted = fit_pipeline(split_dataset, unfitted_pipeline)
    train = select_split(split_dataset, SplitKind.TRAINING)
    train_features = expand_features(settings).fit_transform(train[[TrainingData.x]].to_numpy())
    assert fitted.named_steps[REGRESSOR_STEP].n_features_in_ == train_features.shape[1]


def test_predict_returns_subset_for_selected_split(
    split_dataset: DataFrame[SplitDatasetBase],
    fitted_pipeline: Pipeline,
    selected_split_kind: SplitKind,
) -> None:
    predictions = predict(fitted_pipeline, split_dataset, split=selected_split_kind)
    expected = select_split(split_dataset, selected_split_kind)
    assert len(predictions) == len(expected)
    assert PredictionsWithGroundTruth.y_true in predictions.columns
    assert predictions[Predictions.x].tolist() == expected[TrainingData.x].tolist()
    assert predictions[PredictionsWithGroundTruth.y_true].tolist() == expected[TrainingData.y].tolist()
    assert predictions[Predictions.y_pred].notna().all()
