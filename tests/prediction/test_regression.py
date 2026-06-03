from pandera.typing import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from core import CompositeFeatures, Predictions, PredictionsWithGroundTruth, Settings, SplitKind, select_split
from core.features import expand_features
from core.schemas import SplitDatasetBase, TrainingData
from prediction import (
    FEATURES_STEP,
    REGRESSOR_STEP,
    fit_pipeline,
    predict,
    random_forest_regressor,
    regression_pipeline,
)


def test_random_forest_regressor_uses_settings(settings: Settings) -> None:
    model = random_forest_regressor(settings)
    assert model.n_estimators == settings.n_estimators
    assert model.max_depth == settings.max_depth
    assert model.random_state == settings.seed
    assert model.n_jobs == 1


def test_regression_pipeline_uses_supplied_regressor(settings: Settings) -> None:
    regressor = random_forest_regressor(settings)
    pipeline = regression_pipeline(regressor, settings)
    assert pipeline.named_steps[REGRESSOR_STEP] is regressor
    assert isinstance(pipeline.named_steps[FEATURES_STEP], CompositeFeatures)


def test_regression_pipeline_accepts_alternative_regressor(settings: Settings) -> None:
    regressor = LinearRegression()
    pipeline = regression_pipeline(regressor, settings)
    assert pipeline.named_steps[REGRESSOR_STEP] is regressor


def test_fit_pipeline_fits_training_features(
    split_dataset: DataFrame[SplitDatasetBase],
    settings: Settings,
    unfitted_pipeline: Pipeline,
) -> None:
    fitted = fit_pipeline(split_dataset, unfitted_pipeline)
    train = select_split(split_dataset, SplitKind.TRAINING)
    train_features = expand_features(settings).fit_transform(train[TrainingData.x].to_numpy())
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
