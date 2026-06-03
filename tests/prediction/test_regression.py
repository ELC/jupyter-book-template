from pandera.typing import DataFrame
from sklearn.ensemble import RandomForestRegressor

from core import Predictions, PredictionsWithGroundTruth, Settings, SplitKind, select_split
from core.features import expand_features
from core.schemas import SplitDatasetBase, TrainingData
from prediction import fit_random_forest, predict, random_forest_regressor


def test_random_forest_regressor_uses_settings(settings: Settings) -> None:
    model = random_forest_regressor(settings)
    assert model.n_estimators == settings.n_estimators
    assert model.max_depth == settings.max_depth
    assert model.random_state == settings.seed
    assert model.n_jobs == 1


def test_fit_random_forest_fits_training_features(
    split_dataset: DataFrame[SplitDatasetBase],
    settings: Settings,
    unfitted_regressor: RandomForestRegressor,
) -> None:
    fitted = fit_random_forest(split_dataset, unfitted_regressor, settings)
    train = select_split(split_dataset, SplitKind.TRAINING)
    train_features = expand_features(train[TrainingData.x].to_numpy(), settings)
    assert fitted.n_features_in_ == train_features.shape[1]


def test_predict_returns_subset_for_selected_split(
    split_dataset: DataFrame[SplitDatasetBase],
    settings: Settings,
    fitted_model: RandomForestRegressor,
    selected_predict_split: SplitKind,
) -> None:
    predictions = predict(fitted_model, split_dataset, settings, split=selected_predict_split)
    expected = select_split(split_dataset, selected_predict_split)
    assert len(predictions) == len(expected)
    assert PredictionsWithGroundTruth.y_true in predictions.columns
    assert predictions[Predictions.x].tolist() == expected[TrainingData.x].tolist()
    assert predictions[PredictionsWithGroundTruth.y_true].tolist() == expected[TrainingData.y].tolist()
    assert predictions[Predictions.y_pred].notna().all()
