from typing import cast

import pytest
from pandera.typing import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from core import Settings, SplitKind, TrainingData, build_split_dataset
from core.schemas import SplitDatasetBase
from prediction import fit_pipeline, random_forest_regressor, regression_pipeline
from simulation import generate_dataset


@pytest.fixture(
    params=[pytest.param(split_kind, id=split_kind.value) for split_kind in SplitKind],
)
def selected_split_kind(request: pytest.FixtureRequest) -> SplitKind:
    return cast("SplitKind", request.param)


@pytest.fixture
def settings() -> Settings:
    return Settings(n_samples=200, seed=0, n_estimators=50)


@pytest.fixture
def dataset(settings: Settings) -> DataFrame[TrainingData]:
    return generate_dataset(settings)


@pytest.fixture
def train_remainder_split(
    dataset: DataFrame[TrainingData],
) -> tuple[DataFrame[TrainingData], DataFrame[TrainingData]]:
    train, remainder = train_test_split(dataset, test_size=0.4, random_state=0)
    return (
        train.pipe(DataFrame[TrainingData]),
        remainder.pipe(DataFrame[TrainingData]),
    )


@pytest.fixture
def train_fold(
    train_remainder_split: tuple[DataFrame[TrainingData], DataFrame[TrainingData]],
) -> DataFrame[TrainingData]:
    return train_remainder_split[0]


@pytest.fixture
def remainder_fold(
    train_remainder_split: tuple[DataFrame[TrainingData], DataFrame[TrainingData]],
) -> DataFrame[TrainingData]:
    return train_remainder_split[1]


@pytest.fixture
def calibration_evaluation_split(
    remainder_fold: DataFrame[TrainingData],
) -> tuple[DataFrame[TrainingData], DataFrame[TrainingData]]:
    calibration, evaluation = train_test_split(remainder_fold, test_size=0.5, random_state=0)
    return (
        calibration.pipe(DataFrame[TrainingData]),
        evaluation.pipe(DataFrame[TrainingData]),
    )


@pytest.fixture
def calibration_fold(
    calibration_evaluation_split: tuple[DataFrame[TrainingData], DataFrame[TrainingData]],
) -> DataFrame[TrainingData]:
    return calibration_evaluation_split[0]


@pytest.fixture
def evaluation_fold(
    calibration_evaluation_split: tuple[DataFrame[TrainingData], DataFrame[TrainingData]],
) -> DataFrame[TrainingData]:
    return calibration_evaluation_split[1]


@pytest.fixture
def split_dataset(
    train_fold: DataFrame[TrainingData],
    calibration_fold: DataFrame[TrainingData],
    evaluation_fold: DataFrame[TrainingData],
) -> DataFrame[SplitDatasetBase]:
    return build_split_dataset(train_fold, calibration_fold, evaluation_fold)


@pytest.fixture
def unfitted_pipeline(settings: Settings) -> Pipeline:
    return regression_pipeline(random_forest_regressor(settings), settings)


@pytest.fixture
def fitted_pipeline(
    split_dataset: DataFrame[SplitDatasetBase],
    unfitted_pipeline: Pipeline,
) -> Pipeline:
    return fit_pipeline(split_dataset, unfitted_pipeline)
