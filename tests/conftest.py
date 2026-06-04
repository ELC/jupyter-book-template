from typing import cast

import pytest
from pandera.typing import DataFrame
from sklearn.pipeline import Pipeline

from core import Settings, SplitKind, TrainingData, build_split_dataset, select_split
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
def split_dataset(dataset: DataFrame[TrainingData]) -> DataFrame[SplitDatasetBase]:
    return build_split_dataset(dataset)


@pytest.fixture
def train_fold(split_dataset: DataFrame[SplitDatasetBase]) -> DataFrame[TrainingData]:
    return select_split(split_dataset, SplitKind.TRAINING)


@pytest.fixture
def calibration_fold(split_dataset: DataFrame[SplitDatasetBase]) -> DataFrame[TrainingData]:
    return select_split(split_dataset, SplitKind.CALIBRATION)


@pytest.fixture
def evaluation_fold(split_dataset: DataFrame[SplitDatasetBase]) -> DataFrame[TrainingData]:
    return select_split(split_dataset, SplitKind.EVALUATION)


@pytest.fixture
def unfitted_pipeline(settings: Settings) -> Pipeline:
    return regression_pipeline(random_forest_regressor(settings), settings)


@pytest.fixture
def fitted_pipeline(
    split_dataset: DataFrame[SplitDatasetBase],
    unfitted_pipeline: Pipeline,
) -> Pipeline:
    return fit_pipeline(split_dataset, unfitted_pipeline)
