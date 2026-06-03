import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from prediction import MultiRegressor


@pytest.fixture
def linear_estimator() -> LinearRegression:
    return LinearRegression()


@pytest.fixture
def tree_estimator() -> DecisionTreeRegressor:
    return DecisionTreeRegressor(random_state=0)


@pytest.fixture
def multi_estimator_pairs(
    linear_estimator: LinearRegression,
    tree_estimator: DecisionTreeRegressor,
) -> list[tuple[str, object]]:
    return [("linear", linear_estimator), ("tree", tree_estimator)]


@pytest.fixture
def multi_regressor(multi_estimator_pairs: list[tuple[str, object]]) -> MultiRegressor:
    return MultiRegressor(estimators=multi_estimator_pairs)


@pytest.fixture
def multi_regressor_x() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(50, 1))


@pytest.fixture
def multi_regressor_y(multi_regressor_x: np.ndarray) -> np.ndarray:
    return multi_regressor_x.ravel() * 2.0 + 1.0


@pytest.fixture
def fitted_multi_regressor(
    multi_regressor: MultiRegressor,
    multi_regressor_x: np.ndarray,
    multi_regressor_y: np.ndarray,
) -> MultiRegressor:
    return multi_regressor.fit(multi_regressor_x, multi_regressor_y)
