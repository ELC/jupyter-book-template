import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from prediction import MultiRegressor


def test_fit_populates_named_estimators(fitted_multi_regressor: MultiRegressor) -> None:
    assert list(fitted_multi_regressor.named_estimators_) == ["linear", "tree"]
    assert fitted_multi_regressor.names_ == ["linear", "tree"]


def test_transform_returns_per_base_predictions(
    fitted_multi_regressor: MultiRegressor,
    multi_regressor_x: np.ndarray,
    multi_estimator_pairs: list[tuple[str, object]],
) -> None:
    transformed = fitted_multi_regressor.transform(multi_regressor_x)
    assert transformed.shape == (len(multi_regressor_x), len(multi_estimator_pairs))
    expected_linear = fitted_multi_regressor.named_estimators_["linear"].predict(multi_regressor_x)
    expected_tree = fitted_multi_regressor.named_estimators_["tree"].predict(multi_regressor_x)
    np.testing.assert_array_equal(transformed[:, 0], expected_linear)
    np.testing.assert_array_equal(transformed[:, 1], expected_tree)


def test_predict_averages_transform(
    fitted_multi_regressor: MultiRegressor,
    multi_regressor_x: np.ndarray,
) -> None:
    np.testing.assert_allclose(
        fitted_multi_regressor.predict(multi_regressor_x),
        fitted_multi_regressor.transform(multi_regressor_x).mean(axis=1),
    )


def test_clone_round_trips(multi_regressor: MultiRegressor) -> None:
    cloned = clone(multi_regressor)
    assert [name for name, _ in cloned.estimators] == ["linear", "tree"]
    assert cloned.named_estimators_ == {}
    assert cloned.estimators_ == []
    assert isinstance(cloned.estimators[0][1], LinearRegression)


def test_sk_visual_block_exposes_named_estimators(
    multi_regressor: MultiRegressor,
    multi_estimator_pairs: list[tuple[str, object]],
) -> None:
    block = multi_regressor._sk_visual_block_()
    expected_names = [name for name, _ in multi_estimator_pairs]
    expected_estimators = [estimator for _, estimator in multi_estimator_pairs]
    assert list(block.names) == expected_names
    assert list(block.estimators) == expected_estimators
    assert block.kind == "parallel"
