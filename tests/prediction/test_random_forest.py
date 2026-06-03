from core import Settings
from prediction import random_forest_regressor


def test_random_forest_regressor_uses_settings(settings: Settings) -> None:
    model = random_forest_regressor(settings)
    assert model.n_estimators == settings.n_estimators
    assert model.max_depth == settings.max_depth
    assert model.random_state == settings.seed
    assert model.n_jobs == 1
