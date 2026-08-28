from sklearn.ensemble import RandomForestRegressor

from core import Settings


def random_forest_regressor(settings: Settings) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=settings.n_estimators,
        max_depth=settings.max_depth,
        random_state=settings.seed,
        n_jobs=1,
    )
