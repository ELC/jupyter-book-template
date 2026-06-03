from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from core.settings import Settings

SCALER_STEP = "scaler"
SVR_STEP = "svr"


def svm_regressor(settings: Settings) -> Pipeline:
    return Pipeline(
        steps=[
            (SCALER_STEP, StandardScaler()),
            (SVR_STEP, SVR(gamma=settings.svm_gamma, kernel=settings.svm_kernel)),
        ],
    )
