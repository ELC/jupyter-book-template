from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from core import Settings
from prediction import SCALER_STEP, SVR_STEP, svm_regressor


def test_svm_regressor_uses_settings(settings: Settings) -> None:
    pipeline = svm_regressor(settings)
    assert isinstance(pipeline, Pipeline)
    scaler = pipeline.named_steps[SCALER_STEP]
    svr = pipeline.named_steps[SVR_STEP]
    assert isinstance(scaler, StandardScaler)
    assert isinstance(svr, SVR)
    assert settings.svm_gamma == svr.gamma
    assert svr.kernel == settings.svm_kernel
