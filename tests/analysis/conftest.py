import pytest

from core import Settings


@pytest.fixture
def deterministic_bootstrap_settings() -> Settings:
    return Settings(n_resamples=20, confidence_level=0.90, seed=0)


@pytest.fixture
def positive_width_bootstrap_settings() -> Settings:
    return Settings(n_resamples=30, confidence_level=0.90, seed=1)


@pytest.fixture
def varying_width_bootstrap_settings() -> Settings:
    return Settings(n_resamples=50, confidence_level=0.95, seed=0)


@pytest.fixture
def minimal_resample_bootstrap_settings() -> Settings:
    return Settings(n_resamples=5)


@pytest.fixture
def ten_resample_bootstrap_settings(settings: Settings) -> Settings:
    return Settings(**{**settings.model_dump(), "n_resamples": 10})


@pytest.fixture
def empirical_coverage_bootstrap_settings(settings: Settings) -> Settings:
    return Settings(**{**settings.model_dump(), "n_resamples": 80, "confidence_level": 0.90, "seed": 0})
