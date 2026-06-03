import pytest

from core import Settings


@pytest.fixture
def deterministic_bootstrap_settings() -> Settings:
    return Settings(n_resamples=20, confidence_level=0.90, rng=0)


@pytest.fixture
def positive_width_bootstrap_settings() -> Settings:
    return Settings(n_resamples=30, confidence_level=0.90, rng=1)


@pytest.fixture
def varying_width_bootstrap_settings() -> Settings:
    return Settings(n_resamples=50, confidence_level=0.95, rng=0)


@pytest.fixture
def minimal_resample_bootstrap_settings() -> Settings:
    return Settings(n_resamples=5)
