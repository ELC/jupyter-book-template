import pytest

from core import Settings


@pytest.fixture
def deterministic_generate_dataset_settings() -> Settings:
    return Settings(n_samples=100, seed=42)


@pytest.fixture
def sample_count_generate_dataset_settings() -> Settings:
    return Settings(n_samples=75, seed=1)


@pytest.fixture
def bounded_x_generate_dataset_settings() -> Settings:
    return Settings(n_samples=200, seed=2, x_min=-10.0, x_max=10.0)


@pytest.fixture
def heteroscedastic_noise_settings() -> Settings:
    return Settings(
        n_samples=10_000,
        seed=0,
        x_min=-10.0,
        x_max=10.0,
        trend_coefficient=0.0,
        seasonality_amplitude=0.0,
        noise_heteroscedasticity=0.5,
    )


@pytest.fixture
def constant_x_generate_dataset_settings() -> Settings:
    return Settings(
        n_samples=50,
        seed=0,
        x_min=5.0,
        x_max=5.0,
        trend_coefficient=0.0,
        seasonality_amplitude=0.0,
    )
