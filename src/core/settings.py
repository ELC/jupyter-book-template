from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(frozen=True)

    n_samples: int = 400
    x_min: float = -10.0
    x_max: float = 10.0
    noise_scale: float = 0.2
    noise_heteroscedasticity: float = 20
    seed: int = 0
    seasonality_amplitude: float = 10.0
    seasonality_frequency: float = 0.5
    n_resamples: int = 200
    rng: int = 0
    confidence_level: float = 0.95
    n_estimators: int = 100
    max_depth: int = 5
    bootstrap_n_jobs: int = -1
    trend_coefficient: float = 0.3
    polynomial_degree: int = 5
    fourier_terms: int = 6
