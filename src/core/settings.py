from typing import Self

import numpy as np
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.schemas import DEFAULT_X_MAX, DEFAULT_X_MIN, ConformityScoreKind


class Settings(BaseSettings):
    model_config = SettingsConfigDict(frozen=True)

    n_samples: int = Field(default=400, gt=0)
    x_min: float = Field(default=DEFAULT_X_MIN)
    x_max: float = Field(default=DEFAULT_X_MAX)
    noise_scale: float = 0.2
    noise_heteroscedasticity: float = 20
    seed: int = 0
    seasonality_amplitude: float = 10.0
    seasonality_frequency: float = 0.5
    n_resamples: int = Field(default=200, gt=0)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    n_estimators: int = Field(default=100, gt=0)
    max_depth: int = 5
    bootstrap_n_jobs: int = 1
    trend_coefficient: float = 0.3
    polynomial_degree: int = Field(default=5, ge=0)
    fourier_terms: int = Field(default=6, ge=0)
    svm_gamma: float = Field(default=0.1, gt=0)
    svm_kernel: str = "rbf"
    conformity_score: ConformityScoreKind = ConformityScoreKind.RESIDUAL_NORMALIZED

    @property
    def bootstrap_seed(self) -> int:
        child_seeds = np.random.SeedSequence(self.seed).spawn(2)
        return int(child_seeds[1].generate_state(1)[0])

    @model_validator(mode="after")
    def validate_x_range(self) -> Self:
        if self.x_min > self.x_max:  # pragma: no cover
            msg = "x_min must be less than or equal to x_max"
            raise ValueError(msg)
        return self
