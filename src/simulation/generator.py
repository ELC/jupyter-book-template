import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from core.schemas import TrainingData
from core.settings import Settings


def _signal(x: np.ndarray) -> np.ndarray:
    return x


def _trend(x: np.ndarray, settings: Settings) -> np.ndarray:
    return settings.trend_coefficient * x


def _seasonality(x: np.ndarray, settings: Settings) -> np.ndarray:
    return settings.seasonality_amplitude * np.cos(settings.seasonality_frequency * x)


def mean_function(x: np.ndarray, settings: Settings) -> np.ndarray:
    return _signal(x) + _trend(x, settings) + _seasonality(x, settings)


def _heteroscedastic_noise(
    rng: np.random.Generator,
    x: np.ndarray,
    settings: Settings,
) -> np.ndarray:
    x_span = settings.x_max - settings.x_min
    x_relative = (x - settings.x_min) / x_span if x_span > 0 else np.zeros_like(x)
    scale = settings.noise_scale * (1.0 + settings.noise_heteroscedasticity * x_relative)
    return rng.normal(0.0, scale, size=settings.n_samples)


def generate_dataset(settings: Settings) -> DataFrame[TrainingData]:
    rng = np.random.default_rng(settings.seed)
    x = rng.uniform(settings.x_min, settings.x_max, size=settings.n_samples)
    noise = _heteroscedastic_noise(rng, x, settings)
    y = mean_function(x, settings) + noise
    return pd.DataFrame({TrainingData.x: x, TrainingData.y: y}).pipe(DataFrame[TrainingData])
