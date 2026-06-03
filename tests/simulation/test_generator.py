import numpy as np

from core import Settings, TrainingData
from simulation import generate_dataset


def test_generate_dataset_is_deterministic(
    deterministic_generate_dataset_settings: Settings,
) -> None:
    first = generate_dataset(deterministic_generate_dataset_settings)
    second = generate_dataset(deterministic_generate_dataset_settings)
    assert first.equals(second)


def test_generate_dataset_respects_sample_count(
    sample_count_generate_dataset_settings: Settings,
) -> None:
    dataset = generate_dataset(sample_count_generate_dataset_settings)
    assert len(dataset) == 75


def test_generate_dataset_x_within_bounds(
    bounded_x_generate_dataset_settings: Settings,
) -> None:
    dataset = generate_dataset(bounded_x_generate_dataset_settings)
    assert dataset[TrainingData.x].min() >= -10.0
    assert dataset[TrainingData.x].max() <= 10.0


def test_heteroscedastic_noise_is_lower_at_x_min(
    heteroscedastic_noise_settings: Settings,
) -> None:
    dataset = generate_dataset(heteroscedastic_noise_settings)
    x = dataset[TrainingData.x].to_numpy()
    noise = dataset[TrainingData.y].to_numpy() - x
    assert np.var(noise[x <= heteroscedastic_noise_settings.x_min + 1.0]) < np.var(
        noise[x >= heteroscedastic_noise_settings.x_max - 1.0]
    )


def test_generate_dataset_accepts_zero_x_span(
    constant_x_generate_dataset_settings: Settings,
) -> None:
    dataset = generate_dataset(constant_x_generate_dataset_settings)
    assert len(dataset) == constant_x_generate_dataset_settings.n_samples
    assert (dataset[TrainingData.x] == constant_x_generate_dataset_settings.x_min).all()
