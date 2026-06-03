import altair as alt
from pandera.typing import DataFrame

from core import TrainingData
from visualization import plot_dataset


def test_plot_dataset_returns_layer_chart(
    dataset: DataFrame[TrainingData],
) -> None:
    chart = plot_dataset(dataset)
    assert isinstance(chart, alt.LayerChart)
    assert chart.layer is not None
    assert len(chart.layer) == 1
