import altair as alt
from pandera.typing import DataFrame

from core.schemas import TrainingData
from visualization._common import configure_altair


def plot_dataset(data: DataFrame[TrainingData]) -> alt.FacetChart | alt.LayerChart:
    configure_altair()
    scatter = (
        alt
        .Chart(data)
        .mark_circle(opacity=0.5, size=30)
        .encode(
            x=alt.X(f"{TrainingData.x}:Q", title="x"),
            y=alt.Y(f"{TrainingData.y}:Q", title="y"),
            tooltip=[f"{TrainingData.x}:Q", f"{TrainingData.y}:Q"],
        )
    )
    return alt.layer(scatter).properties(
        width=640,
        height=400,
        title="Synthetic training data",
    )
