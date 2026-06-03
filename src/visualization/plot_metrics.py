import altair as alt
import pandas as pd
from pandera.typing import DataFrame

from core.schemas import (
    IntervalMetricKind,
    IntervalMetricReport,
    MetricReport,
    RegressionMetricKind,
)
from visualization._common import configure_altair
from visualization.theme import (
    CHART_HEIGHT,
    CHART_WIDTH,
    INTERVAL_COLOR_DOMAIN,
    INTERVAL_COLOR_RANGE,
)


def plot_regression_metrics(metrics: DataFrame[MetricReport]) -> alt.FacetChart | alt.LayerChart:
    configure_altair()
    return (
        alt
        .Chart(metrics)
        .mark_errorbar(ticks=True)
        .encode(
            x=alt.X(
                f"{MetricReport.metric}:N",
                title=None,
                sort=[member.value for member in RegressionMetricKind],
            ),
            y=alt.Y(f"{MetricReport.lower}:Q", title="Metric value"),
            y2=f"{MetricReport.upper}:Q",
            tooltip=[
                f"{MetricReport.metric}:N",
                f"{MetricReport.lower}:Q",
                f"{MetricReport.upper}:Q",
            ],
        )
        .properties(
            width=CHART_WIDTH,
            height=CHART_HEIGHT,
            title="Regression metrics (bootstrap CI)",
        )
    )


def plot_interval_metrics(
    confidence: DataFrame[IntervalMetricReport],
    prediction: DataFrame[IntervalMetricReport],
) -> alt.FacetChart | alt.LayerChart:
    configure_altair()
    report = pd.concat([confidence, prediction], ignore_index=True).pipe(
        DataFrame[IntervalMetricReport],
    )
    return (
        alt
        .Chart(report)
        .mark_bar(width=20)
        .encode(
            x=alt.X(
                f"{IntervalMetricReport.metric}:N",
                title=None,
                sort=[member.value for member in IntervalMetricKind],
            ),
            xOffset=alt.XOffset(f"{IntervalMetricReport.kind}:N"),
            y=alt.Y(f"{IntervalMetricReport.value}:Q", title="Score"),
            color=alt.Color(
                f"{IntervalMetricReport.kind}:N",
                title="Interval",
                scale=alt.Scale(
                    domain=INTERVAL_COLOR_DOMAIN,
                    range=INTERVAL_COLOR_RANGE,
                ),
            ),
            tooltip=[
                f"{IntervalMetricReport.kind}:N",
                f"{IntervalMetricReport.metric}:N",
                f"{IntervalMetricReport.value}:Q",
            ],
        )
        .properties(
            width=CHART_WIDTH,
            height=CHART_HEIGHT,
            title="Interval quality metrics",
        )
    )
