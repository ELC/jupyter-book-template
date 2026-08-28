from typing import cast

import altair as alt
from pandera.typing import DataFrame

from analysis import ModelComparisonReport
from core import (
    IntervalKind,
    IntervalMetricKind,
    IntervalMetricReportByModel,
    MetricReportByModel,
    RegressionMetricKind,
)
from visualization._common import configure_altair

_REGRESSION_FACET_WIDTH = 420
_REGRESSION_FACET_HEIGHT = 240
_INTERVAL_FACET_WIDTH = 420
_INTERVAL_FACET_HEIGHT = 240


def _regression_metric_layers(
    metrics: DataFrame[MetricReportByModel],
) -> alt.LayerChart | alt.FacetChart:
    color = alt.Color(
        f"{MetricReportByModel.model}:N",
        legend=alt.Legend(title="Model"),
    )
    tooltip = [
        f"{MetricReportByModel.model}:N",
        f"{MetricReportByModel.metric}:N",
        f"{MetricReportByModel.lower}:Q",
        f"{MetricReportByModel.upper}:Q",
    ]
    base = alt.Chart(metrics).encode(
        y=alt.Y(f"{MetricReportByModel.model}:N", title=None),
        color=color,
        tooltip=tooltip,
    )
    range_rule = base.mark_rule(size=3).encode(
        x=alt.X(f"{MetricReportByModel.lower}:Q", title="Metric value"),
        x2=f"{MetricReportByModel.upper}:Q",
    )
    lower_tick = base.mark_tick(thickness=2, size=18).encode(
        x=alt.X(f"{MetricReportByModel.lower}:Q", title="Metric value"),
    )
    upper_tick = base.mark_tick(thickness=2, size=18).encode(
        x=alt.X(f"{MetricReportByModel.upper}:Q", title="Metric value"),
    )
    return alt.layer(range_rule, lower_tick, upper_tick)


def plot_regression_metrics(report: ModelComparisonReport) -> alt.FacetChart:
    configure_altair()
    layered = _regression_metric_layers(report.regression_metrics).properties(
        width=_REGRESSION_FACET_WIDTH,
        height=_REGRESSION_FACET_HEIGHT,
    )
    return (
        layered
        .facet(
            column=alt.Column(
                f"{MetricReportByModel.metric}:N",
                sort=[member.value for member in RegressionMetricKind],
                title=None,
            ),
        )
        .resolve_scale(x="independent")
        .properties(title="Regression metrics (bootstrap CI)")
    )


def _interval_metric_layers(
    metrics: DataFrame[IntervalMetricReportByModel],
) -> alt.LayerChart:
    color = alt.Color(
        f"{IntervalMetricReportByModel.model}:N",
        legend=alt.Legend(title="Model"),
    )
    tooltip = [
        f"{IntervalMetricReportByModel.model}:N",
        f"{IntervalMetricReportByModel.metric}:N",
        f"{IntervalMetricReportByModel.lower}:Q",
        f"{IntervalMetricReportByModel.upper}:Q",
    ]
    base = alt.Chart(metrics).encode(
        y=alt.Y(f"{IntervalMetricReportByModel.model}:N", title=None),
        color=color,
        tooltip=tooltip,
    )
    range_rule = base.mark_rule(size=3).encode(
        x=alt.X(f"{IntervalMetricReportByModel.lower}:Q", title="Score"),
        x2=f"{IntervalMetricReportByModel.upper}:Q",
    )
    lower_tick = base.mark_tick(thickness=2, size=18).encode(
        x=alt.X(f"{IntervalMetricReportByModel.lower}:Q", title="Score"),
    )
    upper_tick = base.mark_tick(thickness=2, size=18).encode(
        x=alt.X(f"{IntervalMetricReportByModel.upper}:Q", title="Score"),
    )
    return cast("alt.LayerChart", alt.layer(range_rule, lower_tick, upper_tick))


def _interval_kind_chart(
    metrics: DataFrame[IntervalMetricReportByModel],
    kind: IntervalKind,
) -> alt.FacetChart:
    layered = _interval_metric_layers(metrics).properties(
        width=_INTERVAL_FACET_WIDTH,
        height=_INTERVAL_FACET_HEIGHT,
    )
    return (
        layered
        .facet(
            column=alt.Column(
                f"{IntervalMetricReportByModel.metric}:N",
                sort=[member.value for member in IntervalMetricKind],
                title=None,
            ),
        )
        .resolve_scale(x="independent")
        .properties(title=f"{kind.value.title()} intervals (bootstrap CI)")
    )


def plot_interval_metrics(report: ModelComparisonReport) -> alt.VConcatChart:
    configure_altair()
    confidence_chart = _interval_kind_chart(report.confidence_metrics, IntervalKind.CONFIDENCE)
    prediction_chart = _interval_kind_chart(report.prediction_metrics, IntervalKind.PREDICTION)
    return alt.vconcat(confidence_chart, prediction_chart).properties(
        title="Interval quality metrics",
    )
