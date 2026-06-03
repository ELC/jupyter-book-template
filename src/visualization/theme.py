from collections.abc import Mapping

from core.schemas import IntervalKind

CHART_WIDTH = 640
CHART_HEIGHT = 400
SCATTER_OPACITY = 0.5
INTERVAL_BAND_OPACITY = 0.35

INTERVAL_COLORS: Mapping[IntervalKind, str] = {
    IntervalKind.CONFIDENCE: "#1f77b4",
    IntervalKind.PREDICTION: "#ff7f0e",
}

INTERVAL_COLOR_DOMAIN = [member.value for member in IntervalKind]
INTERVAL_COLOR_RANGE = [INTERVAL_COLORS[member] for member in IntervalKind]
