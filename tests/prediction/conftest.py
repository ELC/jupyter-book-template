import pytest

from core import SplitKind


@pytest.fixture(
    params=[pytest.param(split_kind, id=split_kind.value) for split_kind in SplitKind],
)
def selected_predict_split(request: pytest.FixtureRequest) -> SplitKind:
    return request.param
