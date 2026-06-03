import altair as alt


def configure_altair() -> None:
    alt.data_transformers.disable_max_rows()
