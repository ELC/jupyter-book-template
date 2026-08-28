import pandas as pd
from mapie.conformity_scores import BaseRegressionScore, ResidualNormalisedScore
from mapie.regression import SplitConformalRegressor
from pandera.typing import DataFrame
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from core import ConformityScoreKind, PredictionInterval, Settings, SplitDatasetBase, SplitKind, prepare_split
from prediction.regression import FEATURES_STEP, REGRESSOR_STEP


def _residual_estimator(pipeline: Pipeline) -> Pipeline:
    # The residual model regresses log|y - y_hat| on the same expanded basis
    # the predictor sees (poly + Fourier), so it can capture heteroscedasticity
    # that is nonlinear in the raw input. A LinearRegression head is enough
    # because the basis already encodes the nonlinearity.
    features = clone(pipeline.named_steps[FEATURES_STEP])
    return Pipeline(
        steps=[
            (FEATURES_STEP, features),
            (REGRESSOR_STEP, LinearRegression()),
        ],
    )


def build_conformity_score(
    pipeline: Pipeline,
    settings: Settings,
) -> str | BaseRegressionScore:
    if settings.conformity_score is ConformityScoreKind.RESIDUAL_NORMALIZED:
        return ResidualNormalisedScore(
            residual_estimator=_residual_estimator(pipeline),
            prefit=False,
            random_state=settings.seed,
        )
    return settings.conformity_score.value


def fit_conformal(
    data: DataFrame[SplitDatasetBase],
    pipeline: Pipeline,
    settings: Settings,
) -> SplitConformalRegressor:
    estimator = clone(pipeline)
    train = prepare_split(data, SplitKind.TRAINING)
    calib = prepare_split(data, SplitKind.CALIBRATION)
    conformal = SplitConformalRegressor(
        estimator=estimator,
        confidence_level=settings.confidence_level,
        conformity_score=build_conformity_score(pipeline, settings),
        prefit=False,
    )
    conformal.fit(train.x, train.y)
    conformal.conformalize(calib.x, calib.y)
    return conformal


def conformal_intervals(
    model: SplitConformalRegressor,
    data: DataFrame[SplitDatasetBase],
) -> DataFrame[PredictionInterval]:
    eval_split = prepare_split(data, SplitKind.EVALUATION)
    _, intervals = model.predict_interval(eval_split.x)
    lower, upper = intervals[..., 0].T
    return pd.DataFrame(
        {
            PredictionInterval.x: eval_split.x[:, 0],
            PredictionInterval.lower: lower,
            PredictionInterval.upper: upper,
        },
    ).pipe(DataFrame[PredictionInterval])
