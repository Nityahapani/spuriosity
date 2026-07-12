"""
Batteries-included fit/predict functions for use with StressTest and
compare_models, so a first stress test can run in ~5 lines without the user
writing any model-fitting code themselves.

Uniform contract: every `*_fit(data, ...) -> FitResult` returns a `FitResult`
dataclass exposing `.coefficients` (a dict matching GroundTruth.true_coefficients
naming, where applicable) and `.raw_model` (the underlying fitted object, for
anything not covered by the uniform interface). Every `*_predict(fit_result,
data) -> np.ndarray` returns predictions on new data.

v1 references: OLS (statsmodels), sklearn LinearRegression, simple 2x2 DiD,
DoubleML-based PLR (ATE) estimator, logit (for binary outcomes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


@dataclass
class FitResult:
    """Uniform wrapper around a fitted reference model.

    `coefficients` uses the same naming convention as
    `GroundTruth.true_coefficients` where applicable (e.g. patsy's
    `"Intercept"` naming), so a StressTest can directly diff the two
    dictionaries. Models that don't naturally produce named coefficients
    (e.g. a black-box ML estimator used inside DoubleML) may leave
    `coefficients` sparse or empty and rely on `.raw_model` /
    `.ate_estimate` instead.
    """

    coefficients: dict[str, float] = field(default_factory=dict)
    raw_model: Any = None
    ate_estimate: Optional[float] = None
    ate_std_error: Optional[float] = None


# ----------------------------------------------------------------------
# OLS (statsmodels)
# ----------------------------------------------------------------------


def ols_fit(data: pd.DataFrame, formula: str) -> FitResult:
    """Fit OLS via statsmodels' formula API.

    `formula` follows the same patsy syntax as `PanelGenerator.set_outcome`
    (e.g. ``"y ~ x1 + x2 + treat"``), so its right-hand-side column naming
    (including the automatic ``"Intercept"``) lines up directly with
    `GroundTruth.true_coefficients`.
    """
    model = smf.ols(formula, data=data).fit()
    coefficients = model.params.to_dict()
    return FitResult(coefficients=coefficients, raw_model=model)


def ols_predict(fit_result: FitResult, data: pd.DataFrame) -> np.ndarray:
    predictions: np.ndarray = np.asarray(fit_result.raw_model.predict(data))
    return predictions


# ----------------------------------------------------------------------
# sklearn LinearRegression
# ----------------------------------------------------------------------


def sklearn_lr_fit(data: pd.DataFrame, features: list[str], target: str) -> FitResult:
    """Fit sklearn's LinearRegression on the given feature columns.

    Coefficients are reported under each feature's own column name, plus
    ``"Intercept"`` for the intercept term, matching the patsy/statsmodels
    naming convention used elsewhere in spuriosity.
    """
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(data[features], data[target])
    coefficients = {name: float(c) for name, c in zip(features, model.coef_)}
    coefficients["Intercept"] = float(model.intercept_)
    return FitResult(coefficients=coefficients, raw_model=model)


def sklearn_lr_predict(fit_result: FitResult, data: pd.DataFrame) -> np.ndarray:
    model = fit_result.raw_model
    features = [f for f in fit_result.coefficients if f != "Intercept"]
    predictions: np.ndarray = np.asarray(model.predict(data[features]))
    return predictions


# ----------------------------------------------------------------------
# Simple 2x2 Difference-in-Differences
# ----------------------------------------------------------------------


def did_fit(data: pd.DataFrame, outcome: str, treatment: str, period: str, post_period: int) -> FitResult:
    """Fit a standard 2x2 difference-in-differences specification via OLS:

        outcome ~ treatment * post

    where ``post = 1[period >= post_period]``. The DiD estimate of the
    treatment effect is the coefficient on the ``treatment:post``
    interaction term, reported under the key
    ``f"{treatment}:post"`` in `.coefficients`, and also directly available
    as `.ate_estimate`.
    """
    df = data.copy()
    df["post"] = (df[period] >= post_period).astype(int)
    formula = f"{outcome} ~ {treatment} * post"
    model = smf.ols(formula, data=df).fit()
    coefficients = model.params.to_dict()
    interaction_key = f"{treatment}:post"
    ate = coefficients.get(interaction_key)
    se = float(model.bse.get(interaction_key)) if interaction_key in model.bse else None
    return FitResult(coefficients=coefficients, raw_model=model, ate_estimate=ate, ate_std_error=se)


def did_predict(fit_result: FitResult, data: pd.DataFrame) -> np.ndarray:
    """Predict using the fitted DiD model. `data` must already contain a
    ``"post"`` column (as constructed by `did_fit`); if not present it is
    reconstructed assuming the same post-period cutoff is not available
    here, so callers should generally predict on the same `data` passed to
    `did_fit`, or add ``"post"`` themselves beforehand.
    """
    if "post" not in data.columns:
        raise ValueError(
            "did_predict requires a 'post' column on `data` (as constructed by did_fit); "
            "predicting on new data requires adding this column yourself first."
        )
    predictions: np.ndarray = np.asarray(fit_result.raw_model.predict(data))
    return predictions


# ----------------------------------------------------------------------
# DoubleML-based ATE estimator (optional dependency)
# ----------------------------------------------------------------------


def doubleml_fit(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    covariates: list[str],
    n_folds: int = 5,
) -> FitResult:
    """Fit a DoubleML partially linear regression (PLR) model to estimate
    the ATE of `treatment` on `outcome`, controlling for `covariates` via
    random forest nuisance models.

    Requires the optional `doubleml` dependency
    (``pip install spuriosity[doubleml]``); raises a clear `ImportError`
    with installation instructions if not available.
    """
    try:
        from doubleml import DoubleMLData, DoubleMLPLR
    except ImportError as e:
        raise ImportError(
            "doubleml_fit requires the optional 'doubleml' dependency. "
            "Install it with: pip install spuriosity[doubleml]"
        ) from e
    from sklearn.ensemble import RandomForestRegressor

    dml_data = DoubleMLData(data, y_col=outcome, d_cols=treatment, x_cols=covariates)
    ml_l = RandomForestRegressor(n_estimators=100, max_depth=5)
    ml_m = RandomForestRegressor(n_estimators=100, max_depth=5)
    model = DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=n_folds)
    model.fit()

    ate = float(model.coef[0])
    se = float(model.se[0])
    return FitResult(
        coefficients={treatment: ate}, raw_model=model, ate_estimate=ate, ate_std_error=se
    )


def doubleml_predict(fit_result: FitResult, data: pd.DataFrame) -> np.ndarray:
    """DoubleML PLR estimates an average treatment effect, not a
    row-level predictive model in the usual sense. This returns a
    constant-effect prediction (`ate_estimate` broadcast to every row),
    useful for comparing against a true CATE function in aggregate but not
    meaningful as a row-level prediction under real heterogeneity.
    """
    if fit_result.ate_estimate is None:
        raise ValueError("FitResult has no ate_estimate; was it produced by doubleml_fit?")
    return np.full(len(data), fit_result.ate_estimate, dtype=float)


# ----------------------------------------------------------------------
# Logit
# ----------------------------------------------------------------------


def logit_fit(data: pd.DataFrame, formula: str) -> FitResult:
    """Fit a logistic regression via statsmodels' formula API, for binary
    outcomes (e.g. modeling the selection mechanism itself, or a binary
    treatment/outcome DGP)."""
    model = smf.logit(formula, data=data).fit(disp=0)
    coefficients = model.params.to_dict()
    return FitResult(coefficients=coefficients, raw_model=model)


def logit_predict(fit_result: FitResult, data: pd.DataFrame) -> np.ndarray:
    """Returns predicted probabilities (not hard class labels)."""
    predictions: np.ndarray = np.asarray(fit_result.raw_model.predict(data))
    return predictions
