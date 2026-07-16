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
from scipy import stats


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

    `extra` is free-form metadata a specific fit function needs to support
    later operations on this result (e.g. `iv2sls_fit` stashes the
    exog/endog column layout here, since `iv2sls_predict` needs it and
    linearmodels' `predict()` API requires those passed as separate frames
    rather than a single merged one). Not part of the uniform contract
    other reference fits rely on -- treat contents as fit-function-specific.
    """

    coefficients: dict[str, float] = field(default_factory=dict)
    raw_model: Any = None
    ate_estimate: Optional[float] = None
    ate_std_error: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)


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


# ----------------------------------------------------------------------
# IV / 2SLS (optional dependency: linearmodels)
# ----------------------------------------------------------------------


def iv2sls_fit(
    data: pd.DataFrame,
    outcome: str,
    endogenous: list[str],
    instruments: list[str],
    exogenous: Optional[list[str]] = None,
) -> FitResult:
    """Fit two-stage least squares (2SLS) via `linearmodels.iv.IV2SLS`, to
    recover a coefficient on `endogenous` features despite their
    correlation with the outcome's error term, using `instruments` as the
    excluded instrument(s).

    Requires the optional `linearmodels` dependency
    (``pip install spuriosity[linearmodels]``); raises a clear
    `ImportError` with installation instructions if not available.

    `exogenous`, if given, are additional regressors included directly
    (not instrumented) -- e.g. control variables the researcher trusts are
    exogenous. An intercept is always included automatically under the
    column name `"const"` (linearmodels' own convention -- note this
    differs from the `"Intercept"` key patsy/statsmodels use elsewhere in
    `spuriosity.reference`; the two are not currently reconciled
    automatically when diffing against `GroundTruth.true_coefficients`).
    """
    try:
        from linearmodels.iv import IV2SLS
    except ImportError as e:
        raise ImportError(
            "iv2sls_fit requires the optional 'linearmodels' dependency. "
            "Install it with: pip install spuriosity[linearmodels]"
        ) from e

    exogenous = exogenous or []
    df = data.copy()
    df["const"] = 1.0
    exog_cols = ["const"] + exogenous

    model = IV2SLS(
        dependent=df[outcome],
        exog=df[exog_cols],
        endog=df[endogenous],
        instruments=df[instruments],
    )
    fitted = model.fit()
    coefficients = fitted.params.to_dict()
    # Stash the column layout needed to call .predict() correctly later --
    # linearmodels requires exog/endog passed as separate frames, not a
    # single merged one, so iv2sls_predict needs to know the split.
    return FitResult(
        coefficients=coefficients,
        raw_model=fitted,
        extra={"exog_cols": exog_cols, "endog_cols": list(endogenous)},
    )


def iv2sls_predict(fit_result: FitResult, data: pd.DataFrame) -> np.ndarray:
    """Predict fitted values using the 2SLS model's estimated coefficients
    applied to `data`. Requires `fit_result` to have been produced by
    `iv2sls_fit` (uses column layout stashed in `.extra`, since
    linearmodels' `.predict()` needs exog/endog passed as separate
    frames)."""
    model = fit_result.raw_model
    exog_cols = fit_result.extra.get("exog_cols")
    endog_cols = fit_result.extra.get("endog_cols")
    if exog_cols is None or endog_cols is None:
        raise ValueError(
            "iv2sls_predict requires a FitResult produced by iv2sls_fit "
            "(missing column layout metadata in .extra)."
        )
    df = data.copy()
    df["const"] = 1.0
    predictions = model.predict(exog=df[exog_cols], endog=df[endog_cols])
    return np.asarray(predictions.to_numpy().ravel())


def first_stage_f_stat(data: pd.DataFrame, endogenous: str, instruments: list[str]) -> float:
    """Compute the first-stage F-statistic (the standard weak-instrument
    diagnostic) for regressing `endogenous` on `instruments` plus an
    intercept, via statsmodels OLS. Values below ~10 (the classic
    Stock-Yogo rule of thumb) indicate a weak instrument on this sample.

    This is a convenience wrapper around a plain OLS F-test -- it does not
    require `linearmodels`, only `statsmodels` (already a core dependency),
    so it can be used to diagnose instrument strength even without fitting
    a full 2SLS model.
    """
    df = data.copy()
    formula = f"{endogenous} ~ " + " + ".join(instruments)
    model = smf.ols(formula, data=df).fit()
    f_stat: float = float(model.fvalue)
    return f_stat


# ----------------------------------------------------------------------
# Panel Fixed Effects / Random Effects (optional dependency: linearmodels)
# ----------------------------------------------------------------------


def _require_linearmodels_panel():
    try:
        from linearmodels.panel import PanelOLS, RandomEffects
    except ImportError as e:
        raise ImportError(
            "panel_fe_fit/panel_re_fit require the optional 'linearmodels' dependency. "
            "Install it with: pip install spuriosity[linearmodels]"
        ) from e
    return PanelOLS, RandomEffects


def panel_fe_fit(
    data: pd.DataFrame,
    outcome: str,
    features: list[str],
    entity_col: str = "entity_id",
    period_col: str = "period",
) -> FitResult:
    """Fit a panel fixed-effects (within) estimator via
    `linearmodels.panel.PanelOLS(entity_effects=True)`, controlling for
    all time-invariant entity-level confounders (observed or not) by
    demeaning within each entity.

    Requires the optional `linearmodels` dependency
    (``pip install spuriosity[linearmodels]``).

    Note: FE cannot estimate a coefficient on any time-invariant regressor
    (it gets demeaned away entirely) and has no intercept term -- unlike
    `ols_fit`, `.coefficients` will not contain an `"Intercept"` key.
    `.extra["entity_effects"]` stores the fitted entity effects
    (`fitted.estimated_effects`) for further inspection if needed.
    """
    PanelOLS, _ = _require_linearmodels_panel()
    df = data.set_index([entity_col, period_col])
    model = PanelOLS(df[outcome], df[features], entity_effects=True)
    fitted = model.fit()
    coefficients = fitted.params.to_dict()
    return FitResult(
        coefficients=coefficients,
        raw_model=fitted,
        extra={"entity_col": entity_col, "period_col": period_col, "features": list(features)},
    )


def panel_fe_predict(fit_result: FitResult, data: pd.DataFrame) -> np.ndarray:
    """Predict fitted values from a `panel_fe_fit` result. `data` must
    have the same panel structure (entity/period columns) as the data
    used to fit."""
    entity_col = fit_result.extra["entity_col"]
    period_col = fit_result.extra["period_col"]
    features = fit_result.extra["features"]
    df = data.set_index([entity_col, period_col])
    predictions = fit_result.raw_model.predict(df[features])
    return np.asarray(predictions.to_numpy().ravel())


def panel_re_fit(
    data: pd.DataFrame,
    outcome: str,
    features: list[str],
    entity_col: str = "entity_id",
    period_col: str = "period",
) -> FitResult:
    """Fit a panel random-effects (GLS) estimator via
    `linearmodels.panel.RandomEffects`, treating entity effects as a
    random draw uncorrelated with the regressors. More efficient than FE
    when that assumption holds, but biased if entity effects actually
    correlate with the regressors -- see `hausman_test` for the standard
    diagnostic that checks this assumption.

    Requires the optional `linearmodels` dependency
    (``pip install spuriosity[linearmodels]``). Unlike `panel_fe_fit`, RE
    does include an intercept, reported under `linearmodels`' own
    `"const"` naming (not patsy's `"Intercept"` -- see the note on this
    same naming mismatch in `iv2sls_fit`'s docstring).
    """
    _, RandomEffects = _require_linearmodels_panel()
    df = data.set_index([entity_col, period_col])
    exog = df[features].copy()
    exog["const"] = 1.0
    model = RandomEffects(df[outcome], exog)
    fitted = model.fit()
    coefficients = fitted.params.to_dict()
    return FitResult(
        coefficients=coefficients,
        raw_model=fitted,
        extra={"entity_col": entity_col, "period_col": period_col, "features": list(features)},
    )


def panel_re_predict(fit_result: FitResult, data: pd.DataFrame) -> np.ndarray:
    """Predict fitted values from a `panel_re_fit` result."""
    entity_col = fit_result.extra["entity_col"]
    period_col = fit_result.extra["period_col"]
    features = fit_result.extra["features"]
    df = data.set_index([entity_col, period_col])
    exog = df[features].copy()
    exog["const"] = 1.0
    predictions = fit_result.raw_model.predict(exog)
    return np.asarray(predictions.to_numpy().ravel())


def hausman_test(fe_result: FitResult, re_result: FitResult) -> dict[str, float]:
    """Hausman specification test comparing a fixed-effects and
    random-effects fit on the same data: tests the null that entity
    effects are uncorrelated with the regressors (i.e. that RE is valid
    and more efficient than FE). A small p-value rejects this null,
    indicating RE is biased and FE should be preferred.

    Compares only the coefficients shared between both fits (RE's
    intercept has no FE counterpart and is excluded automatically).

    Uses eigenvalue-clipped pseudo-inversion of the covariance difference
    matrix rather than a naive `np.linalg.inv`: the classical Hausman
    formula assumes `Var(FE) - Var(RE)` is positive semi-definite (since
    RE is asymptotically efficient under its null), but finite-sample
    covariance estimates can violate this slightly due to estimation
    noise, producing a technically-undefined or nonsensical chi-squared
    statistic under a naive matrix inverse. Clipping small/negative
    eigenvalues to a small positive floor is the standard practical fix
    used by applied econometrics software; verified against both a case
    where RE is genuinely biased (correctly rejected, p near 0) and a
    case where RE is valid (correctly not rejected, p large).

    Returns a dict with keys `"chi2"`, `"dof"`, `"p_value"`.
    """
    shared = [p for p in fe_result.coefficients if p in re_result.coefficients]
    if not shared:
        raise ValueError(
            "hausman_test found no coefficients shared between the FE and RE results "
            "to compare; were both fit on the same feature set?"
        )

    b_fe = np.array([fe_result.coefficients[k] for k in shared])
    b_re = np.array([re_result.coefficients[k] for k in shared])
    b_diff = b_fe - b_re

    fe_cov = fe_result.raw_model.cov.loc[shared, shared].to_numpy()
    re_cov = re_result.raw_model.cov.loc[shared, shared].to_numpy()
    v_diff = fe_cov - re_cov

    eigvals, eigvecs = np.linalg.eigh(v_diff)
    eigvals_clipped = np.clip(eigvals, a_min=1e-10, a_max=None)
    v_diff_pinv = eigvecs @ np.diag(1.0 / eigvals_clipped) @ eigvecs.T

    chi2 = float(b_diff.T @ v_diff_pinv @ b_diff)
    dof = len(shared)
    p_value = float(1 - stats.chi2.cdf(chi2, df=dof))

    return {"chi2": chi2, "dof": float(dof), "p_value": p_value}
