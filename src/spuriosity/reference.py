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


# ----------------------------------------------------------------------
# Propensity Score Matching (optional dependency: scikit-learn)
# ----------------------------------------------------------------------


def psm_fit(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    covariates: list[str],
) -> FitResult:
    """Estimate the ATE of a binary `treatment` via nearest-neighbor
    propensity score matching: fit a logistic regression of `treatment` on
    `covariates` to estimate each unit's propensity score, then match each
    treated unit to its nearest control unit by propensity score
    (with replacement) and average the matched outcome differences.

    Requires the optional `sklearn` dependency
    (``pip install spuriosity[sklearn]``); raises a clear `ImportError`
    with installation instructions if not available.

    `.ate_estimate` is the matched-pairs average treatment effect on the
    treated (ATT, technically -- matching is only performed for treated
    units against controls, not the reverse). `.extra` contains:

    - `"propensity_scores"`: the fitted propensity score for every row in
      `data`, in the same row order.
    - `"common_support_fraction"`: the fraction of treated units whose
      propensity score falls within the control group's observed
      propensity score range (and vice versa isn't checked separately,
      since ATT only requires treated-side overlap). Low values (well
      below 1.0) indicate poor overlap between treated and control groups
      -- matches for units outside common support are extrapolating the
      propensity model rather than genuinely comparing similar units, and
      the ATT estimate should be treated with more caution the lower this
      fraction is.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as e:
        raise ImportError(
            "psm_fit requires the optional 'sklearn' dependency. "
            "Install it with: pip install spuriosity[sklearn]"
        ) from e
    from scipy.spatial import cKDTree

    df = data.reset_index(drop=True)
    n_treated = int(df[treatment].sum())
    n_control = len(df) - n_treated
    if n_treated == 0 or n_control == 0:
        raise ValueError(
            f"psm_fit requires both treated and control units; got "
            f"{n_treated} treated and {n_control} control."
        )

    ps_model = LogisticRegression()
    ps_model.fit(df[covariates], df[treatment])
    propensity_scores = ps_model.predict_proba(df[covariates])[:, 1]

    treated_mask = df[treatment].to_numpy().astype(bool)
    treated_ps = propensity_scores[treated_mask]
    control_ps = propensity_scores[~treated_mask]

    control_min, control_max = control_ps.min(), control_ps.max()
    within_support = (treated_ps >= control_min) & (treated_ps <= control_max)
    common_support_fraction = float(within_support.mean())

    control_outcomes = df.loc[~treated_mask, outcome].to_numpy()
    treated_outcomes = df.loc[treated_mask, outcome].to_numpy()

    tree = cKDTree(control_ps.reshape(-1, 1))
    _, matched_idx = tree.query(treated_ps.reshape(-1, 1), k=1)
    matched_control_outcomes = control_outcomes[matched_idx]

    matched_differences = treated_outcomes - matched_control_outcomes
    ate_estimate = float(matched_differences.mean())
    ate_std_error = float(matched_differences.std(ddof=1) / np.sqrt(len(matched_differences)))

    return FitResult(
        coefficients={treatment: ate_estimate},
        raw_model=ps_model,
        ate_estimate=ate_estimate,
        ate_std_error=ate_std_error,
        extra={
            "propensity_scores": propensity_scores,
            "common_support_fraction": common_support_fraction,
            "covariates": list(covariates),
        },
    )


def psm_predict(fit_result: FitResult, data: pd.DataFrame) -> np.ndarray:
    """PSM estimates an average treatment effect on the treated, not a
    row-level predictive model in the usual sense. This returns a
    constant-effect prediction (`ate_estimate` broadcast to every row),
    matching the convention already used by `doubleml_predict`."""
    if fit_result.ate_estimate is None:
        raise ValueError("FitResult has no ate_estimate; was it produced by psm_fit?")
    return np.full(len(data), fit_result.ate_estimate, dtype=float)


# ----------------------------------------------------------------------
# ML baseline: XGBoost (optional dependency)
# ----------------------------------------------------------------------


def xgboost_fit(
    data: pd.DataFrame,
    outcome: str,
    features: list[str],
    n_estimators: int = 100,
    max_depth: int = 4,
    **xgb_kwargs: object,
) -> FitResult:
    """Fit an XGBoost regressor as a pure predictive ML baseline, for the
    "does a flexible nonlinear model beat econometric assumptions" stress
    test -- e.g. comparing recovery/prediction quality against `ols_fit`
    on data generated with a nonlinear outcome (`set_outcome(fn=...)`) or
    unmodeled interactions.

    Requires the optional `xgboost` dependency
    (``pip install spuriosity[xgboost]``); raises a clear `ImportError`
    with installation instructions if not available.

    Unlike the econometric reference fits, XGBoost does not produce
    interpretable per-feature coefficients -- `.coefficients` is left
    empty (so `coef_rmse` is simply not computed for this model, the same
    "not applicable" convention used elsewhere; feature importances are
    available via `.extra["feature_importances"]` instead, which is a
    fundamentally different quantity from a linear coefficient and should
    not be compared against `GroundTruth.true_coefficients`).

    Extra `xgb_kwargs` are passed through to `xgboost.XGBRegressor`.
    """
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError(
            "xgboost_fit requires the optional 'xgboost' dependency. "
            "Install it with: pip install spuriosity[xgboost]"
        ) from e

    model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, **xgb_kwargs)
    model.fit(data[features], data[outcome])

    feature_importances = dict(zip(features, model.feature_importances_.tolist()))

    return FitResult(
        coefficients={},
        raw_model=model,
        extra={"features": list(features), "feature_importances": feature_importances},
    )


def xgboost_predict(fit_result: FitResult, data: pd.DataFrame) -> np.ndarray:
    features = fit_result.extra["features"]
    predictions: np.ndarray = np.asarray(fit_result.raw_model.predict(data[features]))
    return predictions


# ----------------------------------------------------------------------
# Causal Forest (optional dependency: econml)
# ----------------------------------------------------------------------


def causal_forest_fit(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    covariates: list[str],
    n_estimators: int = 200,
    discrete_treatment: bool = True,
    random_state: Optional[int] = None,
) -> FitResult:
    """Fit a causal forest via `econml.dml.CausalForestDML`, to estimate a
    full CATE function (not just the ATE) as a nonparametric function of
    `covariates` -- the natural counterpart to `spuriosity`'s `add_hte`
    for testing whether a flexible ML-based estimator recovers
    heterogeneous treatment effects, as opposed to `doubleml_fit`'s
    single average effect.

    Requires the optional `econml` dependency
    (``pip install spuriosity[econml]``); raises a clear `ImportError`
    with installation instructions if not available.

    Note on first-import cost: `econml` depends on `numba`, which performs
    a one-time LLVM/JIT compilation cache build on first import in a
    given environment (observed ~15-20s cold, ~2s warm in testing) --
    this is a one-time environment cost, not a per-call cost, but is
    worth knowing about if this is the first `causal_forest_fit` call in
    a fresh virtualenv/CI run.

    Nuisance models default to random forests, with `model_t` correctly
    specified as a *classifier* when `discrete_treatment=True` (the
    default) -- using a regressor there is a common mistake that `econml`
    only warns about rather than rejecting; `causal_forest_fit` avoids it
    by construction.

    `n_estimators` must be evenly divisible by 4 (`CausalForestDML`'s
    `subforest_size` default, not exposed as a separate parameter here);
    this is validated upfront with a clear error rather than letting
    `econml`'s own internal `ValueError` (raised deep inside its `fit()`
    call stack) surface directly.

    `.coefficients` is left empty (like `xgboost_fit` -- a causal forest
    has no single per-feature coefficient; its whole point is a
    covariate-varying effect surface). Use `.raw_model.effect(X)` directly
    for CATE predictions at arbitrary covariate points, or
    `causal_forest_predict` for the per-row CATE on `data` itself.
    `.extra["ate_estimate"]` reports the population-average effect (the
    mean of the per-row CATE predictions on the training data) for
    convenience when only the ATE is needed.
    """
    try:
        from econml.dml import CausalForestDML
    except ImportError as e:
        raise ImportError(
            "causal_forest_fit requires the optional 'econml' dependency. "
            "Install it with: pip install spuriosity[econml]"
        ) from e
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    # CausalForestDML requires n_estimators to be evenly divisible by its
    # subforest_size parameter (default 4, not exposed here since v1 of
    # this wrapper doesn't need to tune it) -- validated upfront with a
    # clear message, since econml's own internal error on this is a raw
    # ValueError from deep inside its fit() call stack that doesn't
    # explain the constraint clearly (encountered directly during
    # development: n_estimators=50 fails since 50 is not divisible by 4).
    _default_subforest_size = 4
    if n_estimators % _default_subforest_size != 0:
        raise ValueError(
            f"n_estimators ({n_estimators}) must be evenly divisible by CausalForestDML's "
            f"subforest_size (default {_default_subforest_size}). Try a multiple of "
            f"{_default_subforest_size}, e.g. {(n_estimators // _default_subforest_size + 1) * _default_subforest_size}."
        )

    model_t: object
    if discrete_treatment:
        model_t = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_state)
    else:
        model_t = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=random_state)

    est = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=random_state),
        model_t=model_t,
        discrete_treatment=discrete_treatment,
        n_estimators=n_estimators,
        random_state=random_state,
    )
    X = data[covariates].to_numpy()
    est.fit(Y=data[outcome].to_numpy(), T=data[treatment].to_numpy(), X=X)

    per_row_cate = np.asarray(est.effect(X))
    ate_estimate = float(per_row_cate.mean())

    return FitResult(
        coefficients={},
        raw_model=est,
        ate_estimate=ate_estimate,
        extra={"covariates": list(covariates), "ate_estimate": ate_estimate},
    )


def causal_forest_predict(fit_result: FitResult, data: pd.DataFrame) -> np.ndarray:
    """Returns the per-row CATE estimate (the treatment effect at each
    row's covariate values), NOT an outcome prediction -- unlike every
    other `*_predict` function in this module. This matches what a causal
    forest is actually for: estimating a heterogeneous effect surface,
    not predicting the outcome variable itself.
    """
    covariates = fit_result.extra["covariates"]
    X = data[covariates].to_numpy()
    effects: np.ndarray = np.asarray(fit_result.raw_model.effect(X))
    return effects
