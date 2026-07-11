"""
Batteries-included fit/predict functions for use with StressTest and
compare_models, so a first stress test can run in ~5 lines without the user
writing any model-fitting code themselves.

v1 targets: OLS (statsmodels), sklearn LinearRegression, simple 2x2 DiD,
basic DoubleML-style CATE estimator, logit (for selection-bias-aware
binary outcomes).
"""

from __future__ import annotations


def ols_fit(data, formula: str):
    raise NotImplementedError


def ols_predict(model, data):
    raise NotImplementedError


def sklearn_lr_fit(data, features: list[str], target: str):
    raise NotImplementedError


def sklearn_lr_predict(model, data):
    raise NotImplementedError


def did_fit(data, outcome: str, treatment: str, period: str, post_period: int):
    raise NotImplementedError


def did_predict(model, data):
    raise NotImplementedError


def doubleml_fit(data, outcome: str, treatment: str, covariates: list[str]):
    raise NotImplementedError


def doubleml_predict(model, data):
    raise NotImplementedError


def logit_fit(data, formula: str):
    raise NotImplementedError


def logit_predict(model, data):
    raise NotImplementedError
