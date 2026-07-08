"""
statistics_engine.py — Classical statistical analysis toolkit.

Named `statistics_engine` (not `statistics`) to avoid shadowing the Python
standard-library `statistics` module.

Covers: descriptive statistics, correlation (Pearson/Spearman) with
significance, normality tests (Shapiro/D'Agostino), t-test, one-way ANOVA,
chi-square test of independence, confidence intervals, and basic OLS
regression diagnostics.

Every function degrades gracefully (returns a message instead of raising)
so a single bad column never crashes the whole analysis page.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from config import THRESH
from utils.helpers import get_numeric_cols, get_categorical_cols, maybe_sample


@dataclass
class TestResult:
    test_name: str
    statistic: Optional[float]
    p_value: Optional[float]
    significant: Optional[bool]
    interpretation: str


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = get_numeric_cols(df)
    if not num_cols:
        return pd.DataFrame()
    desc = df[num_cols].describe().T
    desc["skew"] = df[num_cols].skew(numeric_only=True)
    desc["kurtosis"] = df[num_cols].kurtosis(numeric_only=True)
    desc["missing_pct"] = (df[num_cols].isna().mean() * 100).round(1)
    desc["iqr"] = (df[num_cols].quantile(0.75) - df[num_cols].quantile(0.25)).round(3)
    desc["mad"] = (df[num_cols].mad()).round(3)
    return desc.round(3)


def autocorrelation(series: pd.Series, lags: list[int] | tuple[int, ...] | None = None) -> pd.Series:
    """Return autocorrelation values for given lags, using Pearson correlation."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 3:
        return pd.Series(dtype=float)
    lags = list(lags or [1, 2, 3])
    values = []
    for lag in lags:
        if lag >= len(clean):
            values.append(np.nan)
            continue
        x = clean.iloc[:-lag]
        y = clean.iloc[lag:]
        values.append(float(x.corr(y)))
    return pd.Series(values, index=lags, dtype=float)


def normality_test(series: pd.Series) -> TestResult:
    clean = series.dropna()
    if len(clean) < 8:
        return TestResult("Shapiro-Wilk", None, None, None, "Insufficient data (need ≥ 8 values).")
    sample = clean.sample(min(len(clean), 5000), random_state=THRESH.random_state)
    try:
        stat, p = stats.shapiro(sample)
        sig = p < THRESH.normality_alpha
        interp = ("Data significantly deviates from normal distribution (p < 0.05)."
                   if sig else "No significant deviation from normality detected.")
        return TestResult("Shapiro-Wilk", round(float(stat), 4), round(float(p), 4), sig, interp)
    except Exception as e:
        return TestResult("Shapiro-Wilk", None, None, None, f"Test failed: {e}")


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    num_cols = get_numeric_cols(df)
    if len(num_cols) < 2:
        return pd.DataFrame()
    return df[num_cols].corr(method=method).round(3)


def significant_correlations(df: pd.DataFrame, method: str = "pearson") -> list[dict]:
    """Return pairs whose |r| exceeds the 'moderate' threshold, with p-values."""
    num_cols = get_numeric_cols(df)
    results = []
    for i, c1 in enumerate(num_cols):
        for c2 in num_cols[i + 1:]:
            pair = df[[c1, c2]].dropna()
            if len(pair) < 5:
                continue
            try:
                if method == "spearman":
                    r, p = stats.spearmanr(pair[c1], pair[c2])
                else:
                    r, p = stats.pearsonr(pair[c1], pair[c2])
            except Exception:
                continue
            if abs(r) >= THRESH.correlation_moderate:
                strength = "strong" if abs(r) >= THRESH.correlation_strong else "moderate"
                direction = "positive" if r > 0 else "negative"
                results.append({
                    "feature_1": c1, "feature_2": c2, "r": round(float(r), 3),
                    "p_value": round(float(p), 4), "strength": strength, "direction": direction,
                })
    return sorted(results, key=lambda d: abs(d["r"]), reverse=True)


def t_test(df: pd.DataFrame, numeric_col: str, group_col: str) -> TestResult:
    groups = df[[numeric_col, group_col]].dropna()
    levels = groups[group_col].unique()
    if len(levels) != 2:
        return TestResult("Independent t-test", None, None, None,
                           f"Grouping column must have exactly 2 levels (found {len(levels)}).")
    a = groups[groups[group_col] == levels[0]][numeric_col]
    b = groups[groups[group_col] == levels[1]][numeric_col]
    if len(a) < 2 or len(b) < 2:
        return TestResult("Independent t-test", None, None, None, "Insufficient samples in one group.")
    try:
        stat, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        sig = p < 0.05
        interp = (f"Mean {numeric_col} significantly differs between '{levels[0]}' and '{levels[1]}' (p < 0.05)."
                   if sig else f"No significant difference in {numeric_col} between groups.")
        return TestResult("Welch's t-test", round(float(stat), 4), round(float(p), 4), sig, interp)
    except Exception as e:
        return TestResult("Welch's t-test", None, None, None, f"Test failed: {e}")


def anova_test(df: pd.DataFrame, numeric_col: str, group_col: str) -> TestResult:
    groups = df[[numeric_col, group_col]].dropna()
    samples = [g[numeric_col].values for _, g in groups.groupby(group_col) if len(g) >= 2]
    if len(samples) < 2:
        return TestResult("One-way ANOVA", None, None, None, "Need ≥ 2 groups with ≥ 2 observations each.")
    try:
        stat, p = stats.f_oneway(*samples)
        sig = p < 0.05
        interp = (f"Mean {numeric_col} significantly differs across levels of {group_col} (p < 0.05)."
                   if sig else f"No significant difference in {numeric_col} across groups.")
        return TestResult("One-way ANOVA", round(float(stat), 4), round(float(p), 4), sig, interp)
    except Exception as e:
        return TestResult("One-way ANOVA", None, None, None, f"Test failed: {e}")


def chi_square_test(df: pd.DataFrame, col_a: str, col_b: str) -> TestResult:
    try:
        table = pd.crosstab(df[col_a], df[col_b])
        if table.shape[0] < 2 or table.shape[1] < 2:
            return TestResult("Chi-square", None, None, None, "Need ≥ 2 categories in each column.")
        stat, p, dof, _ = stats.chi2_contingency(table)
        sig = p < 0.05
        interp = (f"'{col_a}' and '{col_b}' appear to be associated (p < 0.05)."
                   if sig else f"No significant association found between '{col_a}' and '{col_b}'.")
        return TestResult("Chi-square", round(float(stat), 4), round(float(p), 4), sig, interp)
    except Exception as e:
        return TestResult("Chi-square", None, None, None, f"Test failed: {e}")


def confidence_interval(series: pd.Series, confidence: float = None) -> dict:
    confidence = confidence or THRESH.confidence_level
    clean = series.dropna()
    if len(clean) < 2:
        return {"mean": None, "lower": None, "upper": None}
    mean = clean.mean()
    sem = stats.sem(clean)
    margin = sem * stats.t.ppf((1 + confidence) / 2, len(clean) - 1)
    return {"mean": round(float(mean), 4), "lower": round(float(mean - margin), 4),
            "upper": round(float(mean + margin), 4), "confidence": confidence}


def regression_diagnostics(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    """Simple OLS diagnostics: slope, intercept, R², residual normality."""
    pair = df[[x_col, y_col]].dropna()
    if len(pair) < 5:
        return {"error": "Insufficient data for regression (need ≥ 5 rows)."}
    try:
        slope, intercept, r, p, se = stats.linregress(pair[x_col], pair[y_col])
        residuals = pair[y_col] - (slope * pair[x_col] + intercept)
        return {
            "slope": round(float(slope), 4), "intercept": round(float(intercept), 4),
            "r_squared": round(float(r ** 2), 4), "p_value": round(float(p), 4),
            "std_err": round(float(se), 4),
            "residual_mean": round(float(residuals.mean()), 4),
            "residual_std": round(float(residuals.std()), 4),
        }
    except Exception as e:
        return {"error": str(e)}
