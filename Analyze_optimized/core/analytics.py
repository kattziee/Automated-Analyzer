"""
analytics.py — Dual-engine anomaly detection + trend/pattern discovery +
time-series forecasting.

Anomaly Detection:
  - IsolationForest (multivariate, sklearn)
  - IQR Z-score (univariate, per numeric column)
  - Combined boolean flag with human-readable reasons

Forecasting:
  - statsmodels Exponential Smoothing (Holt-Winters), zero extra dependencies
  - Minimum row count enforced before running
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import THRESH
from utils.helpers import get_numeric_cols, maybe_sample


def compare_datasets(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    """Return a lightweight shape and schema comparison summary for two datasets."""
    return {
        "left_rows": int(len(left)),
        "right_rows": int(len(right)),
        "left_cols": int(len(left.columns)),
        "right_cols": int(len(right.columns)),
        "shared_columns": sorted(set(left.columns) & set(right.columns)),
        "left_only_columns": sorted(set(left.columns) - set(right.columns)),
        "right_only_columns": sorted(set(right.columns) - set(left.columns)),
        "row_delta": int(len(right) - len(left)),
    }

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

MIN_ROWS_FORECAST = 14


@dataclass
class ForecastResult:
    method: str
    forecast_df: Optional[pd.DataFrame]
    message: str
    seasonal_df: Optional[pd.DataFrame] = None


class AnalyticsEngine:
    """Anomaly detection, trend analysis, and forecasting — never raises."""

    # ── Anomaly Detection ────────────────────────────────────────────────

    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["is_anomaly"] = False
        result["anomaly_score"] = 0.0
        result["anomaly_reason"] = ""

        num_cols = get_numeric_cols(df)
        if not num_cols or len(df) < THRESH.min_rows_anomaly:
            result["anomaly_reason"] = "Insufficient data for anomaly detection."
            return result

        iso_flags = self._isolation_forest(df, num_cols)
        iqr_flags, iqr_reasons = self._iqr_zscore(df, num_cols)
        combined = iso_flags | iqr_flags
        result["is_anomaly"] = combined

        reasons = []
        for i in range(len(df)):
            r = []
            if iso_flags.iloc[i]:
                r.append("Multi-dimensional outlier (IsolationForest)")
            if iqr_flags.iloc[i] and iqr_reasons[i]:
                r.append(f"Univariate spike: {iqr_reasons[i]}")
            reasons.append("; ".join(r) if r else "")
        result["anomaly_reason"] = reasons

        try:
            X, _ = self._prepare_anomaly_features(df, num_cols)
            iso = IsolationForest(contamination=THRESH.isolation_contamination,
                                   random_state=THRESH.random_state, n_jobs=-1)
            iso.fit(X)
            result["anomaly_score"] = np.round(-iso.decision_function(X), 4)
        except Exception:
            pass

        return result

    def _prepare_anomaly_features(self, df: pd.DataFrame, num_cols: list[str]) -> tuple[np.ndarray, StandardScaler]:
        scaled = StandardScaler()
        X = scaled.fit_transform(df[num_cols].fillna(df[num_cols].median()))
        return X, scaled

    def _isolation_forest(self, df: pd.DataFrame, num_cols: list[str]) -> pd.Series:
        try:
            X, _ = self._prepare_anomaly_features(df, num_cols)
            iso = IsolationForest(contamination=THRESH.isolation_contamination,
                                   random_state=THRESH.random_state, n_jobs=-1)
            preds = iso.fit_predict(X)
            return pd.Series(preds == -1, index=df.index)
        except Exception:
            return pd.Series(False, index=df.index)

    def _iqr_zscore(self, df: pd.DataFrame, num_cols: list[str]) -> tuple[pd.Series, list[str]]:
        any_flag = pd.Series(False, index=df.index)
        per_row: list[list[str]] = [[] for _ in range(len(df))]
        pos_map = {idx: i for i, idx in enumerate(df.index)}

        for col in num_cols:
            series = df[col].dropna()
            if len(series) < THRESH.min_rows_anomaly:
                continue
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            z = (df[col] - series.median()).abs() / (IQR * 0.7413)
            mask = z > THRESH.zscore_threshold
            any_flag = any_flag | mask.fillna(False)
            for idx in df.index[mask.fillna(False)]:
                val = df.loc[idx, col]
                per_row[pos_map[idx]].append(f"{col}={val:.3g} (z={z.loc[idx]:.1f})")

        return any_flag, [", ".join(r) for r in per_row]

    def outlier_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-column IQR outlier counts — cheap, always available."""
        rows = []
        for col in get_numeric_cols(df):
            series = df[col].dropna()
            if len(series) < 4:
                continue
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR = Q3 - Q1
            lo, hi = Q1 - THRESH.iqr_multiplier * IQR, Q3 + THRESH.iqr_multiplier * IQR
            n_out = int(((series < lo) | (series > hi)).sum())
            rows.append({"column": col, "outliers": n_out,
                         "pct": round(100 * n_out / len(series), 2),
                         "lower_bound": round(float(lo), 3), "upper_bound": round(float(hi), 3)})
        return pd.DataFrame(rows).sort_values("outliers", ascending=False) if rows else pd.DataFrame()

    # ── Trend / pattern discovery ────────────────────────────────────────

    def trend_direction(self, series: pd.Series) -> dict:
        """Simple linear-trend slope sign + magnitude via OLS on index."""
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if len(clean) < 5:
            return {"trend": "insufficient_data"}
        x = np.arange(len(clean))
        try:
            slope, intercept = np.polyfit(x, clean.values, 1)
        except Exception:
            return {"trend": "unknown"}
        pct_change = (slope * len(clean)) / abs(clean.mean()) * 100 if clean.mean() != 0 else 0
        direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
        return {"trend": direction, "slope": round(float(slope), 4), "pct_change_est": round(float(pct_change), 2)}

    def time_series_summary(self, df: pd.DataFrame, date_col: str, value_col: str) -> dict:
        """Return a compact summary for time-series data using a sampled frame when needed."""
        sample = maybe_sample(df, n=THRESH.sample_rows_for_heavy_ops)
        ts = sample[[date_col, value_col]].copy()
        ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
        ts = ts.dropna(subset=[date_col, value_col]).sort_values(date_col)
        ts[value_col] = pd.to_numeric(ts[value_col], errors="coerce")
        ts = ts.dropna(subset=[value_col])
        if ts.empty:
            return {"points": 0, "trend": "insufficient_data"}
        return {
            "points": int(len(ts)),
            "trend": self.trend_direction(ts[value_col]),
            "start": ts[date_col].min().date().isoformat(),
            "end": ts[date_col].max().date().isoformat(),
        }

    # ── Forecasting ───────────────────────────────────────────────────────

    def forecast(self, df: pd.DataFrame, date_col: str, value_col: str, horizon_days: int = 30) -> ForecastResult:
        if not STATSMODELS_AVAILABLE:
            return ForecastResult("unavailable", None, "statsmodels is not installed.")
        try:
            ts = df[[date_col, value_col]].copy()
            ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
            ts = ts.dropna(subset=[date_col, value_col]).sort_values(date_col)
            ts[value_col] = pd.to_numeric(ts[value_col], errors="coerce")
            ts = ts.dropna(subset=[value_col])
        except Exception as e:
            return ForecastResult("unavailable", None, f"Data preparation error: {e}")

        if len(ts) < MIN_ROWS_FORECAST:
            return ForecastResult("unavailable", None,
                                   f"Need ≥ {MIN_ROWS_FORECAST} valid points (found {len(ts)}).")

        try:
            daily = ts.set_index(date_col)[value_col].resample("D").mean().interpolate()
            n = len(daily)
            seasonal_periods = min(7, n // 2) if n // 2 >= 2 else None
            trend_type = "add" if n > (seasonal_periods or 2) * 2 else None
            seasonal_type = "add" if seasonal_periods and seasonal_periods > 1 and n > seasonal_periods * 2 else None

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ExponentialSmoothing(
                    daily, trend=trend_type, seasonal=seasonal_type,
                    seasonal_periods=seasonal_periods if seasonal_type else None,
                    initialization_method="estimated",
                ).fit(optimized=True)

            fvals = model.forecast(horizon_days)
            future_dates = pd.date_range(start=daily.index[-1] + pd.Timedelta(days=1),
                                          periods=horizon_days, freq="D")
            resid_std = np.std(model.resid) if hasattr(model, "resid") else fvals.std()

            result_df = pd.DataFrame({
                "ds": list(daily.index) + list(future_dates),
                "yhat": list(daily.values) + list(fvals.values),
                "yhat_lower": list(daily.values) + list(fvals.values - 1.96 * resid_std),
                "yhat_upper": list(daily.values) + list(fvals.values + 1.96 * resid_std),
            })

            seasonal_df = None
            if seasonal_type and n >= 2 * seasonal_periods:
                try:
                    decomp = seasonal_decompose(daily, period=seasonal_periods, model="additive", extrapolate_trend="freq")
                    seasonal_df = pd.DataFrame({
                        "ds": daily.index, "trend": decomp.trend, "seasonal": decomp.seasonal, "resid": decomp.resid,
                    })
                except Exception:
                    pass

            return ForecastResult("ets", result_df,
                                   "Exponential Smoothing forecast (95% CI, approximate).", seasonal_df)
        except Exception as e:
            return ForecastResult("unavailable", None, f"Forecast failed: {e}")
