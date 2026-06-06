"""
AnalyticsEngine — Dual-engine anomaly detection + time-series forecasting.

Anomaly Detection:
  - IsolationForest (multi-dimensional, sklearn)
  - IQR Z-score per numeric column
  - Combined flags with human-readable reasons

Forecasting:
  - Prophet (primary) with 30-day horizon + CI bands
  - statsmodels ARIMA (automatic fallback if Prophet unavailable)
  - Minimum 30 data-points required before running
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ── Prophet: graceful import ──────────────────────────────────────────────────
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# ── statsmodels ARIMA: fallback ───────────────────────────────────────────────
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


MIN_ROWS_FORECAST = 30
ISOLATION_CONTAMINATION = 0.05
IQR_ZSCORE_THRESHOLD = 3.0


@dataclass
class ForecastResult:
    method: str                     # "prophet" | "ets" | "unavailable"
    forecast_df: Optional[pd.DataFrame]  # columns: ds, yhat, yhat_lower, yhat_upper
    message: str


class AnalyticsEngine:
    """Wraps anomaly detection and forecasting in zero-crash protocols."""

    # ── Anomaly Detection ─────────────────────────────────────────────────────

    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run IsolationForest + IQR dual-engine detection.
        Returns df with added columns:
          is_anomaly (bool), anomaly_score (float), anomaly_reason (str)
        """
        result = df.copy()
        result["is_anomaly"] = False
        result["anomaly_score"] = 0.0
        result["anomaly_reason"] = ""

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not num_cols or len(df) < 10:
            result["anomaly_reason"] = "Insufficient data for anomaly detection."
            return result

        # ── IsolationForest ──────────────────────────────────────────────
        iso_flags = self._isolation_forest(df, num_cols)

        # ── IQR Z-score ──────────────────────────────────────────────────
        iqr_flags, iqr_reasons = self._iqr_zscore(df, num_cols)

        # ── Combine ───────────────────────────────────────────────────────
        combined = iso_flags | iqr_flags
        result["is_anomaly"] = combined

        reasons = []
        for i in range(len(df)):
            r = []
            if iso_flags.iloc[i]:
                r.append("Multi-dim outlier (IsolationForest)")
            if iqr_flags.iloc[i] and iqr_reasons[i]:
                r.append(f"Univariate spike: {iqr_reasons[i]}")
            reasons.append("; ".join(r) if r else "")
        result["anomaly_reason"] = reasons

        # Score: IsolationForest decision function (higher = more normal)
        try:
            scaler = StandardScaler()
            X = scaler.fit_transform(df[num_cols].fillna(0))
            iso = IsolationForest(
                contamination=ISOLATION_CONTAMINATION,
                random_state=42,
                n_jobs=-1,
            )
            iso.fit(X)
            scores = iso.decision_function(X)
            # Invert so higher score = more anomalous
            result["anomaly_score"] = np.round(-scores, 4)
        except Exception:
            pass

        return result

    def _isolation_forest(
        self, df: pd.DataFrame, num_cols: list[str]
    ) -> pd.Series:
        try:
            scaler = StandardScaler()
            X = scaler.fit_transform(df[num_cols].fillna(df[num_cols].median()))
            iso = IsolationForest(
                contamination=ISOLATION_CONTAMINATION,
                random_state=42,
                n_jobs=-1,
            )
            preds = iso.fit_predict(X)
            return pd.Series(preds == -1, index=df.index)
        except Exception:
            return pd.Series(False, index=df.index)

    def _iqr_zscore(
        self, df: pd.DataFrame, num_cols: list[str]
    ) -> tuple[pd.Series, list[str]]:
        any_flag = pd.Series(False, index=df.index)
        per_row_reasons: list[list[str]] = [[] for _ in range(len(df))]

        for col in num_cols:
            series = df[col].dropna()
            if len(series) < 10:
                continue
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            z_scores = (df[col] - series.median()).abs() / (IQR * 0.7413)
            outlier_mask = z_scores > IQR_ZSCORE_THRESHOLD
            any_flag = any_flag | outlier_mask.fillna(False)
            for i in df.index[outlier_mask.fillna(False)]:
                val = df.loc[i, col]
                per_row_reasons[i].append(
                    f"{col}={val:.3g} (z={z_scores.loc[i]:.1f})"
                )

        reason_strs = [", ".join(r) for r in per_row_reasons]
        return any_flag, reason_strs

    # ── Forecasting ───────────────────────────────────────────────────────────

    def forecast(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        horizon_days: int = 30,
    ) -> ForecastResult:
        """
        Run 30-day forecast.  Tries Prophet first, falls back to ETS.
        Returns ForecastResult with a standardized DataFrame:
          ds, yhat, yhat_lower, yhat_upper
        """
        # Validate inputs
        try:
            ts = df[[date_col, value_col]].copy()
            ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
            ts = ts.dropna(subset=[date_col, value_col])
            ts = ts.sort_values(date_col)
            ts[value_col] = pd.to_numeric(ts[value_col], errors="coerce")
            ts = ts.dropna(subset=[value_col])
        except Exception as e:
            return ForecastResult("unavailable", None, f"Data prep error: {e}")

        if len(ts) < MIN_ROWS_FORECAST:
            return ForecastResult(
                "unavailable",
                None,
                f"Need ≥ {MIN_ROWS_FORECAST} valid data points for forecasting "
                f"(found {len(ts)}). Please select a denser time series.",
            )

        # Try Prophet
        if PROPHET_AVAILABLE:
            result = self._prophet_forecast(ts, date_col, value_col, horizon_days)
            if result.forecast_df is not None:
                return result

        # Fallback: ETS (Holt-Winters)
        if STATSMODELS_AVAILABLE:
            return self._ets_forecast(ts, date_col, value_col, horizon_days)

        return ForecastResult(
            "unavailable",
            None,
            "No forecasting library available. Install `prophet` or `statsmodels`.",
        )

    def _prophet_forecast(
        self,
        ts: pd.DataFrame,
        date_col: str,
        value_col: str,
        horizon: int,
    ) -> ForecastResult:
        try:
            prophet_df = ts.rename(columns={date_col: "ds", value_col: "y"})[["ds", "y"]]
            # Aggregate by day to avoid duplicate-timestamp errors
            prophet_df = (
                prophet_df.groupby("ds", as_index=False)["y"]
                .mean()
                .sort_values("ds")
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = Prophet(
                    daily_seasonality=False,
                    weekly_seasonality=True,
                    yearly_seasonality=True,
                    interval_width=0.95,
                )
                m.fit(prophet_df)
                future = m.make_future_dataframe(periods=horizon)
                forecast = m.predict(future)

            result_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
            result_df.columns = ["ds", "yhat", "yhat_lower", "yhat_upper"]
            return ForecastResult("prophet", result_df, "Prophet forecast (95% CI).")
        except Exception as e:
            return ForecastResult("prophet_failed", None, f"Prophet failed: {e}")

    def _ets_forecast(
        self,
        ts: pd.DataFrame,
        date_col: str,
        value_col: str,
        horizon: int,
    ) -> ForecastResult:
        try:
            daily = (
                ts.set_index(date_col)[value_col]
                .resample("D")
                .mean()
                .interpolate()
            )
            n = len(daily)
            seasonal_periods = min(7, n // 2)
            trend_type = "add" if n > seasonal_periods * 2 else None
            seasonal_type = "add" if seasonal_periods > 1 and n > seasonal_periods * 2 else None

            model = ExponentialSmoothing(
                daily,
                trend=trend_type,
                seasonal=seasonal_type,
                seasonal_periods=seasonal_periods if seasonal_type else None,
                initialization_method="estimated",
            ).fit(optimized=True)

            forecast_vals = model.forecast(horizon)
            last_date = daily.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D"
            )

            # Compute simple confidence interval (±1.96 * residual std)
            resid_std = np.std(model.resid) if hasattr(model, "resid") else forecast_vals.std()

            result_df = pd.DataFrame({
                "ds": list(daily.index) + list(future_dates),
                "yhat": list(daily.values) + list(forecast_vals.values),
                "yhat_lower": list(daily.values) + list(forecast_vals.values - 1.96 * resid_std),
                "yhat_upper": list(daily.values) + list(forecast_vals.values + 1.96 * resid_std),
            })
            return ForecastResult(
                "ets",
                result_df,
                "Exponential Smoothing forecast (95% CI). Prophet not installed.",
            )
        except Exception as e:
            return ForecastResult("unavailable", None, f"ETS forecast failed: {e}")
