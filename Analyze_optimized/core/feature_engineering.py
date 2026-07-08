"""Feature engineering utilities for richer analytics workflows."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from utils.helpers import get_numeric_cols


def engineer_features(
    df: pd.DataFrame,
    *,
    datetime_cols: Iterable[str] | None = None,
    numeric_cols: Iterable[str] | None = None,
    max_lags: int = 2,
) -> tuple[pd.DataFrame, list[str]]:
    """Create lag, rolling, calendar, and categorical frequency features.

    The function is intentionally deterministic and conservative so it can be
    safely applied to large data while preserving the original columns.
    """
    features = df.copy()
    created: list[str] = []

    date_cols = list(datetime_cols or [])
    if not date_cols:
        date_cols = [c for c in features.columns if pd.api.types.is_datetime64_any_dtype(features[c])]

    for col in date_cols:
        if col not in features.columns:
            continue
        dt = pd.to_datetime(features[col], errors="coerce")
        if dt.notna().sum() == 0:
            continue
        features[col] = dt
        features[f"{col}_year"] = dt.dt.year
        features[f"{col}_month"] = dt.dt.month
        features[f"{col}_day_of_week"] = dt.dt.dayofweek
        features["day_of_week"] = dt.dt.dayofweek
        created.extend([f"{col}_year", f"{col}_month", f"{col}_day_of_week", "day_of_week"])

    for col in list(numeric_cols or get_numeric_cols(features)):
        if col not in features.columns:
            continue
        series = pd.to_numeric(features[col], errors="coerce")
        features[col] = series
        if len(features) >= 2:
            for lag in range(1, max_lags + 1):
                lag_name = f"{col}_lag_{lag}"
                features[lag_name] = series.shift(lag)
                created.append(lag_name)
            rolling_name = f"{col}_rolling_mean_3"
            features[rolling_name] = series.rolling(window=3, min_periods=1).mean()
            created.append(rolling_name)

    for col in [c for c in features.columns if c not in created and features[c].dtype == "object"]:
        if features[col].nunique(dropna=True) <= 20:
            freq = features[col].value_counts(dropna=False)
            features[f"{col}_freq"] = features[col].map(freq)
            created.append(f"{col}_freq")

    return features, created
