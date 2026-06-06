"""Shared utility helpers."""
from __future__ import annotations

import re
import io
import pandas as pd
import numpy as np
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# Numeric coercion
# ──────────────────────────────────────────────────────────────────────────────

_CURRENCY_RE = re.compile(r"[\$₹€£¥,\s]")
_PCT_RE = re.compile(r"%\s*$")


def coerce_numeric(series: pd.Series) -> pd.Series:
    """
    Strip currency symbols, commas, whitespace, and trailing '%' from a
    string Series, then convert to float.  Non-convertible values → NaN.
    """
    if series.dtype != object:
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(_CURRENCY_RE, "", regex=True)
        .str.replace(_PCT_RE, "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def try_coerce_column(series: pd.Series) -> pd.Series:
    """
    Attempt numeric coercion on an object column.
    Returns the coerced series only if >50% of non-null values convert
    successfully (avoids clobbering genuinely categorical columns).
    """
    if series.dtype != object:
        return series

    coerced = coerce_numeric(series)
    non_null_orig = series.notna().sum()
    if non_null_orig == 0:
        return series

    success_rate = coerced.notna().sum() / non_null_orig
    if success_rate >= 0.5:
        return coerced
    return series


# ──────────────────────────────────────────────────────────────────────────────
# DataFrame helpers
# ──────────────────────────────────────────────────────────────────────────────

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to UTF-8 CSV bytes for st.download_button."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def quality_score(df: pd.DataFrame) -> float:
    """
    0-100 completeness score: percentage of non-null cells in the DataFrame.
    """
    total = df.size
    if total == 0:
        return 0.0
    return round(100 * df.notna().sum().sum() / total, 1)


def get_numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=np.number).columns.tolist()


def get_categorical_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def get_datetime_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["datetime64"]).columns.tolist()


def safe_sample(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return df.head(min(n, len(df)))
