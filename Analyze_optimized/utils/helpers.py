"""helpers.py — Shared, dependency-light utility functions."""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd
import pandas.api.types as ptypes

from config import THRESH

_CURRENCY_RE = re.compile(r"[\$₹€£¥,\s]")
_PCT_RE = re.compile(r"%\s*$")


def is_textual(series: pd.Series) -> bool:
    """True for object/string-like columns; False for numeric/datetime/bool.

    Uses semantic dtype checks rather than `dtype == object` because pandas
    2.3+/3.x may default to a dedicated `str` extension dtype that is NOT
    equal to `object`, which would otherwise silently skip coercion.
    """
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series) \
            or pd.api.types.is_bool_dtype(series):
        return False
    return True


# ── Numeric / type coercion ─────────────────────────────────────────────────

def coerce_numeric(series: pd.Series) -> pd.Series:
    """Strip currency symbols, thousands separators, and trailing '%' then
    convert to float. Non-convertible values become NaN."""
    if not is_textual(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(_CURRENCY_RE, "", regex=True)
        .str.replace(_PCT_RE, "", regex=True)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "N/A": np.nan,
                   "NA": np.nan, "null": np.nan, "NULL": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def try_coerce_column(series: pd.Series, min_success: float = 0.5) -> pd.Series:
    """Coerce an object column to numeric only if the majority of non-null
    values convert successfully (avoids clobbering genuine text columns)."""
    if not is_textual(series):
        return series
    non_null = series.notna().sum()
    if non_null == 0:
        return series
    coerced = coerce_numeric(series)
    if coerced.notna().sum() / non_null >= min_success:
        return coerced
    return series


def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory footprint by downcasting numeric dtypes in place."""
    for col in df.select_dtypes(include=["int64", "int32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def maybe_sample(df: pd.DataFrame, n: int = None, random_state: int = None) -> pd.DataFrame:
    """Return a deterministic sample for heavy computations on large data."""
    n = n or THRESH.sample_rows_for_heavy_ops
    random_state = random_state if random_state is not None else THRESH.random_state
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=random_state)


# ── Reporting helpers ────────────────────────────────────────────────────────

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def df_to_json_bytes(df: pd.DataFrame) -> bytes:
    return df.to_json(orient="records", date_format="iso", indent=2).encode("utf-8")


def human_size(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def quality_score(df: pd.DataFrame) -> float:
    """0-100 completeness score: percentage of non-null cells."""
    total = df.size
    if total == 0:
        return 0.0
    return round(100 * df.notna().sum().sum() / total, 1)


def safe_col_name(name: str) -> str:
    return str(name).strip() or "unnamed"


# ── Column-type accessors (schema-agnostic, tolerant of empty frames) ──────

def _get_columns_by_predicate(df: pd.DataFrame, predicate) -> list[str]:
    """Return all columns in a frame matching a predicate over the column values."""
    return [col for col in df.columns if predicate(df[col])]


def get_numeric_cols(df: pd.DataFrame) -> list[str]:
    return _get_columns_by_predicate(df, ptypes.is_numeric_dtype)


def get_categorical_cols(df: pd.DataFrame) -> list[str]:
    """Include object, category, and pandas string extension dtypes."""
    return _get_columns_by_predicate(df, lambda series: (
        ptypes.is_object_dtype(series)
        or ptypes.is_categorical_dtype(series)
        or ptypes.is_string_dtype(series)
    ))


def get_datetime_cols(df: pd.DataFrame) -> list[str]:
    return _get_columns_by_predicate(df, ptypes.is_datetime64_any_dtype)


def get_boolean_cols(df: pd.DataFrame) -> list[str]:
    return _get_columns_by_predicate(df, ptypes.is_bool_dtype)


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    try:
        return a / b if b else default
    except Exception:
        return default
