"""
schema.py — Semantic schema inference.

Goes beyond pandas dtypes to classify every column into one of:
    numeric | categorical | binary | ordinal | datetime | boolean |
    id | text | constant | mixed

This drives downstream chart selection, ML target detection, and cleaning
decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from config import THRESH


class ColumnType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    ORDINAL = "ordinal"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    ID = "id"
    TEXT = "text"
    CONSTANT = "constant"
    MIXED = "mixed"


_ORDINAL_HINTS = {
    "low", "medium", "high", "small", "large", "poor", "fair", "good", "excellent",
    "bronze", "silver", "gold", "platinum", "junior", "senior", "beginner",
    "intermediate", "advanced", "expert",
}
_ID_NAME_HINT = ("id", "_id", "uuid", "code", "key", "index", "no.", "number")


@dataclass
class ColumnProfile:
    name: str
    inferred_type: ColumnType
    dtype: str
    n_unique: int
    n_missing: int
    missing_pct: float
    sample_values: list = field(default_factory=list)


def _looks_like_id(series: pd.Series, name: str) -> bool:
    n = series.notna().sum()
    if n == 0:
        return False
    uniqueness = series.nunique(dropna=True) / n
    name_hint = any(h in name.lower() for h in _ID_NAME_HINT)
    return uniqueness >= THRESH.id_uniqueness_ratio and (name_hint or uniqueness == 1.0)


def _looks_ordinal(series: pd.Series) -> bool:
    vals = {str(v).strip().lower() for v in series.dropna().unique()}
    if not vals or len(vals) > 12:
        return False
    return len(vals & _ORDINAL_HINTS) >= 2


def infer_column_type(df: pd.DataFrame, col: str) -> ColumnType:
    series = df[col]
    n = len(series)
    n_missing = series.isna().sum()
    n_non_null = n - n_missing

    if n_non_null == 0:
        return ColumnType.CONSTANT

    n_unique = series.nunique(dropna=True)

    if n_unique <= 1:
        return ColumnType.CONSTANT

    if pd.api.types.is_bool_dtype(series):
        return ColumnType.BOOLEAN

    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnType.DATETIME

    if pd.api.types.is_numeric_dtype(series):
        if n_unique == 2:
            return ColumnType.BINARY
        if _looks_like_id(series, col):
            return ColumnType.ID
        # low-cardinality integers with small range can behave ordinally
        if pd.api.types.is_integer_dtype(series) and n_unique <= 10 and \
                series.min() >= 0 and (series.max() - series.min()) <= 20:
            return ColumnType.ORDINAL
        return ColumnType.NUMERIC

    # object / category columns
    if n_unique == 2:
        return ColumnType.BINARY

    if _looks_like_id(series, col):
        return ColumnType.ID

    if _looks_ordinal(series):
        return ColumnType.ORDINAL

    unique_ratio = n_unique / n_non_null
    avg_len = series.dropna().astype(str).str.len().mean() if n_non_null else 0

    if unique_ratio > THRESH.categorical_max_unique_ratio and \
            n_unique > THRESH.categorical_max_unique_abs and avg_len > THRESH.text_avg_len_threshold:
        return ColumnType.TEXT

    if unique_ratio > THRESH.categorical_max_unique_ratio and n_unique > THRESH.categorical_max_unique_abs:
        return ColumnType.MIXED

    return ColumnType.CATEGORICAL


def build_schema(df: pd.DataFrame) -> dict[str, ColumnProfile]:
    profiles: dict[str, ColumnProfile] = {}
    n = len(df)
    for col in df.columns:
        try:
            ctype = infer_column_type(df, col)
        except Exception:
            ctype = ColumnType.MIXED
        n_missing = int(df[col].isna().sum())
        n_unique = int(df[col].nunique(dropna=True))
        sample = df[col].dropna().unique()[:5].tolist()
        profiles[col] = ColumnProfile(
            name=col,
            inferred_type=ctype,
            dtype=str(df[col].dtype),
            n_unique=n_unique,
            n_missing=n_missing,
            missing_pct=round(100 * n_missing / n, 1) if n else 0.0,
            sample_values=[str(s) for s in sample],
        )
    return profiles


def columns_of_type(schema: dict[str, ColumnProfile], *types: ColumnType) -> list[str]:
    return [name for name, p in schema.items() if p.inferred_type in types]
