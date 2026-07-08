"""
validators.py — Defensive validation helpers.

Every public entry point in the pipeline should validate its inputs using
these helpers so failures surface as clear, recoverable messages instead of
stack traces.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class ValidationError(Exception):
    """Raised for recoverable, user-facing validation failures."""


@dataclass
class ValidationSummary:
    valid: bool
    issues: list[str] = field(default_factory=list)
    missing_cells: int = 0
    duplicate_rows: int = 0
    rows: int = 0
    columns: int = 0
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)


def validate_dataframe(df: pd.DataFrame, *, allow_empty: bool = False) -> ValidationSummary:
    summary = ValidationSummary(valid=True, rows=0, columns=0)
    issues: list[str] = []

    if df is None:
        issues.append("No dataset is loaded.")
    elif not isinstance(df, pd.DataFrame):
        issues.append("Loaded object is not a valid table.")
    else:
        summary.rows = int(df.shape[0])
        summary.columns = int(df.shape[1])
        summary.missing_cells = int(df.isna().sum().sum())
        summary.duplicate_rows = int(df.duplicated().sum())
        summary.numeric_columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        summary.categorical_columns = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

        if df.shape[1] == 0:
            issues.append("Dataset has no columns.")
        if df.shape[0] == 0 and not allow_empty:
            issues.append("Dataset has no rows.")
        if summary.duplicate_rows > 0:
            issues.append(f"Dataset contains {summary.duplicate_rows} duplicate row(s).")
        if summary.missing_cells > 0:
            issues.append(f"Dataset contains {summary.missing_cells} missing value(s).")

    if issues:
        summary.issues = issues
        critical = [i for i in issues if "no columns" in i.lower() or "no rows" in i.lower() or "not a valid table" in i.lower() or "loaded" in i.lower()]
        summary.valid = not critical
        if critical:
            raise ValidationError("; ".join(critical))
        return summary

    summary.issues = []
    return summary


def validate_columns_exist(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return only the columns that actually exist; never raises."""
    return [c for c in columns if c in df.columns]


def require_numeric(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise ValidationError(f"Column '{col}' not found.")
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise ValidationError(f"Column '{col}' is not numeric.")


def sanitize_infinities(df: pd.DataFrame) -> pd.DataFrame:
    """Replace +/-inf with NaN across numeric columns (never raises)."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    return df


def dedupe_columns(columns: list[str]) -> list[str]:
    """Rename duplicate column headers by suffixing _1, _2, ..."""
    seen: dict[str, int] = {}
    result = []
    for c in columns:
        c = str(c).strip() or "unnamed"
        if c not in seen:
            seen[c] = 0
            result.append(c)
        else:
            seen[c] += 1
            result.append(f"{c}_{seen[c]}")
    return result


def safe_bool(value, default: bool = False) -> bool:
    try:
        return bool(value)
    except Exception:
        return default
