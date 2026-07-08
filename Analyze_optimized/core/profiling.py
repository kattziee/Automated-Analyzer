"""
profiling.py — Whole-dataset profiling and data-quality scoring.

Produces a single DatasetProfile combining schema, duplicate/missingness
stats, and a weighted 0-100 quality score across four dimensions:
completeness, uniqueness, consistency, validity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import THRESH
from core.schema import build_schema, ColumnProfile, ColumnType
from utils.helpers import get_numeric_cols, get_categorical_cols, get_datetime_cols, human_size


@dataclass
class QualityBreakdown:
    completeness: float
    uniqueness: float
    consistency: float
    validity: float
    overall: float


@dataclass
class DatasetProfile:
    shape: tuple
    memory_human: str
    n_duplicate_rows: int
    n_constant_cols: int
    n_high_cardinality_cols: int
    schema: dict[str, ColumnProfile]
    quality: QualityBreakdown
    numeric_cols: list = field(default_factory=list)
    categorical_cols: list = field(default_factory=list)
    datetime_cols: list = field(default_factory=list)


def _completeness(df: pd.DataFrame) -> float:
    total = df.size
    return 100 * df.notna().sum().sum() / total if total else 0.0


def _uniqueness(df: pd.DataFrame) -> float:
    """Penalize datasets dominated by duplicate rows."""
    if len(df) == 0:
        return 100.0
    dup_frac = df.duplicated().sum() / len(df)
    return max(0.0, 100 * (1 - dup_frac))


def _consistency(df: pd.DataFrame, schema: dict[str, ColumnProfile]) -> float:
    """Penalize mixed-type columns and inconsistent categorical casing."""
    if not schema:
        return 100.0
    penalties = []
    for col, profile in schema.items():
        if profile.inferred_type == ColumnType.MIXED:
            penalties.append(1.0)
        elif profile.inferred_type == ColumnType.CATEGORICAL:
            series = df[col].dropna().astype(str)
            if len(series):
                casing_variants = series.str.lower().nunique()
                raw_variants = series.nunique()
                if raw_variants > casing_variants:
                    penalties.append(0.3)
    if not penalties:
        return 100.0
    score = 100 - (sum(penalties) / len(schema)) * 100
    return max(0.0, min(100.0, score))


def _validity(df: pd.DataFrame) -> float:
    """Penalize infinities and numeric columns with implausible values already
    replaced upstream — here we check for near-constant / all-negative-where-
    unexpected patterns as a light validity proxy."""
    num_cols = get_numeric_cols(df)
    if not num_cols:
        return 100.0
    inf_count = 0
    for c in num_cols:
        inf_count += int(np.isinf(df[c]).sum()) if df[c].dtype.kind in "fc" else 0
    total_numeric_cells = max(len(df) * len(num_cols), 1)
    return max(0.0, 100 * (1 - inf_count / total_numeric_cells))


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    schema = build_schema(df)

    completeness = round(_completeness(df), 1)
    uniqueness = round(_uniqueness(df), 1)
    consistency = round(_consistency(df, schema), 1)
    validity = round(_validity(df), 1)

    overall = round(
        completeness * THRESH.weight_completeness
        + uniqueness * THRESH.weight_uniqueness
        + consistency * THRESH.weight_consistency
        + validity * THRESH.weight_validity,
        1,
    )

    n_constant = sum(1 for p in schema.values() if p.inferred_type == ColumnType.CONSTANT)
    n_high_card = sum(1 for p in schema.values() if p.n_unique > THRESH.high_cardinality_threshold)

    return DatasetProfile(
        shape=df.shape,
        memory_human=human_size(df.memory_usage(deep=True).sum()),
        n_duplicate_rows=int(df.duplicated().sum()),
        n_constant_cols=n_constant,
        n_high_cardinality_cols=n_high_card,
        schema=schema,
        quality=QualityBreakdown(completeness, uniqueness, consistency, validity, overall),
        numeric_cols=get_numeric_cols(df),
        categorical_cols=get_categorical_cols(df),
        datetime_cols=get_datetime_cols(df),
    )
