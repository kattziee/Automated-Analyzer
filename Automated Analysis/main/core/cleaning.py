"""
CleaningEngine — Automated data scrubbing pipeline.

Steps (in order):
  1. Drop hopelessly sparse columns (>60% missing)
  2. Median imputation for numeric, mode for categorical
  3. Winsorize numeric columns at 1st / 99th percentiles
  4. Collapse rare categorical levels (<2% frequency) → "Other"
  5. Return cleaned DataFrame + human-readable audit log
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize

from utils.helpers import get_numeric_cols, get_categorical_cols


# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CleaningReport:
    dropped_columns: list[str]
    imputed_numeric: dict[str, float]   # col → fill value used
    imputed_categorical: dict[str, str] # col → fill value used
    winsorized_columns: list[str]
    collapsed_categories: dict[str, list[str]]  # col → values → "Other"
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    audit_log: list[str] = field(default_factory=list)


class CleaningEngine:
    """
    Applies a sequential data-cleaning pipeline.  All operations are
    non-destructive to the original DataFrame; a copy is returned.
    """

    SPARSE_THRESHOLD = 0.60   # drop columns with > 60% missing
    RARE_THRESHOLD   = 0.02   # collapse categories with < 2% frequency
    WINSOR_LIMITS    = (0.01, 0.01)  # 1st / 99th percentile caps

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
        """
        Run the full cleaning pipeline.
        Returns (cleaned_df, CleaningReport).
        """
        df = df.copy()
        audit: list[str] = []

        rows_before = len(df)
        cols_before = len(df.columns)

        # ── Step 1: Drop sparse columns ───────────────────────────────────
        dropped_cols = self._drop_sparse(df, audit)
        df = df.drop(columns=dropped_cols, errors="ignore")

        # ── Step 2: Imputation ────────────────────────────────────────────
        imputed_num, imputed_cat = self._impute(df, audit)

        # ── Step 3: Winsorization ─────────────────────────────────────────
        winsorized = self._winsorize(df, audit)

        # ── Step 4: Rare category collapse ────────────────────────────────
        collapsed = self._collapse_rare(df, audit)

        report = CleaningReport(
            dropped_columns=dropped_cols,
            imputed_numeric=imputed_num,
            imputed_categorical=imputed_cat,
            winsorized_columns=winsorized,
            collapsed_categories=collapsed,
            rows_before=rows_before,
            rows_after=len(df),
            cols_before=cols_before,
            cols_after=len(df.columns),
            audit_log=audit,
        )

        return df, report

    # ── Step implementations ──────────────────────────────────────────────────

    def _drop_sparse(self, df: pd.DataFrame, audit: list) -> list[str]:
        """Identify columns with > SPARSE_THRESHOLD fraction of nulls."""
        if len(df) == 0:
            return []
        null_fracs = df.isnull().mean()
        to_drop = null_fracs[null_fracs > self.SPARSE_THRESHOLD].index.tolist()
        for col in to_drop:
            pct = round(null_fracs[col] * 100, 1)
            audit.append(
                f"🗑️  Dropped column **{col}** ({pct}% missing — above {self.SPARSE_THRESHOLD*100:.0f}% threshold)."
            )
        return to_drop

    def _impute(
        self, df: pd.DataFrame, audit: list
    ) -> tuple[dict, dict]:
        num_cols = get_numeric_cols(df)
        cat_cols = get_categorical_cols(df)

        imputed_num: dict[str, float] = {}
        imputed_cat: dict[str, str] = {}

        for col in num_cols:
            n_missing = df[col].isna().sum()
            if n_missing == 0:
                continue
            fill_val = df[col].median()
            if pd.isna(fill_val):
                fill_val = 0.0
            df[col] = df[col].fillna(fill_val)
            imputed_num[col] = round(fill_val, 4)
            audit.append(
                f"🔢 Imputed **{n_missing}** missing values in **{col}** "
                f"with median `{fill_val:.4g}`."
            )

        for col in cat_cols:
            n_missing = df[col].isna().sum()
            if n_missing == 0:
                continue
            mode_series = df[col].mode()
            fill_val = mode_series.iloc[0] if len(mode_series) else "Unknown"
            df[col] = df[col].fillna(fill_val)
            imputed_cat[col] = str(fill_val)
            audit.append(
                f"🔤 Imputed **{n_missing}** missing values in **{col}** "
                f"with mode `{fill_val}`."
            )

        return imputed_num, imputed_cat

    def _winsorize(self, df: pd.DataFrame, audit: list) -> list[str]:
        """Cap extreme values at 1st / 99th percentiles."""
        num_cols = get_numeric_cols(df)
        winsorized: list[str] = []

        for col in num_cols:
            try:
                original = df[col].copy()
                winsorized_arr = winsorize(
                    df[col].values, limits=self.WINSOR_LIMITS
                )
                df[col] = winsorized_arr
                n_capped = (df[col] != original).sum()
                if n_capped > 0:
                    winsorized.append(col)
                    p1 = df[col].min()
                    p99 = df[col].max()
                    audit.append(
                        f"📊 Winsorized **{col}**: capped {n_capped} values "
                        f"to [{p1:.4g}, {p99:.4g}]."
                    )
            except Exception:
                pass  # skip if column has all-same values etc.

        return winsorized

    def _collapse_rare(
        self, df: pd.DataFrame, audit: list
    ) -> dict[str, list[str]]:
        """Replace categories with < RARE_THRESHOLD frequency with 'Other'."""
        cat_cols = get_categorical_cols(df)
        collapsed: dict[str, list[str]] = {}

        for col in cat_cols:
            if df[col].nunique() <= 2:
                continue  # don't collapse binary columns
            freq = df[col].value_counts(normalize=True)
            rare_values = freq[freq < self.RARE_THRESHOLD].index.tolist()
            if rare_values:
                df[col] = df[col].where(~df[col].isin(rare_values), other="Other")
                collapsed[col] = rare_values
                audit.append(
                    f"🏷️  Collapsed {len(rare_values)} rare values in **{col}** → 'Other' "
                    f"(below {self.RARE_THRESHOLD*100:.0f}% frequency threshold)."
                )

        return collapsed
