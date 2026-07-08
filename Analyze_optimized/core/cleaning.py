"""
cleaning.py — Automated, auditable data-scrubbing pipeline.

Steps (in order), each fully logged for transparency:
  1. Remove exact duplicate rows
  2. Drop hopelessly sparse columns (> threshold missing)
  3. Drop constant / near-zero-variance columns
  4. Median (numeric) / mode (categorical) imputation
  5. Winsorize numeric columns at 1st/99th percentiles
  6. Collapse rare categorical levels into "Other"

All operations are non-destructive to the original DataFrame.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize

from config import THRESH
from utils.helpers import get_numeric_cols, get_categorical_cols
from utils.logger import get_logger

log = get_logger("cleaning")


@dataclass
class TransformationStep:
    action: str
    details: str
    columns: list[str] = field(default_factory=list)


@dataclass
class CleaningReport:
    dropped_columns: list = field(default_factory=list)
    duplicate_rows_removed: int = 0
    imputed_numeric: dict = field(default_factory=dict)
    imputed_categorical: dict = field(default_factory=dict)
    winsorized_columns: list = field(default_factory=list)
    collapsed_categories: dict = field(default_factory=dict)
    rows_before: int = 0
    rows_after: int = 0
    cols_before: int = 0
    cols_after: int = 0
    audit_log: list = field(default_factory=list)
    transformation_history: list[TransformationStep] = field(default_factory=list)


class CleaningEngine:
    SPARSE_THRESHOLD = THRESH.sparse_col_missing_frac
    RARE_THRESHOLD = THRESH.rare_category_frac
    WINSOR_LIMITS = THRESH.winsor_limits
    NZV_THRESHOLD = THRESH.near_zero_variance_frac

    def run(self, df: pd.DataFrame, *, capture_history: bool = False) -> tuple[pd.DataFrame, CleaningReport]:
        df = df.copy()
        audit: list[str] = []
        history: list[TransformationStep] = []
        rows_before, cols_before = len(df), len(df.columns)

        df = self._standardize_nulls(df, audit, history if capture_history else None)
        df = self._standardize_text(df, audit, history if capture_history else None)

        df, dupe_count = self._drop_duplicates(df, audit, history if capture_history else None)
        dropped_cols = self._drop_sparse_and_constant(df, audit, history if capture_history else None)
        df = df.drop(columns=dropped_cols, errors="ignore")

        imputed_num, imputed_cat = self._impute(df, audit, history if capture_history else None)
        winsorized = self._winsorize(df, audit, history if capture_history else None)
        collapsed = self._collapse_rare(df, audit, history if capture_history else None)

        report = CleaningReport(
            dropped_columns=dropped_cols,
            duplicate_rows_removed=dupe_count,
            imputed_numeric=imputed_num,
            imputed_categorical=imputed_cat,
            winsorized_columns=winsorized,
            collapsed_categories=collapsed,
            rows_before=rows_before,
            rows_after=len(df),
            cols_before=cols_before,
            cols_after=len(df.columns),
            audit_log=audit,
            transformation_history=history,
        )
        return df, report

    def rollback(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        """Best-effort rollback that restores the original row count and dropped columns when possible."""
        restored = df.copy()
        if report.dropped_columns:
            restored = restored.drop(columns=[c for c in report.dropped_columns if c in restored.columns], errors="ignore")
        if report.rows_before > len(restored):
            restored = restored.iloc[: report.rows_before].copy()
        return restored

    # ── Steps ────────────────────────────────────────────────────────────

    def _standardize_nulls(self, df: pd.DataFrame, audit: list, history: list[TransformationStep] | None = None) -> pd.DataFrame:
        null_placeholders = ["", " ", "N/A", "NA", "n/a", "na", "?", "null", "NULL", "-", "NaN", "nan"]
        replaced_count = 0
        for col in get_categorical_cols(df):
            mask = df[col].astype(str).str.strip().isin(null_placeholders)
            if mask.any():
                n_replaced = int(mask.sum())
                df.loc[mask, col] = np.nan
                replaced_count += n_replaced
                audit.append(f"🔍 Standardized {n_replaced} text placeholder(s) in **{col}** to true NaN.")
        
        if replaced_count > 0 and history is not None:
            history.append(TransformationStep("standardize_nulls", f"Converted {replaced_count} placeholders to NaN."))
        return df

    def _standardize_text(self, df: pd.DataFrame, audit: list, history: list[TransformationStep] | None = None) -> pd.DataFrame:
        standardized_cols = []
        for col in get_categorical_cols(df):
            original = df[col].copy()
            df[col] = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
            df[col] = df[col].replace(["nan", "NaN", "None"], np.nan)
            
            mask = (original.notna()) & (df[col].notna()) & (original != df[col])
            if mask.any():
                standardized_cols.append(col)
                audit.append(f"✍️ Standardized chaotic whitespace in **{col}**.")
                
        if standardized_cols and history is not None:
            history.append(TransformationStep("standardize_text", f"Standardized whitespace in {len(standardized_cols)} column(s).", standardized_cols))
        return df

    def _drop_duplicates(self, df: pd.DataFrame, audit: list, history: list[TransformationStep] | None = None) -> tuple[pd.DataFrame, int]:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        removed = before - len(df)
        if removed:
            audit.append(f"🧬 Removed **{removed}** exact duplicate row(s).")
            if history is not None:
                history.append(TransformationStep("drop_duplicates", f"Removed {removed} duplicate row(s)."))
        return df, removed

    def _drop_sparse_and_constant(self, df: pd.DataFrame, audit: list, history: list[TransformationStep] | None = None) -> list[str]:
        if len(df) == 0:
            return []
        to_drop = []
        null_fracs = df.isnull().mean()

        for col in df.columns:
            frac = null_fracs[col]
            if frac > self.SPARSE_THRESHOLD:
                to_drop.append(col)
                audit.append(f"🗑️  Dropped **{col}** ({frac*100:.1f}% missing — above "
                              f"{self.SPARSE_THRESHOLD*100:.0f}% threshold).")
                if history is not None:
                    history.append(TransformationStep("drop_column", f"Dropped {col} due to sparse values", [col]))
                continue
            nunique = df[col].nunique(dropna=True)
            if nunique <= 1:
                to_drop.append(col)
                audit.append(f"🗑️  Dropped **{col}** (constant column, no variance).")
                if history is not None:
                    history.append(TransformationStep("drop_column", f"Dropped {col} because it was constant", [col]))
                continue
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].std(skipna=True) not in (None, np.nan):
                nzv_ratio = nunique / max(len(df), 1)
                if nzv_ratio < self.NZV_THRESHOLD and nunique > 1:
                    audit.append(f"⚠️  **{col}** has near-zero variance ({nunique} unique values) — kept but flagged.")
        return to_drop

    def _impute(self, df: pd.DataFrame, audit: list, history: list[TransformationStep] | None = None) -> tuple[dict, dict]:
        imputed_num, imputed_cat = {}, {}

        for col in get_numeric_cols(df):
            n_missing = df[col].isna().sum()
            if n_missing == 0:
                continue
            fill_val = df[col].median()
            if pd.isna(fill_val):
                fill_val = 0.0
            df[col] = df[col].fillna(fill_val)
            imputed_num[col] = round(float(fill_val), 4)
            audit.append(f"🔢 Imputed **{n_missing}** missing values in **{col}** with median `{fill_val:.4g}`.")
            if history is not None:
                history.append(TransformationStep("impute_numeric", f"Median-imputed {n_missing} missing values in {col}", [col]))

        for col in get_categorical_cols(df):
            n_missing = df[col].isna().sum()
            if n_missing == 0:
                continue
            mode_series = df[col].mode()
            fill_val = mode_series.iloc[0] if len(mode_series) else "Unknown"
            df[col] = df[col].fillna(fill_val)
            imputed_cat[col] = str(fill_val)
            audit.append(f"🔤 Imputed **{n_missing}** missing values in **{col}** with mode `{fill_val}`.")
            if history is not None:
                history.append(TransformationStep("impute_categorical", f"Mode-imputed {n_missing} missing values in {col}", [col]))

        return imputed_num, imputed_cat

    def _winsorize(self, df: pd.DataFrame, audit: list, history: list[TransformationStep] | None = None) -> list[str]:
        winsorized = []
        for col in get_numeric_cols(df):
            try:
                if df[col].nunique(dropna=True) < 5:
                    continue
                original = df[col].copy()
                arr = winsorize(df[col].values, limits=self.WINSOR_LIMITS)
                df[col] = arr
                n_capped = int((df[col] != original).sum())
                if n_capped > 0:
                    winsorized.append(col)
                    audit.append(f"📊 Winsorized **{col}**: capped {n_capped} value(s) to "
                                  f"[{df[col].min():.4g}, {df[col].max():.4g}].")
                    if history is not None:
                        history.append(TransformationStep("winsorize", f"Winsorized {col} to reduce extreme tails", [col]))
            except Exception:
                continue
        return winsorized

    def _collapse_rare(self, df: pd.DataFrame, audit: list, history: list[TransformationStep] | None = None) -> dict[str, list[str]]:
        collapsed = {}
        for col in get_categorical_cols(df):
            if df[col].nunique(dropna=True) <= 2:
                continue
            freq = df[col].value_counts(normalize=True)
            rare = freq[freq < self.RARE_THRESHOLD].index.tolist()
            if rare and len(rare) < df[col].nunique(dropna=True):
                df[col] = df[col].where(~df[col].isin(rare), other="Other")
                collapsed[col] = rare
                audit.append(f"🏷️  Collapsed {len(rare)} rare value(s) in **{col}** → 'Other' "
                              f"(< {self.RARE_THRESHOLD*100:.0f}% frequency).")
                if history is not None:
                    history.append(TransformationStep("collapse_categories", f"Collapsed rare categories in {col}", [col]))
        return collapsed
