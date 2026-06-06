"""
IngestionEngine — Agnostic multi-format file parser.

Supports: CSV, TSV, JSON, Excel (XLS/XLSX), ODS.
Performs aggressive numeric coercion and safe datetime inference.
"""
from __future__ import annotations

import io
import json
import re
import traceback
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from utils.helpers import try_coerce_column, get_numeric_cols, get_categorical_cols


# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SchemaReport:
    shape: tuple
    dtypes: dict
    null_counts: dict
    null_pct: dict
    cardinality: dict
    numeric_cols: list
    categorical_cols: list
    datetime_cols: list
    warnings: list[str] = field(default_factory=list)


class IngestionEngine:
    """
    Parse a Streamlit UploadedFile (or a file-like bytes object) into a
    clean-ish DataFrame.  All operations are wrapped in try/except so that
    messy data never crashes the application.
    """

    SUPPORTED_EXTENSIONS = {
        ".csv", ".tsv", ".txt", ".json",
        ".xls", ".xlsx", ".ods",
    }

    def __init__(self):
        self._warnings: list[str] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def parse_file(self, uploaded_file) -> tuple[pd.DataFrame, SchemaReport]:
        """
        Entry point.  Returns (DataFrame, SchemaReport).
        Raises ValueError with a descriptive message on unrecoverable failure.
        """
        self._warnings = []
        name: str = getattr(uploaded_file, "name", "unknown")
        ext = self._get_extension(name)

        raw_df = self._dispatch_parse(uploaded_file, ext, name)
        df = self._coerce_schema(raw_df)
        report = self._build_report(df)
        return df, report

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def _dispatch_parse(self, f, ext: str, name: str) -> pd.DataFrame:
        try:
            if ext in (".csv", ".txt"):
                return self._parse_csv(f, sep=",")
            elif ext == ".tsv":
                return self._parse_csv(f, sep="\t")
            elif ext == ".json":
                return self._parse_json(f)
            elif ext in (".xls", ".xlsx"):
                return pd.read_excel(f, engine="openpyxl" if ext == ".xlsx" else "xlrd")
            elif ext == ".ods":
                return pd.read_excel(f, engine="odf")
            else:
                raise ValueError(
                    f"Unsupported file type '{ext}'. "
                    f"Supported: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
                )
        except Exception as e:
            raise ValueError(f"Failed to parse '{name}': {e}") from e

    def _parse_csv(self, f, sep: str) -> pd.DataFrame:
        """Try several encodings to handle Windows-1252, UTF-8-BOM, Latin-1."""
        raw_bytes = f.read() if hasattr(f, "read") else f
        for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
            try:
                return pd.read_csv(
                    io.BytesIO(raw_bytes),
                    sep=sep,
                    low_memory=False,
                    encoding=enc,
                    on_bad_lines="warn",
                )
            except (UnicodeDecodeError, Exception):
                continue
        raise ValueError("Could not decode CSV file with any standard encoding.")

    def _parse_json(self, f) -> pd.DataFrame:
        raw_bytes = f.read() if hasattr(f, "read") else f
        try:
            data = json.loads(raw_bytes.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # Try common nested structures: {data: [...]} or {records: [...]}
            for key in ("data", "records", "rows", "results", "items"):
                if key in data and isinstance(data[key], list):
                    return pd.DataFrame(data[key])
            # Fallback: treat dict as single-row or orient as columns
            try:
                return pd.DataFrame([data])
            except Exception:
                return pd.DataFrame.from_dict(data)
        else:
            raise ValueError("JSON root must be a list or dict.")

    # ── Schema coercion ───────────────────────────────────────────────────────

    def _coerce_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1. Drop fully empty rows.
        2. Strip whitespace from column names.
        3. Attempt numeric coercion on object columns (strips $, ₹, €, %, ,).
        4. Attempt datetime inference on object columns with date-like names.
        """
        # Drop fully empty rows
        before = len(df)
        df = df.dropna(how="all").reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            self._warnings.append(f"Dropped {dropped} fully-empty rows.")

        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]

        # Numeric coercion
        for col in df.columns:
            if df[col].dtype == object:
                coerced = try_coerce_column(df[col])
                if coerced is not df[col]:
                    df[col] = coerced
                    self._warnings.append(
                        f"Column '{col}' coerced to numeric (stripped currency/% symbols)."
                    )

        # Datetime inference for columns with date-like names
        date_hints = re.compile(
            r"date|time|dt|timestamp|year|month|day", re.IGNORECASE
        )
        for col in df.columns:
            if date_hints.search(col) and df[col].dtype == object:
                converted = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                success_rate = converted.notna().sum() / max(df[col].notna().sum(), 1)
                if success_rate >= 0.4:
                    df[col] = converted
                    n_failed = df[col].isna().sum() - (df[col].isna().sum() - converted.isna().sum())
                    self._warnings.append(
                        f"Column '{col}' parsed as datetime "
                        f"({int(success_rate*100)}% success rate)."
                    )

        return df

    # ── Report ────────────────────────────────────────────────────────────────

    def _build_report(self, df: pd.DataFrame) -> SchemaReport:
        null_counts = df.isnull().sum().to_dict()
        null_pct = {
            c: round(100 * v / len(df), 1) if len(df) else 0
            for c, v in null_counts.items()
        }
        cardinality = {
            c: int(df[c].nunique()) for c in df.columns
        }
        return SchemaReport(
            shape=df.shape,
            dtypes={c: str(df[c].dtype) for c in df.columns},
            null_counts=null_counts,
            null_pct=null_pct,
            cardinality=cardinality,
            numeric_cols=get_numeric_cols(df),
            categorical_cols=get_categorical_cols(df),
            datetime_cols=df.select_dtypes(include=["datetime64"]).columns.tolist(),
            warnings=list(self._warnings),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_extension(filename: str) -> str:
        idx = filename.rfind(".")
        if idx == -1:
            return ""
        return filename[idx:].lower()
