"""
ingestion.py — Format-agnostic file parser.

Supports CSV, TSV, TXT, JSON, Excel (XLS/XLSX), ODS, and Parquet.
Every parsing branch is wrapped defensively; failures raise ValidationError
with a clear, user-facing message rather than propagating a stack trace.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import THRESH
from utils.helpers import try_coerce_column, downcast_numeric, safe_col_name, is_textual
from utils.validators import ValidationError, dedupe_columns, sanitize_infinities
from utils.logger import get_logger

log = get_logger("ingestion")

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".json", ".xls", ".xlsx", ".ods", ".parquet"}
_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1", "cp1252")


@dataclass
class SchemaReport:
    shape: tuple
    dtypes: dict
    null_counts: dict
    null_pct: dict
    cardinality: dict
    memory_bytes: int
    warnings: list = field(default_factory=list)


class IngestionEngine:
    """Parses an uploaded file-like object into a coerced DataFrame."""

    def __init__(self):
        self._warnings: list[str] = []

    # ── Public API ───────────────────────────────────────────────────────

    def parse_file(self, uploaded_file) -> tuple[pd.DataFrame, SchemaReport]:
        self._warnings = []
        name = getattr(uploaded_file, "name", "unknown")
        ext = self._extension(name)

        if ext not in SUPPORTED_EXTENSIONS:
            raise ValidationError(
                f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        raw_df = self._dispatch(uploaded_file, ext, name)
        if raw_df is None or raw_df.shape[1] == 0:
            raise ValidationError(f"'{name}' could not be parsed into a table (no columns found).")

        df = self._coerce_schema(raw_df)
        df = sanitize_infinities(df)
        df = downcast_numeric(df)
        report = self._build_report(df)
        return df, report

    # ── Format dispatch ──────────────────────────────────────────────────

    def _dispatch(self, f, ext: str, name: str) -> pd.DataFrame:
        try:
            if ext in (".csv", ".txt"):
                return self._parse_delimited(f, sep=",")
            if ext == ".tsv":
                return self._parse_delimited(f, sep="\t")
            if ext == ".json":
                return self._parse_json(f)
            if ext == ".xlsx":
                return self._parse_excel(f, engine="openpyxl")
            if ext == ".xls":
                return self._parse_excel(f, engine="xlrd")
            if ext == ".ods":
                return self._parse_excel(f, engine="odf")
            if ext == ".parquet":
                return pd.read_parquet(io.BytesIO(self._read_bytes(f)))
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Failed to parse '{name}': {e}") from e
        return None

    @staticmethod
    def _read_bytes(f) -> bytes:
        return f.read() if hasattr(f, "read") else f

    def _parse_delimited(self, f, sep: str) -> pd.DataFrame:
        raw = self._read_bytes(f)
        last_err = None
        for enc in _ENCODINGS:
            try:
                df = pd.read_csv(
                    io.BytesIO(raw), sep=sep,
                    encoding=enc, on_bad_lines="skip", engine="python",
                )
                if enc != "utf-8":
                    self._warnings.append(f"File decoded using fallback encoding '{enc}'.")
                return df
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        raise ValidationError(f"Could not decode file with any supported encoding ({last_err}).")

    def _parse_json(self, f) -> pd.DataFrame:
        raw = self._read_bytes(f)
        try:
            data = json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON: {e}") from e

        if isinstance(data, list):
            return pd.json_normalize(data)
        if isinstance(data, dict):
            for key in ("data", "records", "rows", "results", "items"):
                if key in data and isinstance(data[key], list):
                    return pd.json_normalize(data[key])
            try:
                return pd.json_normalize([data])
            except Exception:
                return pd.DataFrame.from_dict(data)
        raise ValidationError("JSON root must be an object or array.")

    def _parse_excel(self, f, engine: str) -> pd.DataFrame:
        raw = self._read_bytes(f)
        buf = io.BytesIO(raw)
        xls = pd.ExcelFile(buf, engine=engine)
        if not xls.sheet_names:
            raise ValidationError("Workbook contains no sheets.")
        # Pick the first non-empty sheet
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            if not df.empty and df.shape[1] > 0:
                if len(xls.sheet_names) > 1:
                    self._warnings.append(
                        f"Workbook had {len(xls.sheet_names)} sheets; using first non-empty sheet '{sheet}'."
                    )
                return df
        raise ValidationError("All sheets in the workbook are empty.")

    # ── Schema coercion ──────────────────────────────────────────────────

    def _coerce_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        before_rows = len(df)
        df = df.dropna(how="all").reset_index(drop=True)
        if before_rows - len(df):
            self._warnings.append(f"Dropped {before_rows - len(df)} fully-empty row(s).")

        df.columns = dedupe_columns([safe_col_name(c) for c in df.columns])

        # Drop fully-empty columns (all NaN) early — still reported downstream.
        empty_cols = [c for c in df.columns if df[c].isna().all()]
        if empty_cols:
            self._warnings.append(f"{len(empty_cols)} column(s) are entirely empty: {', '.join(empty_cols[:5])}"
                                   + (" ..." if len(empty_cols) > 5 else ""))

        # Numeric coercion (currency / percentage strings)
        for col in df.columns:
            if is_textual(df[col]):
                coerced = try_coerce_column(df[col])
                if coerced is not df[col]:
                    df[col] = coerced
                    self._warnings.append(f"'{col}' coerced to numeric (currency/% symbols stripped).")

        # Datetime inference — try any remaining object column with a plausible
        # hit rate, prioritizing columns whose name hints at dates.
        import re
        date_hint = re.compile(r"date|time|dt|timestamp|year|month|day", re.IGNORECASE)
        for col in df.columns:
            if not is_textual(df[col]):
                continue
            looks_like_date = bool(date_hint.search(col))
            non_null = df[col].notna().sum()
            if non_null == 0:
                continue
            try:
                converted = pd.to_datetime(df[col], errors="coerce", format="mixed")
            except (TypeError, ValueError):
                converted = pd.to_datetime(df[col], errors="coerce")
            success_rate = converted.notna().sum() / non_null
            threshold = THRESH.datetime_parse_success_rate if looks_like_date else 0.9
            if success_rate >= threshold:
                df[col] = converted
                self._warnings.append(f"'{col}' parsed as datetime ({success_rate*100:.0f}% success).")

        return df

    def _build_report(self, df: pd.DataFrame) -> SchemaReport:
        null_counts = df.isnull().sum().to_dict()
        n = len(df)
        return SchemaReport(
            shape=df.shape,
            dtypes={c: str(df[c].dtype) for c in df.columns},
            null_counts=null_counts,
            null_pct={c: round(100 * v / n, 1) if n else 0.0 for c, v in null_counts.items()},
            cardinality={c: int(df[c].nunique(dropna=True)) for c in df.columns},
            memory_bytes=int(df.memory_usage(deep=True).sum()),
            warnings=list(self._warnings),
        )

    @staticmethod
    def _extension(filename: str) -> str:
        idx = filename.rfind(".")
        return filename[idx:].lower() if idx != -1 else ""
