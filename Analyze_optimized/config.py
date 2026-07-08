"""
config.py — Central, environment-aware configuration for the Automated Data Analyzer.

All tunable thresholds live here so behaviour can be adjusted without touching
business logic. Values can be overridden via environment variables (prefixed
with ADA_) which is convenient for Docker / cloud deployments.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field


SUPPORTED_PYTHON_VERSIONS = {(3, 10), (3, 11), (3, 12)}


def python_runtime_support() -> dict[str, object]:
    """Return runtime metadata and whether the current interpreter is supported."""
    version = sys.version.split()[0]
    version_tuple = tuple(sys.version_info[:2])
    supported = version_tuple in SUPPORTED_PYTHON_VERSIONS
    return {
        "version": version,
        "major": sys.version_info.major,
        "minor": sys.version_info.minor,
        "supported": supported,
        "recommended": "3.12",
        "reason": (
            "Python 3.10-3.12 is the validated range for this project. "
            "Use a compatible interpreter to avoid build issues with scientific dependencies."
            if not supported else ""
        ),
    }


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(f"ADA_{name}", default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(f"ADA_{name}", default))
    except (TypeError, ValueError):
        return default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(f"ADA_{name}", default)


@dataclass(frozen=True)
class Thresholds:
    """Numeric thresholds driving heuristics across the pipeline."""

    # -- Ingestion / performance guardrails --
    max_rows_full_load: int = _env_int("MAX_ROWS_FULL_LOAD", 500_000)
    large_dataset_rows: int = _env_int("LARGE_DATASET_ROWS", 100_000)
    sample_rows_for_heavy_ops: int = _env_int("SAMPLE_ROWS_HEAVY", 20_000)
    max_cols_wide_dataset: int = _env_int("MAX_COLS_WIDE", 200)

    # -- Cleaning --
    sparse_col_missing_frac: float = _env_float("SPARSE_MISSING_FRAC", 0.60)
    rare_category_frac: float = _env_float("RARE_CATEGORY_FRAC", 0.02)
    winsor_limits: tuple = (0.01, 0.01)
    near_zero_variance_frac: float = _env_float("NZV_FRAC", 0.01)

    # -- Schema inference --
    id_uniqueness_ratio: float = _env_float("ID_UNIQUENESS_RATIO", 0.95)
    categorical_max_unique_ratio: float = _env_float("CAT_MAX_UNIQUE_RATIO", 0.5)
    categorical_max_unique_abs: int = _env_int("CAT_MAX_UNIQUE_ABS", 50)
    text_avg_len_threshold: int = _env_int("TEXT_AVG_LEN", 30)
    high_cardinality_threshold: int = _env_int("HIGH_CARDINALITY", 100)
    datetime_parse_success_rate: float = _env_float("DT_SUCCESS_RATE", 0.6)

    # -- Outliers / anomalies --
    iqr_multiplier: float = _env_float("IQR_MULT", 1.5)
    zscore_threshold: float = _env_float("ZSCORE_THRESH", 3.0)
    isolation_contamination: float = _env_float("ISO_CONTAM", 0.05)
    min_rows_anomaly: int = _env_int("MIN_ROWS_ANOMALY", 10)

    # -- Statistics --
    normality_alpha: float = _env_float("NORMALITY_ALPHA", 0.05)
    correlation_strong: float = _env_float("CORR_STRONG", 0.7)
    correlation_moderate: float = _env_float("CORR_MODERATE", 0.4)
    confidence_level: float = _env_float("CONFIDENCE_LEVEL", 0.95)

    # -- ML --
    min_rows_ml: int = _env_int("MIN_ROWS_ML", 30)
    test_size: float = _env_float("TEST_SIZE", 0.2)
    cv_folds: int = _env_int("CV_FOLDS", 5)
    max_kmeans_k: int = _env_int("MAX_KMEANS_K", 8)
    random_state: int = _env_int("RANDOM_STATE", 42)

    # -- Data quality scoring weights (must sum to ~1.0) --
    weight_completeness: float = 0.4
    weight_uniqueness: float = 0.2
    weight_consistency: float = 0.2
    weight_validity: float = 0.2


@dataclass(frozen=True)
class UISettings:
    page_title: str = _env_str("PAGE_TITLE", "Automated Data Analyzer")
    page_icon: str = _env_str("PAGE_ICON", "⚡")
    default_theme: str = _env_str("DEFAULT_THEME", "dark")
    rows_per_page: int = _env_int("ROWS_PER_PAGE", 25)
    max_preview_rows: int = _env_int("MAX_PREVIEW_ROWS", 500)


@dataclass(frozen=True)
class AppPaths:
    """Filesystem paths used by the app and sample-data loader."""

    sample_data_path: str = _env_str("SAMPLE_DATA_PATH", "sample_data/sample_sales.csv")


@dataclass(frozen=True)
class DomainKeywords:
    """Column-name keyword banks used for lightweight domain detection."""

    keywords: dict = field(default_factory=lambda: {
        "Sales / Retail": ["revenue", "sales", "units_sold", "discount", "sku", "order",
                            "product", "category", "region", "returns", "price", "cart"],
        "Finance / Banking": ["balance", "transaction", "account", "interest", "loan",
                               "credit", "debit", "apr", "emi", "fraud", "currency"],
        "Marketing": ["campaign", "impressions", "clicks", "ctr", "conversion", "cpc",
                      "cpa", "roas", "channel", "lead", "engagement"],
        "HR / Workforce": ["employee", "salary", "attrition", "department", "tenure",
                           "performance_rating", "hire_date", "manager", "job_role"],
        "Healthcare": ["patient", "diagnosis", "treatment", "hospital", "physician",
                       "bmi", "blood_pressure", "icd", "dosage", "symptom"],
        "Manufacturing": ["machine", "downtime", "defect", "batch", "yield",
                          "production_line", "shift", "throughput", "scrap"],
        "Telecom": ["call_duration", "churn", "subscriber", "network", "data_usage",
                    "sim", "plan", "roaming"],
        "Logistics": ["shipment", "warehouse", "route", "delivery", "carrier",
                     "freight", "tracking", "eta", "fleet"],
        "Education": ["student", "grade", "gpa", "course", "enrollment", "attendance",
                     "exam", "teacher", "school"],
        "Insurance": ["policy", "premium", "claim", "underwriting", "deductible",
                     "coverage", "beneficiary"],
        "IoT / Sensor": ["sensor", "temperature", "humidity", "voltage", "signal",
                         "device_id", "reading", "telemetry"],
        "E-commerce": ["cart", "checkout", "sku", "add_to_cart", "session", "wishlist",
                       "customer_id", "order_id"],
    })


THRESH = Thresholds()
UI = UISettings()
PATHS = AppPaths()
DOMAINS = DomainKeywords()

APP_VERSION = "2.0.0"
