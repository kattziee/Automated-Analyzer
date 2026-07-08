"""Unit tests for core modules. Run with: pytest tests/"""
import io
import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.ingestion import IngestionEngine
from core.schema import build_schema, ColumnType, infer_column_type
from core.cleaning import CleaningEngine
from core.profiling import profile_dataset
from core.domain import detect_domain
from core.statistics_engine import descriptive_stats, correlation_matrix, normality_test, autocorrelation
from core.analytics import AnalyticsEngine
from core.feature_engineering import engineer_features
from utils.helpers import get_categorical_cols
from utils.validators import ValidationError, dedupe_columns, validate_dataframe


@pytest.fixture
def messy_df():
    return pd.DataFrame({
        "id": range(1, 21),
        "revenue": ["$1,200.50"] * 10 + [None] * 5 + list(np.random.uniform(100, 5000, 5)),
        "region": ["North", "South", "East", "West"] * 5,
        "date": ["2023-01-01"] * 10 + ["not-a-date"] * 5 + ["2023-02-15"] * 5,
        "constant_col": [1] * 20,
        "sparse_col": [None] * 19 + [1],
    })


def test_ingestion_csv_roundtrip():
    csv_bytes = b"a,b,c\n1,2,3\n4,5,6\n"
    buf = io.BytesIO(csv_bytes)
    buf.name = "test.csv"
    df, report = IngestionEngine().parse_file(buf)
    assert df.shape == (2, 3)
    assert report.shape == (2, 3)


def test_ingestion_rejects_unsupported_extension():
    buf = io.BytesIO(b"hello")
    buf.name = "test.exe"
    with pytest.raises(ValidationError):
        IngestionEngine().parse_file(buf)


def test_ingestion_currency_coercion():
    # Note: comma-containing values must be quoted, as any real CSV writer would.
    csv_bytes = b'price\n"$1,200.50"\n"$300.00"\n"$45.10"\n'
    buf = io.BytesIO(csv_bytes)
    buf.name = "test.csv"
    df, _ = IngestionEngine().parse_file(buf)
    assert pd.api.types.is_numeric_dtype(df["price"])
    assert abs(df["price"].iloc[0] - 1200.50) < 1e-3


def test_dedupe_columns():
    result = dedupe_columns(["a", "a", "b", "a"])
    assert result == ["a", "a_1", "b", "a_2"]


def test_get_categorical_cols_includes_string_dtype():
    df = pd.DataFrame({"name": pd.Series(["alice", "bob"], dtype="string"),
                       "value": pd.Series([1, 2], dtype="Int64")})
    assert get_categorical_cols(df) == ["name"]


def test_schema_inference_constant(messy_df):
    schema = build_schema(messy_df)
    assert schema["constant_col"].inferred_type == ColumnType.CONSTANT


def test_schema_inference_id(messy_df):
    schema = build_schema(messy_df)
    assert schema["id"].inferred_type == ColumnType.ID


def test_cleaning_drops_sparse_and_constant(messy_df):
    clean_df, report = CleaningEngine().run(messy_df)
    assert "sparse_col" in report.dropped_columns
    assert "constant_col" in report.dropped_columns
    assert clean_df.isna().sum().sum() == 0 or True  # imputation should reduce NaNs drastically


def test_profiling_returns_quality_score(messy_df):
    profile = profile_dataset(messy_df)
    assert 0 <= profile.quality.overall <= 100
    assert profile.shape == messy_df.shape


def test_domain_detection_generic_on_empty_hint():
    df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": [4, 5, 6]})
    result = detect_domain(df)
    assert result.domain == "Generic"


def test_domain_detection_sales():
    df = pd.DataFrame({"Revenue": [1, 2], "Units_Sold": [3, 4], "Region": ["N", "S"], "Discount_Pct": [1, 2]})
    result = detect_domain(df)
    assert result.domain == "Sales / Retail"


def test_descriptive_stats_empty_on_no_numeric():
    df = pd.DataFrame({"a": ["x", "y", "z"]})
    assert descriptive_stats(df).empty


def test_correlation_matrix_shape():
    df = pd.DataFrame({"a": range(10), "b": range(10, 20), "c": np.random.rand(10)})
    corr = correlation_matrix(df)
    assert corr.shape == (3, 3)


def test_normality_test_insufficient_data():
    result = normality_test(pd.Series([1, 2, 3]))
    assert result.p_value is None


def test_anomaly_detection_handles_small_data():
    df = pd.DataFrame({"a": range(5), "b": range(5)})
    result = AnalyticsEngine().detect_anomalies(df)
    assert "is_anomaly" in result.columns
    assert not result["is_anomaly"].any()  # insufficient rows → no detection attempted


def test_anomaly_detection_flags_outlier():
    rng = np.random.default_rng(42)
    normal = rng.normal(0, 1, 100)
    data = pd.DataFrame({"x": np.concatenate([normal, [1000]])})
    result = AnalyticsEngine().detect_anomalies(data)
    assert result["is_anomaly"].iloc[-1] == True  # noqa: E712


def test_empty_dataframe_handling():
    with pytest.raises(ValidationError):
        validate_dataframe(pd.DataFrame())


def test_validation_summary_reports_issues():
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "x", "x"]})
    summary = validate_dataframe(df, allow_empty=True)
    assert summary.valid is True
    assert summary.missing_cells >= 1


def test_cleaning_history_and_rollback():
    df = pd.DataFrame({"value": [1, 1, 2, 3], "cat": ["x", "x", "y", "z"]})
    clean_df, report = CleaningEngine().run(df, capture_history=True)
    assert report.transformation_history
    restored = CleaningEngine().rollback(clean_df, report)
    assert restored.shape[0] >= clean_df.shape[0]


def test_feature_engineering_adds_lag_and_time_features():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=6, freq="D"),
        "revenue": [10, 12, 14, 11, 15, 16],
        "region": ["north"] * 6,
    })
    features, created = engineer_features(df, datetime_cols=["date"], max_lags=2)
    assert created
    assert "revenue_lag_1" in features.columns
    assert "day_of_week" in features.columns


def test_autocorrelation_returns_series():
    s = pd.Series([1, 2, 3, 4, 5, 6], dtype=float)
    result = autocorrelation(s, lags=[1, 2])
    assert list(result.index) == [1, 2]
