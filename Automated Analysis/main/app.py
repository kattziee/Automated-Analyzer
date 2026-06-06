"""
Enterprise Business Intelligence & Data Engineering Core
========================================================
Streamlit application entry point.

Run with:
    cd bi_dashboard
    streamlit run app.py
"""
from __future__ import annotations

import os
import sys
import pathlib
import textwrap
import traceback

# ── Path setup (allow imports from project root) ──────────────
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from core.ingestion import IngestionEngine
from core.cleaning import CleaningEngine
from core.analytics import AnalyticsEngine, PROPHET_AVAILABLE, STATSMODELS_AVAILABLE
from core.visualization import ChartFactory
from core.grok_client import GrokClient
from utils.helpers import df_to_csv_bytes, quality_score, get_numeric_cols, get_datetime_cols, get_categorical_cols

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise BI Core",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Automated Data Cleaner — powered by xAI Grok",
    },
)

# ── Load CSS ──────────────────────────────────────────────────
CSS_PATH = ROOT / "assets" / "style.css"
if CSS_PATH.exists():
    with open(CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
SAMPLE_DATA_PATH = ROOT.parent / "sample_data" / "sample_sales.csv"
DEFAULT_GROK_KEY = "xai-s3wnctl8zWpsPIDFRA1xqDQkcocEVFpa7J1IVMor5lsJXCaoKjeemedtN9pySFDvNJSLH58mz6lf5Ii5"

# Attempt to read from Streamlit secrets first
try:
    DEFAULT_GROK_KEY = st.secrets.get("GROK_API_KEY", DEFAULT_GROK_KEY)
except Exception:
    pass


# ══════════════════════════════════════════════════════════════
# Cached computation helpers
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def cached_ingest(file_bytes: bytes, file_name: str):
    engine = IngestionEngine()
    import io
    f = io.BytesIO(file_bytes)
    f.name = file_name
    return engine.parse_file(f)


@st.cache_data(show_spinner=False)
def cached_clean(df: pd.DataFrame):
    engine = CleaningEngine()
    return engine.run(df)


@st.cache_data(show_spinner=False)
def cached_anomalies(df: pd.DataFrame):
    engine = AnalyticsEngine()
    return engine.detect_anomalies(df)


@st.cache_data(show_spinner=False)
def cached_forecast(df: pd.DataFrame, date_col: str, value_col: str):
    engine = AnalyticsEngine()
    return engine.forecast(df, date_col, value_col)


@st.cache_data(show_spinner=False)
def load_sample_data() -> pd.DataFrame:
    """Generate or load the sample CSV."""
    if SAMPLE_DATA_PATH.exists():
        return pd.read_csv(SAMPLE_DATA_PATH)
    # Inline generation if file missing
    try:
        gen_path = ROOT.parent / "sample_data" / "generate_sample.py"
        if gen_path.exists():
            exec(compile(open(gen_path).read(), str(gen_path), "exec"),
                 {"__file__": str(gen_path)})
        if SAMPLE_DATA_PATH.exists():
            return pd.read_csv(SAMPLE_DATA_PATH)
    except Exception:
        pass
    # Minimal fallback
    import random, datetime
    rng = np.random.default_rng(42)
    n = 200
    dates = [datetime.date(2023, 1, 1) + datetime.timedelta(days=int(i)) for i in rng.integers(0, 730, n)]
    return pd.DataFrame({
        "Date": dates,
        "Region": rng.choice(["North", "South", "East", "West"], n),
        "Category": rng.choice(["Electronics", "Clothing", "Food", "Home"], n),
        "Revenue": np.round(rng.lognormal(7.5, 1.0, n), 2),
        "Units_Sold": rng.integers(10, 300, n),
        "Discount_Pct": np.round(rng.uniform(0, 30, n), 1),
        "Returns": rng.integers(0, 30, n),
    })


# ══════════════════════════════════════════════════════════════
# UI Helper components
# ══════════════════════════════════════════════════════════════

def render_metric_card(icon: str, value, label: str, accent: str = "") -> str:
    return f"""
    <div class="metric-card {accent}">
        <div class="mc-icon">{icon}</div>
        <div class="mc-value">{value}</div>
        <div class="mc-label">{label}</div>
    </div>"""


def render_quality_bar(score: float) -> str:
    color = "#34d399" if score >= 80 else "#fbbf24" if score >= 60 else "#f43f5e"
    return f"""
    <div style="margin-bottom:0.5rem">
      <span style="font-size:0.82rem;color:#94a3b8;">Data Completeness</span>
      <span style="float:right;font-weight:700;color:{color};">{score}%</span>
    </div>
    <div class="quality-bar-wrap">
      <div class="quality-bar-fill" style="width:{score}%;background:{color};"></div>
    </div>"""


def render_badge(text: str, kind: str = "live") -> str:
    return f'<span class="bi-badge badge-{kind}">{text}</span>'


def section(icon: str, title: str):
    st.markdown(
        f'<div class="section-header">{icon} {title}</div>',
        unsafe_allow_html=True,
    )


def show_chart(title: str, fig: go.Figure, key: str = ""):
    try:
        st.plotly_chart(fig, width="stretch", key=key or title)
    except TypeError:
        # Fallback for older Streamlit builds
        st.plotly_chart(fig, use_container_width=True, key=key or title)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

def build_sidebar() -> tuple:
    """
    Returns (df_raw, grok_key, use_sample).
    df_raw is None if no data loaded yet.
    """
    with st.sidebar:
        # Logo / brand
        st.markdown("""
        <div style="text-align:center;padding:1rem 0 0.5rem;">
          <div style="font-size:2.2rem;">⚡</div>
          <div style="font-size:1.1rem;font-weight:700;color:#a5b4fc;letter-spacing:-0.02em;">
            Automated Data Analysis
          </div>
          <div style="font-size:0.75rem;color:#64748b;margin-top:0.2rem;">
            Powered by xAI Grok
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        # ── Data source ────────────────────────────────────────
        st.markdown("#### 📂 Data Source")
        use_sample = st.toggle("Use built-in sample dataset", value=False)

        df_raw = None
        schema = None
        warnings_list = []

        if use_sample:
            with st.spinner("Loading sample data…"):
                try:
                    raw = load_sample_data()
                    engine = IngestionEngine()
                    import io
                    buf = io.BytesIO(raw.to_csv(index=False).encode())
                    buf.name = "sample_sales.csv"
                    df_raw, schema = engine.parse_file(buf)
                    st.success(f"✅ Sample loaded: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")
                except Exception as e:
                    st.error(f"Failed to load sample: {e}")
        else:
            uploaded = st.file_uploader(
                "Upload your dataset",
                type=["csv", "tsv", "json", "xls", "xlsx", "ods"],
                help="Supports CSV, TSV, JSON, Excel (XLS/XLSX), ODS",
            )
            if uploaded:
                with st.spinner(f"Parsing {uploaded.name}…"):
                    try:
                        file_bytes = uploaded.read()
                        import io
                        buf = io.BytesIO(file_bytes)
                        buf.name = uploaded.name
                        engine = IngestionEngine()
                        df_raw, schema = engine.parse_file(buf)
                        st.success(f"✅ Loaded: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")
                        if schema and schema.warnings:
                            with st.expander("⚠️ Parser warnings", expanded=False):
                                for w in schema.warnings:
                                    st.caption(w)
                    except Exception as e:
                        st.error(f"❌ {e}")

        # Store schema in session so tabs can access it
        if schema:
            st.session_state["schema"] = schema

        st.divider()

        # ── Grok API key ───────────────────────────────────────
        st.markdown("#### 🤖 Grok AI")
        grok_key = st.text_input(
            "xAI API Key",
            value=DEFAULT_GROK_KEY,
            type="password",
            help="Your xAI Grok API key. Pre-filled from secrets.",
        )
        model_info = "grok-3" if grok_key else "—"
        st.caption(f"Model: `{model_info}` · fallback: `grok-3-mini`")

        st.divider()

        # ── System status ──────────────────────────────────────
        st.markdown("#### 🔧 Engine Status")
        status_items = [
            ("IsolationForest", True, "Anomaly detector"),
            ("IQR Z-Score", True, "Anomaly detector"),
            ("Prophet", PROPHET_AVAILABLE, "Forecasting"),
            ("Statsmodels ETS", STATSMODELS_AVAILABLE, "Forecast fallback"),
            ("Grok AI", bool(grok_key), "Cognitive layer"),
        ]
        for name, ok, role in status_items:
            icon = "🟢" if ok else "🔴"
            color = "#34d399" if ok else "#f43f5e"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:center;margin-bottom:0.3rem;">'
                f'<span style="font-size:0.8rem;color:#94a3b8;">{icon} {name}</span>'
                f'<span style="font-size:0.7rem;color:{color};">{role}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        return df_raw, grok_key


# ══════════════════════════════════════════════════════════════
# TAB 1 — Ingest & Preview
# ══════════════════════════════════════════════════════════════

def tab_ingest(df: pd.DataFrame):
    schema = st.session_state.get("schema")

    # Hero metrics
    qs = quality_score(df)
    num_cols = get_numeric_cols(df)
    cat_cols = get_categorical_cols(df)
    dt_cols = get_datetime_cols(df)
    n_nulls = int(df.isnull().sum().sum())

    st.markdown(f"""
    <div class="metric-grid">
      {render_metric_card("📋", f"{df.shape[0]:,}", "Total Rows", "mc-accent-cyan")}
      {render_metric_card("🗂️", df.shape[1], "Columns", "mc-accent-indigo")}
      {render_metric_card("🔢", len(num_cols), "Numeric", "mc-accent-green")}
      {render_metric_card("🏷️", len(cat_cols), "Categorical", "mc-accent-amber")}
      {render_metric_card("⚠️", f"{n_nulls:,}", "Missing Values", "mc-accent-rose")}
    </div>
    {render_quality_bar(qs)}
    """, unsafe_allow_html=True)

    # Data preview
    section("👁️", "Data Preview")
    st.dataframe(df.head(50), use_container_width=True, height=320)

    col1, col2 = st.columns(2)

    # Schema table
    with col1:
        section("📐", "Schema & Null Report")
        schema_data = []
        for c in df.columns:
            schema_data.append({
                "Column": c,
                "Type": str(df[c].dtype),
                "Non-Null": int(df[c].notna().sum()),
                "Null %": f"{df[c].isna().mean()*100:.1f}%",
                "Unique": int(df[c].nunique()),
                "Sample": str(df[c].dropna().iloc[0]) if df[c].notna().any() else "—",
            })
        st.dataframe(pd.DataFrame(schema_data), use_container_width=True, height=350)

    # Numeric stats
    with col2:
        section("📊", "Descriptive Statistics")
        num_df = df.select_dtypes(include=np.number)
        if not num_df.empty:
            st.dataframe(num_df.describe().round(2), use_container_width=True, height=350)
        else:
            st.info("No numeric columns detected.")

    # Parser warnings
    if schema and schema.warnings:
        section("⚙️", "Parser Actions")
        st.markdown(
            '<div class="audit-log">' +
            "<br>".join(f"• {w}" for w in schema.warnings) +
            "</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════
# TAB 2 — Clean & Export
# ══════════════════════════════════════════════════════════════

def tab_clean(df: pd.DataFrame):
    section("🧹", "Running Cleaning Pipeline")

    with st.spinner("Applying cleaning pipeline…"):
        clean_df, report = cached_clean(df)

    # Before / After metrics
    qs_before = quality_score(df)
    qs_after = quality_score(clean_df)

    st.markdown(f"""
    <div class="metric-grid">
      {render_metric_card("📋", f"{report.rows_before:,} → {report.rows_after:,}", "Rows", "mc-accent-cyan")}
      {render_metric_card("🗂️", f"{report.cols_before} → {report.cols_after}", "Columns", "mc-accent-indigo")}
      {render_metric_card("🗑️", len(report.dropped_columns), "Cols Dropped", "mc-accent-rose")}
      {render_metric_card("🔢", len(report.imputed_numeric), "Numeric Imputed", "mc-accent-green")}
      {render_metric_card("📈", f"{qs_before}% → {qs_after}%", "Completeness", "mc-accent-amber")}
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.3, 1])

    with col1:
        section("📋", "Cleaning Audit Log")
        if report.audit_log:
            st.markdown(
                '<div class="audit-log">' +
                "<br>".join(report.audit_log) +
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.success("Dataset was already clean — no actions needed.")

    with col2:
        section("📦", "Cleaned Dataset Preview")
        st.caption(
            f"Showing all **{len(clean_df):,} rows** × **{len(clean_df.columns)} columns** "
            f"(scroll to explore)"
        )
        st.dataframe(clean_df, use_container_width=True, height=320)

        # Download
        csv_bytes = df_to_csv_bytes(clean_df)
        st.download_button(
            label=f"⬇️  Download Full Clean CSV ({len(clean_df):,} rows)",
            data=csv_bytes,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Side-by-side stats comparison
    section("🔍", "Before vs After — Descriptive Statistics")
    num_cols_before = get_numeric_cols(df)
    num_cols_after  = get_numeric_cols(clean_df)
    shared = [c for c in num_cols_before if c in num_cols_after]

    if shared:
        bc1, bc2 = st.columns(2)
        with bc1:
            st.caption("**Before cleaning**")
            st.dataframe(df[shared].describe().round(2), use_container_width=True)
        with bc2:
            st.caption("**After cleaning**")
            st.dataframe(clean_df[shared].describe().round(2), use_container_width=True)

    # Store cleaned df
    st.session_state["clean_df"] = clean_df


# ══════════════════════════════════════════════════════════════
# TAB 3 — Analytics & Visualize
# ══════════════════════════════════════════════════════════════

def tab_analytics(df: pd.DataFrame):
    charts = ChartFactory()

    # Always ensure clean_df exists — run cleaning if Tab 2 was skipped
    if "clean_df" not in st.session_state:
        with st.spinner("Preparing cleaned data for analysis…"):
            _cdf, _ = cached_clean(df)
            st.session_state["clean_df"] = _cdf

    clean_df = st.session_state["clean_df"]

    # ── Anomaly detection ─────────────────────────────────────
    section("🔍", "Multivariate Anomaly Detection")

    with st.spinner("Running dual-engine anomaly detection…"):
        anomaly_df = cached_anomalies(clean_df)

    n_anomalies = int(anomaly_df["is_anomaly"].sum())
    anomaly_rate = round(n_anomalies / max(len(anomaly_df), 1) * 100, 1)

    st.markdown(f"""
    <div class="metric-grid">
      {render_metric_card("🚨", n_anomalies, "Anomalies Flagged", "mc-accent-rose")}
      {render_metric_card("✅", len(anomaly_df) - n_anomalies, "Normal Records", "mc-accent-green")}
      {render_metric_card("📊", f"{anomaly_rate}%", "Anomaly Rate", "mc-accent-amber")}
    </div>
    """, unsafe_allow_html=True)

    if n_anomalies > 0:
        col_a, col_b = st.columns([1.4, 1])
        with col_a:
            st.caption(f"🚨 **{n_anomalies} flagged records** — hover rows for details")
            flagged = anomaly_df[anomaly_df["is_anomaly"]].drop(
                columns=["is_anomaly"], errors="ignore"
            ).sort_values("anomaly_score", ascending=False)
            st.dataframe(flagged, use_container_width=True, height=280)

        with col_b:
            num_cols = get_numeric_cols(clean_df)
            if len(num_cols) >= 2:
                x_col = st.selectbox("X axis", num_cols, index=0, key="anom_x")
                y_col = st.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1), key="anom_y")
                fig = charts.anomaly_chart(anomaly_df, x=x_col, y=y_col)
                show_chart("Anomaly Scatter", fig, "anomaly_scatter")
    else:
        st.success("✅ No anomalies detected. Your dataset looks clean!")

    st.divider()

    # ── Auto visualizations ───────────────────────────────────
    section("📊", "Auto-Generated Visualizations")

    with st.spinner("Generating schema-aware charts…"):
        auto = charts.auto_charts(clean_df)

    if auto:
        # Render in a 2-col grid
        for i in range(0, len(auto), 2):
            c1, c2 = st.columns(2)
            with c1:
                title, fig = auto[i]
                st.caption(f"**{title}**")
                show_chart(title, fig, f"auto_{i}")
            if i + 1 < len(auto):
                with c2:
                    title2, fig2 = auto[i + 1]
                    st.caption(f"**{title2}**")
                    show_chart(title2, fig2, f"auto_{i+1}")
    else:
        st.info("Upload a dataset with numeric columns to generate visualizations.")

    st.divider()

    # ── Custom chart builder ───────────────────────────────────
    section("🎨", "Custom Chart Builder")
    num_cols = get_numeric_cols(clean_df)
    cat_cols = get_categorical_cols(clean_df)
    dt_cols  = get_datetime_cols(clean_df)
    all_cols = clean_df.columns.tolist()

    chart_type = st.selectbox(
        "Chart type",
        ["Bar", "Line", "Scatter", "Histogram", "Box", "Correlation Heatmap"],
        key="custom_chart_type",
    )

    if chart_type == "Correlation Heatmap":
        if len(num_cols) >= 2:
            fig = charts.correlation_heatmap(clean_df)
            show_chart("Correlation Heatmap", fig, "custom_heatmap")
        else:
            st.warning("Need ≥ 2 numeric columns.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            x_options = cat_cols + dt_cols + num_cols if chart_type in ["Bar", "Box"] else all_cols
            x = st.selectbox("X axis", x_options or all_cols, key="cx")
        with c2:
            y = st.selectbox("Y axis", num_cols or all_cols, key="cy")
        with c3:
            color = st.selectbox("Color by (optional)", ["—"] + cat_cols, key="cc")
            color = None if color == "—" else color

        if st.button("Generate Chart", key="gen_chart"):
            try:
                if chart_type == "Bar":
                    fig = charts.bar_chart(clean_df, x=x, y=y)
                elif chart_type == "Line":
                    fig = charts.line_chart(clean_df, x=x, y=y)
                elif chart_type == "Scatter":
                    fig = charts.scatter_chart(clean_df, x=x, y=y, color=color)
                elif chart_type == "Histogram":
                    fig = charts.histogram(clean_df, col=x)
                elif chart_type == "Box":
                    fig = charts.box_chart(clean_df, x=x, y=y)
                else:
                    raise ValueError(f"Unsupported chart type: {chart_type}")
                show_chart(f"{chart_type}: {y} by {x}", fig, "custom_output")
            except Exception as e:
                st.error(f"Chart error: {e}")

    st.divider()

    # ── Forecasting ───────────────────────────────────────────
    section("🔮", "Time-Series Forecasting (30-Day Horizon)")

    method_label = "Prophet" if PROPHET_AVAILABLE else "Exponential Smoothing (ETS)"
    st.caption(f"Using **{method_label}** for forecasting")

    if not dt_cols and not num_cols:
        st.warning("No datetime or numeric columns available for forecasting.")
        return

    fc1, fc2 = st.columns(2)
    with fc1:
        date_col = st.selectbox(
            "Date / time column",
            dt_cols + [c for c in clean_df.columns if "date" in c.lower() or "time" in c.lower()],
            key="fc_date",
        ) if (dt_cols or any("date" in c.lower() for c in clean_df.columns)) else None

    with fc2:
        value_col = st.selectbox("Value to forecast", num_cols, key="fc_val") if num_cols else None

    if date_col and value_col:
        if st.button("▶  Run Forecast", key="run_forecast"):
            with st.spinner(f"Forecasting {value_col} using {method_label}…"):
                result = cached_forecast(clean_df, date_col, value_col)

            st.caption(f"ℹ️ {result.message}")
            if result.forecast_df is not None:
                fig = charts.forecast_chart(clean_df, result.forecast_df, date_col, value_col)
                show_chart("Forecast", fig, "forecast_main")
                # Store for Grok tab
                st.session_state["forecast_result"] = result
                st.session_state["forecast_cols"] = (date_col, value_col)
            else:
                st.warning(result.message)


# ══════════════════════════════════════════════════════════════
# TAB 4 — Grok AI
# ══════════════════════════════════════════════════════════════

def tab_grok(df: pd.DataFrame, grok_key: str):
    clean_df = st.session_state.get("clean_df", df)
    grok = GrokClient(grok_key)

    if not grok.is_available:
        st.error(
            "⚠️ Grok AI is not available. "
            "Please enter a valid xAI API key in the sidebar."
        )
        return

    # ── Executive Summary ─────────────────────────────────────
    section("📝", "Executive Intelligence Summary")
    st.caption("Auto-generated from your dataset using Grok AI")

    if "grok_summary" not in st.session_state:
        st.session_state["grok_summary"] = ""

    col_s, col_b = st.columns([4, 1])
    with col_b:
        regen = st.button("🔄 Regenerate", key="regen_summary")

    if regen or not st.session_state.get("grok_summary"):
        with st.spinner("Grok is analysing your dataset…"):
            summary = grok.generate_summary(clean_df)
            st.session_state["grok_summary"] = summary

    if st.session_state.get("grok_summary"):
        with st.container():
            st.markdown(
                f'<div class="glass-panel">{st.session_state["grok_summary"]}</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Anomaly Explanation ───────────────────────────────────
    section("🚨", "Anomaly Root-Cause Analysis")

    anomaly_df_full = st.session_state.get("anomaly_df_full")
    if anomaly_df_full is None:
        # Run if not cached in session
        with st.spinner("Running anomaly detection for Grok analysis…"):
            anomaly_df_full = cached_anomalies(clean_df)
            st.session_state["anomaly_df_full"] = anomaly_df_full

    flagged = anomaly_df_full[anomaly_df_full["is_anomaly"]]

    if st.button("🔍 Explain Anomalies with Grok", key="explain_anomaly"):
        with st.spinner("Grok is analysing anomalies…"):
            explanation = grok.explain_anomalies(flagged, clean_df)
            st.session_state["anomaly_explanation"] = explanation

    if st.session_state.get("anomaly_explanation"):
        st.markdown(
            f'<div class="glass-panel">{st.session_state["anomaly_explanation"]}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Forecast Explanation ──────────────────────────────────
    forecast_result = st.session_state.get("forecast_result")
    if forecast_result and forecast_result.forecast_df is not None:
        section("🔮", "Forecast Narrative")
        date_col, value_col = st.session_state.get("forecast_cols", ("", ""))

        if st.button("💬 Explain Forecast with Grok", key="explain_forecast"):
            with st.spinner("Grok is narrating the forecast…"):
                narrative = grok.explain_forecast(
                    forecast_result.forecast_df,
                    value_col,
                    forecast_result.method,
                )
                st.session_state["forecast_narrative"] = narrative

        if st.session_state.get("forecast_narrative"):
            st.markdown(
                f'<div class="glass-panel">{st.session_state["forecast_narrative"]}</div>',
                unsafe_allow_html=True,
            )

        st.divider()

    # ── Conversational NLQ ────────────────────────────────────
    section("💬", "Conversational Business Intelligence")
    st.caption("Ask anything about your data in plain English")

    # Init chat history
    if "chat_history_msgs" not in st.session_state:
        st.session_state["chat_history_msgs"] = []
    if "chat_api_history" not in st.session_state:
        st.session_state["chat_api_history"] = []

    # Display chat bubbles
    if st.session_state["chat_history_msgs"]:
        chat_html = '<div class="chat-container">'
        for msg in st.session_state["chat_history_msgs"]:
            role = msg["role"]
            content = msg["content"].replace("\n", "<br>")
            css_cls = "user" if role == "user" else "agent"
            avatar_cls = "user-av" if role == "user" else "agent-av"
            avatar_icon = "👤" if role == "user" else "⚡"
            chat_html += f"""
            <div class="chat-bubble {css_cls}">
              <div class="bubble-avatar {avatar_cls}">{avatar_icon}</div>
              <div class="bubble-text">{content}</div>
            </div>"""
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Input row
    q_col, btn_col = st.columns([5, 1])
    with q_col:
        user_input = st.text_input(
            "Ask a question",
            placeholder='e.g., "Which region has the highest average revenue?" or "Explain the discount impact"',
            key="chat_input",
            label_visibility="collapsed",
        )
    with btn_col:
        send = st.button("Send ➤", key="chat_send", use_container_width=True)

    # Quick-fire suggestions
    suggestions = [
        "Which category has the highest revenue?",
        "What's the trend in sales over time?",
        "Which region has the most anomalies?",
        "Summarize the discount impact on returns",
    ]
    st.caption("💡 Quick questions:")
    cols = st.columns(len(suggestions))
    chosen_suggestion = None
    for i, sug in enumerate(suggestions):
        with cols[i]:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                chosen_suggestion = sug

    # Process message
    query = chosen_suggestion or (user_input if send else None)
    if query and query.strip():
        st.session_state["chat_history_msgs"].append({"role": "user", "content": query})

        with st.spinner("Grok is thinking…"):
            response = grok.chat(
                user_message=query,
                df=clean_df,
                history=st.session_state["chat_api_history"],
            )

        st.session_state["chat_history_msgs"].append({"role": "assistant", "content": response})
        st.session_state["chat_api_history"].extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": response},
        ])
        st.rerun()

    # Clear chat
    if st.session_state["chat_history_msgs"]:
        if st.button("🗑️ Clear Conversation", key="clear_chat"):
            st.session_state["chat_history_msgs"] = []
            st.session_state["chat_api_history"] = []
            st.rerun()


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    # ── Header ────────────────────────────────────────────────
    st.markdown("""
    <div class="bi-header">
      <h1>⚡ Automated Data Analysis & BI</h1>
      <p>Ingest → Clean → Analyse → Visualize → Converse</p>
      <br>
      <span class="bi-badge badge-live">● LIVE</span>&nbsp;
      <span class="bi-badge badge-ai">🤖 Grok AI</span>&nbsp;
      <span class="bi-badge badge-ai">📊 Plotly</span>&nbsp;
      <span class="bi-badge badge-live">🔬 IsolationForest</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────
    df_raw, grok_key = build_sidebar()

    # ── No data state ─────────────────────────────────────────
    if df_raw is None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;">
          <div style="font-size:4rem;margin-bottom:1rem;">📂</div>
          <h2 style="color:#a5b4fc;font-size:1.5rem;font-weight:600;">No Data Loaded</h2>
          <p style="color:#64748b;max-width:440px;margin:0.5rem auto 0;">
            Upload a CSV, Excel, JSON, or ODS file using the sidebar —
            or toggle <b>Use built-in sample dataset</b> to explore with synthetic sales data.
          </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Tabs ──────────────────────────────────────────────────
    tabs = st.tabs([
        "📥 Ingest & Preview",
        "🧹 Clean & Export",
        "📊 Analytics & Visualize",
        "🤖 Grok AI",
    ])

    with tabs[0]:
        try:
            tab_ingest(df_raw)
        except Exception as e:
            st.error(f"Ingest tab error: {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())

    with tabs[1]:
        try:
            tab_clean(df_raw)
        except Exception as e:
            st.error(f"Cleaning tab error: {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())

    with tabs[2]:
        try:
            tab_analytics(df_raw)
        except Exception as e:
            st.error(f"Analytics tab error: {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())

    with tabs[3]:
        try:
            tab_grok(df_raw, grok_key)
        except Exception as e:
            st.error(f"Grok AI tab error: {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
