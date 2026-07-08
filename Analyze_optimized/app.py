"""
app.py — Automated Data Analyzer, Streamlit entry point.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import io
import sys
import pathlib
import traceback

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from config import UI, APP_VERSION, THRESH, PATHS, python_runtime_support
from utils.validators import ValidationError
from utils.helpers import (
    df_to_csv_bytes, quality_score,
    get_numeric_cols, get_categorical_cols, get_datetime_cols,
)
from core.ingestion import IngestionEngine
from core.profiling import profile_dataset
from core.domain import detect_domain
from core.cleaning import CleaningEngine
from core.schema import ColumnType
from core.statistics_engine import (
    descriptive_stats, normality_test, correlation_matrix, significant_correlations,
    t_test, anova_test, chi_square_test, confidence_interval,
)
from core.analytics import AnalyticsEngine
from core.ml_engine import MLEngine, detect_task
from core.visualization import ChartFactory, insight_for
from core.insights import full_report
from core.export import to_excel_bytes, to_html_dashboard, to_pdf_summary
from core.llm_client import LLMClient

st.set_page_config(page_title=UI.page_title, page_icon=UI.page_icon, layout="wide",
                    initial_sidebar_state="expanded")

charts = ChartFactory()
ml = MLEngine()
analytics = AnalyticsEngine()


def initialize_session_state() -> None:
    """Create stable defaults for stateful UI elements."""
    defaults = {
        "schema_report": None,
        "clean_df": None,
        "last_loaded_name": None,
        "recent_sources": [],
        "global_search": "",
        "theme": "dark",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    if "theme" not in st.session_state or st.session_state["theme"] not in {"dark", "light"}:
        try:
            theme = st.query_params.get("theme", ["dark"])[0]
            st.session_state["theme"] = theme if theme in {"dark", "light"} else "dark"
        except Exception:
            st.session_state["theme"] = "dark"


initialize_session_state()


# ══════════════════════════════════════════════════════════════
# Cached compute
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def cached_ingest(file_bytes: bytes, file_name: str):
    engine = IngestionEngine()
    buf = io.BytesIO(file_bytes)
    buf.name = file_name
    return engine.parse_file(buf)


@st.cache_data(show_spinner=False)
def cached_profile(df: pd.DataFrame):
    return profile_dataset(df)


@st.cache_data(show_spinner=False)
def cached_domain(df: pd.DataFrame):
    return detect_domain(df)


@st.cache_data(show_spinner=False)
def cached_clean(df: pd.DataFrame):
    return CleaningEngine().run(df)


@st.cache_data(show_spinner=False)
def cached_anomalies(df: pd.DataFrame):
    return analytics.detect_anomalies(df)


@st.cache_data(show_spinner=False)
def cached_forecast(df: pd.DataFrame, date_col: str, value_col: str):
    return analytics.forecast(df, date_col, value_col)


@st.cache_data(show_spinner=False)
def cached_ml(df: pd.DataFrame, target: str):
    return ml.auto_train(df, target)


@st.cache_data(show_spinner=False)
def load_sample_data() -> pd.DataFrame:
    sample_path = pathlib.Path(PATHS.sample_data_path)
    if not sample_path.is_absolute():
        sample_path = ROOT / sample_path
    if sample_path.exists():
        return pd.read_csv(sample_path)
    rng = np.random.default_rng(THRESH.random_state)
    n = 300
    return pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "Region": rng.choice(["North", "South", "East", "West"], n),
        "Category": rng.choice(["Electronics", "Clothing", "Food", "Home"], n),
        "Revenue": np.round(rng.lognormal(7.5, 1.0, n), 2),
        "Units_Sold": rng.integers(10, 300, n),
        "Discount_Pct": np.round(rng.uniform(0, 30, n), 1),
        "Returns": rng.integers(0, 30, n),
    })


# ══════════════════════════════════════════════════════════════
# UI helpers
# ══════════════════════════════════════════════════════════════

def section(icon: str, title: str):
    st.markdown(f'<div class="section-header">{icon} {title}</div>', unsafe_allow_html=True)


def apply_theme_styles() -> None:
    theme = st.session_state.get("theme", "dark")
    if theme == "light":
        palette = {
            "bg": "#f5f7fb",
            "panel": "rgba(255,255,255,0.94)",
            "card": "#ffffff",
            "border": "rgba(15, 23, 42, 0.08)",
            "text": "#0f172a",
            "muted": "#475569",
            "accent": "#2563eb",
            "accent2": "#0ea5e9",
            "shadow": "0 16px 40px rgba(15, 23, 42, 0.06)",
        }
    else:
        palette = {
            "bg": "#07111f",
            "panel": "rgba(8, 15, 30, 0.9)",
            "card": "rgba(11, 18, 35, 0.9)",
            "border": "rgba(148, 163, 184, 0.16)",
            "text": "#f8fafc",
            "muted": "#cbd5e1",
            "accent": "#2563eb",
            "accent2": "#38bdf8",
            "shadow": "0 16px 40px rgba(2, 6, 23, 0.28)",
        }

    st.markdown(f"""
    <style>
    :root {{
        --bg-base: {palette['bg']};
        --bg-panel: {palette['panel']};
        --bg-card: {palette['card']};
        --border: {palette['border']};
        --text-primary: {palette['text']};
        --text-muted: {palette['muted']};
        --accent: {palette['accent']};
        --accent-2: {palette['accent2']};
        --shadow: {palette['shadow']};
    }}
    .stApp {{ background: radial-gradient(circle at top left, color-mix(in srgb, var(--accent) 14%, transparent), transparent 26%), linear-gradient(180deg, color-mix(in srgb, var(--bg-base) 94%, white 6%), var(--bg-base)); }}
    .block-container {{ padding-top: 1rem; padding-bottom: 2rem; }}
    .sticky-header {{ position: sticky; top: 0; z-index: 1000; backdrop-filter: blur(18px); background: color-mix(in srgb, var(--bg-panel) 94%, transparent); border: 1px solid var(--border); border-radius: 16px; padding: 1rem 1.1rem; margin-bottom: 1rem; box-shadow: var(--shadow); }}
    .header-title {{ font-size: 1.4rem; font-weight: 700; letter-spacing: -0.02em; color: var(--text-primary); }}
    .header-subtitle {{ color: var(--text-muted); font-size: 0.92rem; margin-top: 0.25rem; }}
    .hero-shell {{ background: linear-gradient(135deg, color-mix(in srgb, var(--accent) 16%, var(--bg-panel)), color-mix(in srgb, var(--accent-2) 12%, var(--bg-panel))); border: 1px solid var(--border); border-radius: 20px; padding: 1.3rem 1.4rem; box-shadow: var(--shadow); margin-bottom: 1rem; }}
    .hero-title {{ font-size: 1.55rem; font-weight: 700; letter-spacing: -0.02em; color: var(--text-primary); margin-bottom: 0.3rem; }}
    .hero-text {{ color: var(--text-muted); font-size: 0.96rem; line-height: 1.65; }}
    .glass-panel {{ background: var(--bg-panel); border: 1px solid var(--border); border-radius: 16px; padding: 1rem 1.1rem; box-shadow: var(--shadow); margin-bottom: 1rem; }}
    .section-header {{ font-size: 0.95rem; font-weight: 600; color: var(--accent); margin: 1rem 0 0.7rem; padding-bottom: 0.4rem; border-bottom: 1px solid var(--border); letter-spacing: 0.01em; }}
    .kpi-card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 14px; padding: 0.95rem 1rem; box-shadow: var(--shadow); min-height: 110px; }}
    .kpi-label {{ font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); }}
    .kpi-value {{ font-size: 1.28rem; font-weight: 700; color: var(--text-primary); margin-top: 0.3rem; }}
    .kpi-delta {{ font-size: 0.84rem; color: var(--accent-2); margin-top: 0.3rem; }}
    .sidebar-brand {{ padding: 0.4rem 0.2rem 0.9rem; }}
    .sidebar-brand .title {{ font-size: 1.05rem; font-weight: 700; color: var(--text-primary); }}
    .sidebar-brand .subtitle {{ font-size: 0.8rem; color: var(--text-muted); margin-top: 0.2rem; }}
    .app-badge {{ display: inline-flex; align-items: center; justify-content: center; width: 2.3rem; height: 2.3rem; border-radius: 999px; background: linear-gradient(135deg, var(--accent), var(--accent-2)); color: white; font-weight: 700; margin-right: 0.65rem; box-shadow: 0 10px 24px rgba(37, 99, 235, 0.18); }}
    .empty-state {{ background: var(--bg-panel); border: 1px dashed var(--border); border-radius: 16px; padding: 1.2rem; text-align: center; color: var(--text-muted); }}
    .empty-state h4 {{ color: var(--text-primary); margin: 0.35rem 0; }}
    .empty-state .icon {{ font-size: 1.6rem; margin-bottom: 0.25rem; }}
    .error-state {{ background: color-mix(in srgb, #ef4444 12%, var(--bg-panel)); border: 1px solid color-mix(in srgb, #ef4444 32%, var(--border)); border-radius: 16px; padding: 1rem 1.1rem; color: var(--text-primary); }}
    .progress-shell {{ height: 8px; border-radius: 999px; overflow: hidden; background: rgba(148,163,184,0.22); margin-top: 0.7rem; }}
    .progress-shell > div {{ height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--accent), var(--accent-2)); }}
    .stTabs [data-baseweb="tab-list"] {{ background: var(--bg-panel); border: 1px solid var(--border); border-radius: 999px; padding: 0.15rem; }}
    .stTabs [aria-selected="true"] {{ background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important; color: #fff !important; border-radius: 999px !important; box-shadow: 0 10px 24px rgba(37,99,235,0.18) !important; }}
    .stButton > button {{ background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important; color: white !important; border: none !important; border-radius: 999px !important; font-weight: 600 !important; box-shadow: 0 8px 18px rgba(37,99,235,0.16) !important; }}
    .stDownloadButton > button {{ background: linear-gradient(135deg, #059669, #0f766e) !important; color: white !important; border: none !important; border-radius: 999px !important; }}
    .stTextInput > div > div > input, .stSelectbox > div > div > div, .stNumberInput > div > div > input {{ border-radius: 10px !important; }}
    div[data-testid="stDataFrame"] {{ border-radius: 14px; overflow: hidden; border: 1px solid var(--border); }}
    section[data-testid="stSidebar"] {{ background: color-mix(in srgb, var(--bg-base) 88%, var(--bg-panel)); border-right: 1px solid var(--border) !important; }}
    </style>
    """, unsafe_allow_html=True)


def persist_theme(theme: str) -> None:
    st.session_state["theme"] = theme
    try:
        st.query_params["theme"] = theme
    except Exception:
        pass


def render_theme_toggle() -> None:
    theme = st.session_state.get("theme", "dark")
    label = "☀️ Light mode" if theme == "dark" else "🌙 Dark mode"
    if st.button(label, key="theme_toggle", use_container_width=True):
        persist_theme("light" if theme == "dark" else "dark")
        st.rerun()


def render_search_input() -> None:
    st.text_input("Search workspace", key="global_search", placeholder="Search columns or values…")


def render_breadcrumbs(items: list[str]) -> None:
    if not items:
        return
    crumb = " / ".join(f"<span class='chip'>{item}</span>" for item in items)
    st.markdown(f"<div class='page-enter' style='margin-bottom:0.75rem;'>{crumb}</div>", unsafe_allow_html=True)


def metric_row(items: list[tuple[str, str, str]]):
    cols = st.columns(len(items))
    for c, (label, value, delta) in zip(cols, items):
        with c:
            delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{value}</div>
              {delta_html}
            </div>
            """, unsafe_allow_html=True)


def render_landing_page() -> None:
    st.markdown("""
    <div class="hero-shell page-enter">
      <div class="hero-title">A refined analytics workspace</div>
      <div class="hero-text">A clean, executive-ready environment for ingesting, profiling, cleaning, modeling, and explaining your data with clarity and momentum.</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1.4, 0.9], gap="large")
    with c1:
        st.markdown("""
        <div class="glass-panel page-enter">
          <div style="display:flex; justify-content:space-between; align-items:center; gap:0.5rem; flex-wrap:wrap;">
            <div>
              <div style="font-size:0.86rem; letter-spacing:0.18em; text-transform:uppercase; color:var(--accent);">Quick start</div>
              <div style="font-size:1.15rem; font-weight:700; margin-top:0.25rem;">Bring your data in and launch the workspace</div>
            </div>
            <span class="chip">Live</span>
          </div>
          <div style="margin-top:0.8rem;">
            <div class="skeleton-block" style="height:10px; width:72%; margin-bottom:8px;"></div>
            <div class="skeleton-block" style="height:10px; width:92%; margin-bottom:8px;"></div>
            <div class="skeleton-block" style="height:10px; width:56%;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop a dataset to begin", type=["csv", "tsv", "json", "xls", "xlsx", "ods", "parquet"], label_visibility="collapsed")
        if uploaded:
            handle_uploaded_file(uploaded)

    with c2:
        st.markdown("""
        <div class="glass-panel page-enter">
          <div style="font-size:0.86rem; letter-spacing:0.18em; text-transform:uppercase; color:var(--accent);">Workflow highlights</div>
          <ul style="margin:0.6rem 0 0 1rem; color:var(--text-muted); line-height:1.7;">
            <li>Profile and validate entire datasets in seconds</li>
            <li>Clean, impute, and standardize with one click</li>
            <li>Explore charts, forecasts, and ML suggestions</li>
            <li>Export polished summaries for stakeholders</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    cards = [
        ("📥", "Ingest", "Support for CSV, Excel, JSON, ODS, and Parquet files."),
        ("🧹", "Clean", "Automated deduplication, sparsity handling, and imputation."),
        ("📊", "Visualize", "Schema-aware charts and interactive analysis views."),
        ("🤖", "Model", "AutoML and forecasting for business-ready predictions."),
    ]
    cols = st.columns(4)
    for col, (icon, title, text) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div class="glass-panel page-enter">
              <div style="font-size:1.2rem;">{icon}</div>
              <div style="font-weight:700; margin-top:0.35rem;">{title}</div>
              <div style="color:var(--text-muted); font-size:0.9rem; margin-top:0.25rem;">{text}</div>
            </div>
            """, unsafe_allow_html=True)

    recent = st.session_state.get("recent_sources", [])
    if recent:
        st.markdown("""
        <div class="section-header">🕘 Recent workspace items</div>
        """, unsafe_allow_html=True)
        recent_cols = st.columns(min(3, len(recent)))
        for col, source in zip(recent_cols, recent[-3:]):
            with col:
                st.markdown(f"""
                <div class="glass-panel">
                  <div style="font-weight:700;">{source}</div>
                  <div style="color:var(--text-muted); font-size:0.86rem; margin-top:0.25rem;">Loaded into the analytics workspace</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("""
    <div class="empty-state page-enter">
      <div class="icon">📂</div>
      <h4>No dataset loaded yet</h4>
      <p>Use the sidebar or the upload area above to load a sample or your own file and the workspace will spring to life.</p>
    </div>
    """, unsafe_allow_html=True)


def render_empty_state(title: str, body: str, icon: str = "📦") -> None:
    st.markdown(f"""
    <div class="empty-state page-enter">
      <div class="icon">{icon}</div>
      <h4>{title}</h4>
      <p>{body}</p>
    </div>
    """, unsafe_allow_html=True)


def show_chart_with_insight(title: str, fig, chart_type: str, key: str, **insight_cols):
    st.caption(f"**{title}**")
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)
    with st.expander("💡 Interpretation", expanded=False):
        st.write(insight_for(chart_type, insight_cols.pop("df"), **insight_cols) if "df" in insight_cols else
                 "Review the chart for patterns worth investigating.")


def error_box(context: str, exc: Exception):
    st.markdown(f"""
    <div class="error-state">
      <div style="font-weight:700; margin-bottom:0.3rem;">⚠️ {context}</div>
      <div>{exc}</div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("Technical details"):
        st.code(traceback.format_exc())


def persist_dataset_state(df: pd.DataFrame, report, source_name: str) -> None:
    """Store the dataset and its schema metadata in Streamlit session state."""
    st.session_state["schema_report"] = report
    st.session_state["clean_df"] = None
    st.session_state["last_loaded_name"] = source_name
    recent = st.session_state.get("recent_sources", [])
    if source_name not in recent:
        recent.append(source_name)
    st.session_state["recent_sources"] = recent[-6:]


def handle_uploaded_file(uploaded) -> tuple[pd.DataFrame | None, object | None]:
    with st.spinner(f"Parsing {uploaded.name}…"):
        try:
            df_raw, report = cached_ingest(uploaded.getvalue(), uploaded.name)
            persist_dataset_state(df_raw, report, uploaded.name)
            try:
                st.toast(f"Loaded {uploaded.name}")
            except Exception:
                pass
            st.success(f"✅ {df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")
            if report and report.warnings:
                with st.expander("⚠️ Parser notes"):
                    for w in report.warnings[:15]:
                        st.caption(f"• {w}")
            return df_raw, report
        except ValidationError as e:
            st.error(str(e))
            return None, None
        except Exception as e:
            st.error(f"Unexpected parsing failure: {e}")
            return None, None


# ══════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════

def build_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
          <div style="display:flex;align-items:center;">
            <div class="app-badge">{UI.page_icon}</div>
            <div>
              <div class="title">{UI.page_title}</div>
              <div class="subtitle">v{APP_VERSION}</div>
            </div>
          </div>
        </div>
        """.format(UI=UI, APP_VERSION=APP_VERSION), unsafe_allow_html=True)

        render_theme_toggle()
        st.divider()

        st.markdown("#### 📂 Data Source")
        use_sample = st.toggle("Use built-in sample dataset", value=False, key="use_sample_data")

        df_raw, report = None, None
        if use_sample:
            try:
                raw = load_sample_data()
                buf = io.BytesIO(raw.to_csv(index=False).encode())
                buf.name = "sample_sales.csv"
                df_raw, report = IngestionEngine().parse_file(buf)
                persist_dataset_state(df_raw, report, "sample_sales.csv")
                st.success(f"✅ {df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")
            except ValidationError as e:
                st.error(str(e))
        else:
            uploaded = st.file_uploader("Upload dataset", type=["csv", "tsv", "json", "xls", "xlsx", "ods", "parquet"])
            if uploaded:
                df_raw, report = handle_uploaded_file(uploaded)

        st.divider()
        st.markdown("#### 🤖 AI Enrichment")
        api_key = st.text_input("API Key (OpenAI-compatible)", type="password",
                                 help="Never stored or hardcoded. Leave blank to use the built-in "
                                      "rule-based insight engine only.")
        st.caption("Rule-based insights always work without a key.")

        st.divider()
        st.markdown("#### 🔧 Status")
        for name, ok in [("scikit-learn", True), ("statsmodels", True), ("plotly", True),
                          ("AI enrichment", bool(api_key))]:
            icon = "🟢" if ok else "⚪"
            st.markdown(f'<span style="font-size:0.8rem;color:var(--text-muted);">{icon} {name}</span>',
                        unsafe_allow_html=True)

        return df_raw, api_key


# ══════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════

def filter_dataset_for_view(df: pd.DataFrame) -> pd.DataFrame:
    query = str(st.session_state.get("global_search", "")).strip().lower()
    if not query:
        return df
    mask = np.column_stack([
        df[col].astype(str).str.contains(query, case=False, na=False)
        for col in df.columns
    ])
    return df[mask.any(axis=1)]


def tab_overview(df: pd.DataFrame):
    try:
        profile = cached_profile(df)
        domain = cached_domain(df)
    except Exception as e:
        return error_box("Profiling failed", e)

    filtered_df = filter_dataset_for_view(df)
    metric_row([
        ("Rows", f"{profile.shape[0]:,}", None),
        ("Columns", str(profile.shape[1]), None),
        ("Memory", profile.memory_human, None),
        ("Quality Score", f"{profile.quality.overall}/100", None),
        ("Domain", domain.domain, f"{domain.confidence*100:.0f}% conf." if domain.domain != "Generic" else None),
    ])

    section("👁️", "Data Preview")
    n_preview = min(UI.max_preview_rows, len(filtered_df))
    if filtered_df.empty:
        render_empty_state("No rows matched your search", "Try a broader term to widen the selection.", icon="🔎")
    else:
        st.dataframe(filtered_df.head(n_preview), use_container_width=True, height=320)
        st.caption(f"Showing first {n_preview:,} of {len(filtered_df):,} rows.")

    col1, col2 = st.columns(2)
    with col1:
        section("📐", "Schema")
        rows = [{"Column": name, "Type": p.inferred_type.value, "Dtype": p.dtype,
                 "Unique": p.n_unique, "Missing %": p.missing_pct} for name, p in profile.schema.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=340)
    with col2:
        section("📊", "Quality Breakdown")
        q = profile.quality
        st.dataframe(pd.DataFrame([
            {"Dimension": "Completeness", "Score": q.completeness},
            {"Dimension": "Uniqueness", "Score": q.uniqueness},
            {"Dimension": "Consistency", "Score": q.consistency},
            {"Dimension": "Validity", "Score": q.validity},
        ]), use_container_width=True, height=180)
        if domain.domain != "Generic":
            st.info(f"Matched keywords: {', '.join(domain.matched_keywords)}")
        num_cols = profile.numeric_cols
        if num_cols:
            st.dataframe(df[num_cols].describe().round(2), use_container_width=True, height=180)

    report = st.session_state.get("schema_report")
    if report and report.warnings:
        section("⚙️", "Parser Actions")
        st.markdown('<div class="audit-log">' + "<br>".join(f"• {w}" for w in report.warnings) + "</div>",
                    unsafe_allow_html=True)


def tab_clean(df: pd.DataFrame):
    try:
        clean_df, rep = cached_clean(df)
    except Exception as e:
        return error_box("Cleaning failed", e)

    metric_row([
        ("Rows", f"{rep.rows_before:,} → {rep.rows_after:,}", None),
        ("Columns", f"{rep.cols_before} → {rep.cols_after}", None),
        ("Duplicates Removed", str(rep.duplicate_rows_removed), None),
        ("Cols Dropped", str(len(rep.dropped_columns)), None),
        ("Completeness", f"{quality_score(df)}% → {quality_score(clean_df)}%", None),
    ])

    col1, col2 = st.columns([1.3, 1])
    with col1:
        section("📋", "Audit Log")
        if rep.audit_log:
            st.markdown('<div class="audit-log">' + "<br>".join(rep.audit_log) + "</div>", unsafe_allow_html=True)
        else:
            st.success("Dataset was already clean.")
    with col2:
        section("📦", "Cleaned Preview")
        st.dataframe(clean_df.head(UI.max_preview_rows), use_container_width=True, height=320)
        st.download_button("⬇️ Download Cleaned CSV", df_to_csv_bytes(clean_df),
                           "cleaned_dataset.csv", "text/csv", use_container_width=True)

    st.session_state["clean_df"] = clean_df


def tab_statistics(df: pd.DataFrame):
    num_cols, cat_cols = get_numeric_cols(df), get_categorical_cols(df)

    section("📊", "Descriptive Statistics")
    desc = descriptive_stats(df)
    if not desc.empty:
        st.dataframe(desc, use_container_width=True)
    else:
        st.info("No numeric columns available.")

    section("🔗", "Correlation Analysis")
    method = st.radio("Method", ["pearson", "spearman"], horizontal=True, key="corr_method")
    corr = correlation_matrix(df, method)
    if not corr.empty:
        show_chart_with_insight("Correlation Matrix", charts.correlation_heatmap(df), "correlation",
                                "stat_corr", df=df)
        sig = significant_correlations(df, method)
        if sig:
            st.dataframe(pd.DataFrame(sig), use_container_width=True)
    else:
        st.info("Need ≥ 2 numeric columns.")

    st.divider()
    section("🧪", "Hypothesis Testing")
    test_type = st.selectbox("Test", ["Normality", "T-Test (2 groups)", "ANOVA (3+ groups)",
                                       "Chi-Square (categorical)", "Confidence Interval"])
    try:
        if test_type == "Normality" and num_cols:
            col = st.selectbox("Column", num_cols, key="norm_col")
            res = normality_test(df[col])
            st.write(f"**{res.test_name}** — statistic={res.statistic}, p={res.p_value}")
            st.info(res.interpretation)
        elif test_type == "T-Test (2 groups)" and num_cols and cat_cols:
            c1, c2 = st.columns(2)
            num_c = c1.selectbox("Numeric", num_cols, key="ttest_num")
            grp_c = c2.selectbox("Group", cat_cols, key="ttest_grp")
            res = t_test(df, num_c, grp_c)
            st.write(f"**{res.test_name}** — statistic={res.statistic}, p={res.p_value}")
            st.info(res.interpretation)
        elif test_type == "ANOVA (3+ groups)" and num_cols and cat_cols:
            c1, c2 = st.columns(2)
            num_c = c1.selectbox("Numeric", num_cols, key="anova_num")
            grp_c = c2.selectbox("Group", cat_cols, key="anova_grp")
            res = anova_test(df, num_c, grp_c)
            st.write(f"**{res.test_name}** — statistic={res.statistic}, p={res.p_value}")
            st.info(res.interpretation)
        elif test_type == "Chi-Square (categorical)" and len(cat_cols) >= 2:
            c1, c2 = st.columns(2)
            a = c1.selectbox("Column A", cat_cols, key="chi_a")
            b = c2.selectbox("Column B", [c for c in cat_cols if c != a], key="chi_b")
            res = chi_square_test(df, a, b)
            st.write(f"**{res.test_name}** — statistic={res.statistic}, p={res.p_value}")
            st.info(res.interpretation)
        elif test_type == "Confidence Interval" and num_cols:
            col = st.selectbox("Column", num_cols, key="ci_col")
            ci = confidence_interval(df[col])
            st.write(f"Mean = **{ci['mean']}**, 95% CI = [{ci['lower']}, {ci['upper']}]")
        else:
            st.info("Insufficient column types available for this test.")
    except Exception as e:
        error_box("Test failed", e)


def tab_visualize(df: pd.DataFrame):
    section("📊", "Auto-Generated Visualizations")
    with st.spinner("Selecting schema-aware charts…"):
        auto = charts.auto_charts(df)
    if auto:
        for i in range(0, len(auto), 2):
            c1, c2 = st.columns(2)
            with c1:
                title, fig, ctype = auto[i]
                show_chart_with_insight(title, fig, ctype, f"auto_{i}", df=df)
            if i + 1 < len(auto):
                with c2:
                    title2, fig2, ctype2 = auto[i + 1]
                    show_chart_with_insight(title2, fig2, ctype2, f"auto_{i+1}", df=df)
    else:
        st.info("Upload a dataset with numeric columns to auto-generate charts.")

    st.divider()
    section("🎨", "Custom Chart Builder")
    num_cols, cat_cols, dt_cols = get_numeric_cols(df), get_categorical_cols(df), get_datetime_cols(df)
    all_cols = df.columns.tolist()

    chart_type = st.selectbox("Chart type", [
        "Bar", "Grouped Bar", "Stacked Bar", "Line", "Area", "Scatter", "Bubble", "Histogram",
        "KDE", "Box", "Violin", "ECDF", "Q-Q Plot", "Pie", "Donut", "Treemap", "Sunburst",
        "Count Plot", "Correlation Heatmap", "Pair Plot", "Parallel Coordinates", "Radar",
        "Hexbin", "Rolling Average",
    ])

    try:
        if chart_type == "Correlation Heatmap":
            show_chart_with_insight("Correlation Heatmap", charts.correlation_heatmap(df), "correlation",
                                    "custom_corr", df=df)
        elif chart_type in ("Histogram", "KDE", "ECDF", "Q-Q Plot", "Count Plot"):
            col = st.selectbox("Column", (num_cols if chart_type != "Count Plot" else cat_cols) or all_cols)
            fig = {"Histogram": charts.histogram, "KDE": charts.kde, "ECDF": charts.ecdf,
                   "Q-Q Plot": charts.qq_plot, "Count Plot": charts.count_plot}[chart_type](df, col)
            show_chart_with_insight(f"{chart_type}: {col}", fig, "histogram", "custom_1col", df=df, col=col)
        elif chart_type in ("Pie", "Donut", "Treemap", "Sunburst"):
            names = st.selectbox("Category", cat_cols or all_cols)
            values = st.selectbox("Value (optional)", ["— count —"] + num_cols)
            values = None if values == "— count —" else values
            if chart_type in ("Pie", "Donut"):
                fig = charts.pie_chart(df, names, values) if chart_type == "Pie" else charts.donut_chart(df, names, values)
            else:
                path_cols = st.multiselect("Hierarchy path", cat_cols, default=[names])
                fig = (charts.treemap if chart_type == "Treemap" else charts.sunburst)(df, path_cols or [names], values or num_cols[0])
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Pair Plot":
            cols = st.multiselect("Numeric columns (max 5)", num_cols, default=num_cols[:3])
            color = st.selectbox("Color by (optional)", ["—"] + cat_cols)
            color = None if color == "—" else color
            if cols:
                st.plotly_chart(charts.pair_plot(df, cols, color), use_container_width=True)
        elif chart_type == "Parallel Coordinates":
            cols = st.multiselect("Columns", num_cols, default=num_cols[:4])
            color = st.selectbox("Color by (optional)", ["—"] + num_cols)
            color = None if color == "—" else color
            if cols:
                st.plotly_chart(charts.parallel_coordinates(df, cols, color), use_container_width=True)
        elif chart_type == "Radar":
            cat = st.selectbox("Category", cat_cols or all_cols)
            vals = st.multiselect("Value columns", num_cols, default=num_cols[:4])
            if vals:
                st.plotly_chart(charts.radar_chart(df, cat, vals), use_container_width=True)
        else:
            c1, c2, c3 = st.columns(3)
            x_opts = cat_cols + dt_cols + num_cols
            x = c1.selectbox("X axis", x_opts or all_cols)
            y = c2.selectbox("Y axis", num_cols or all_cols)
            group_opts = ["—"] + cat_cols
            group = c3.selectbox("Group/Color (optional)", group_opts)
            group = None if group == "—" else group

            if chart_type == "Bar":
                fig = charts.bar_chart(df, x, y)
            elif chart_type == "Grouped Bar" and group:
                fig = charts.grouped_bar(df, x, y, group)
            elif chart_type == "Stacked Bar" and group:
                fig = charts.stacked_bar(df, x, y, group)
            elif chart_type == "Line":
                fig = charts.line_chart(df, x, y, group)
            elif chart_type == "Area":
                fig = charts.area_chart(df, x, y, group)
            elif chart_type == "Scatter":
                fig = charts.scatter_chart(df, x, y, group)
            elif chart_type == "Bubble":
                size = st.selectbox("Bubble size", num_cols)
                fig = charts.bubble_chart(df, x, y, size, group)
            elif chart_type == "Box":
                fig = charts.box_chart(df, x, y)
            elif chart_type == "Violin":
                fig = charts.violin(df, x, y)
            elif chart_type == "Hexbin":
                fig = charts.hexbin(df, x, y)
            elif chart_type == "Rolling Average":
                window = st.slider("Window", 2, 30, 7)
                fig = charts.rolling_average(df, x, y, window)
            else:
                fig = charts.bar_chart(df, x, y)
            show_chart_with_insight(f"{chart_type}: {y} by {x}", fig, chart_type.lower(), "custom_out", df=df, x=x, y=y)
    except Exception as e:
        error_box("Chart generation failed", e)


def tab_analytics(df: pd.DataFrame):
    clean_df = st.session_state.get("clean_df", df)

    section("🔍", "Anomaly Detection")
    with st.spinner("Running dual-engine anomaly detection…"):
        anomaly_df = cached_anomalies(clean_df)
    n_anom = int(anomaly_df["is_anomaly"].sum())
    metric_row([("Anomalies", str(n_anom), None), ("Normal", str(len(anomaly_df) - n_anom), None),
                ("Rate", f"{round(100*n_anom/max(len(anomaly_df),1),1)}%", None)])

    if n_anom > 0:
        c1, c2 = st.columns([1.4, 1])
        with c1:
            flagged = anomaly_df[anomaly_df["is_anomaly"]].drop(columns=["is_anomaly"], errors="ignore") \
                        .sort_values("anomaly_score", ascending=False)
            st.dataframe(flagged, use_container_width=True, height=280)
        with c2:
            num_cols = get_numeric_cols(clean_df)
            if len(num_cols) >= 2:
                x = st.selectbox("X", num_cols, key="anom_x")
                y = st.selectbox("Y", num_cols, index=min(1, len(num_cols)-1), key="anom_y")
                st.plotly_chart(charts.anomaly_chart(anomaly_df, x, y), use_container_width=True)
    else:
        st.success("✅ No anomalies detected.")

    section("📈", "Outlier Summary (IQR method)")
    out_summary = analytics.outlier_summary(clean_df)
    if not out_summary.empty:
        st.dataframe(out_summary, use_container_width=True)
    else:
        st.info("No numeric columns to check.")

    st.divider()
    section("🔮", "Forecasting")
    num_cols, dt_cols = get_numeric_cols(clean_df), get_datetime_cols(clean_df)
    if not dt_cols:
        st.warning("No datetime column detected for forecasting.")
        return
    c1, c2 = st.columns(2)
    date_col = c1.selectbox("Date column", dt_cols, key="fc_date")
    value_col = c2.selectbox("Value column", num_cols, key="fc_val") if num_cols else None
    if value_col and st.button("▶ Run Forecast"):
        with st.spinner("Forecasting…"):
            result = cached_forecast(clean_df, date_col, value_col)
        st.caption(f"ℹ️ {result.message}")
        if result.forecast_df is not None:
            st.plotly_chart(charts.forecast_chart(clean_df, result.forecast_df, date_col, value_col),
                            use_container_width=True)
            if result.seasonal_df is not None:
                st.plotly_chart(charts.seasonal_trend(result.seasonal_df), use_container_width=True)


def tab_ml(df: pd.DataFrame):
    clean_df = st.session_state.get("clean_df", df)
    all_cols = clean_df.columns.tolist()

    section("🎯", "Supervised Learning (AutoML)")
    target = st.selectbox("Target column", all_cols, key="ml_target")
    task = detect_task(clean_df, target)
    st.caption(f"Detected task: **{task}**")

    if st.button("▶ Train & Compare Models"):
        with st.spinner("Training candidate models with cross-validation…"):
            report = cached_ml(clean_df, target)
        if report.task == "unavailable":
            st.warning(report.message)
        else:
            st.success(report.message)
            rows = [{"Model": r.name, **r.metrics, "CV Mean": r.cv_mean, "CV Std": r.cv_std} for r in report.results]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            if report.feature_importance is not None:
                st.plotly_chart(charts.feature_importance_chart(report.feature_importance), use_container_width=True)

    st.divider()
    section("🧩", "Unsupervised: PCA & Clustering")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**PCA**")
        if st.button("Run PCA"):
            pca_result = ml.run_pca(clean_df)
            if "error" in pca_result:
                st.warning(pca_result["error"])
            else:
                st.plotly_chart(charts.pca_scatter(pca_result["components"]), use_container_width=True)
                st.caption(f"Explained variance: {pca_result['explained_variance_ratio']}")
                st.dataframe(pca_result["loadings"], use_container_width=True)
    with c2:
        st.markdown("**Clustering (auto-k KMeans)**")
        if st.button("Run Clustering"):
            cl_result = ml.run_clustering(clean_df)
            if "error" in cl_result:
                st.warning(cl_result["error"])
            else:
                st.caption(f"k = {cl_result['k']}, silhouette = {cl_result['silhouette']}")
                num_cols = cl_result["columns_used"]
                if len(num_cols) >= 2:
                    st.plotly_chart(charts.cluster_plot(clean_df, num_cols[0], num_cols[1], cl_result["labels"]),
                                    use_container_width=True)


def tab_insights(df: pd.DataFrame, api_key: str):
    clean_df = st.session_state.get("clean_df", df)
    try:
        profile = cached_profile(clean_df)
        domain = cached_domain(clean_df)
        report = full_report(clean_df, profile, domain)
    except Exception as e:
        return error_box("Insight generation failed", e)

    section("📝", "Executive Summary")
    st.markdown(f'<div class="glass-panel">{report["executive_summary"]}</div>', unsafe_allow_html=True)

    if api_key:
        llm = LLMClient(api_key)
        if llm.is_available and st.button("✨ Enrich with AI"):
            with st.spinner("Generating enriched narrative…"):
                st.markdown(f'<div class="glass-panel">{llm.enrich_summary(clean_df, report["executive_summary"])}</div>',
                           unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        section("🔎", "Key Findings")
        for f in report["key_findings"]:
            st.markdown(f"- {f}")
        section("⚠️", "Business Risks")
        for r in report["risks"]:
            st.markdown(f"- {r}")
    with c2:
        section("💡", "Opportunities")
        for o in report["opportunities"]:
            st.markdown(f"- {o}")
        section("✅", "Recommendations")
        for r in report["recommendations"]:
            st.markdown(f"- {r}")

    section("🗺️", "Action Plan")
    for i, step in enumerate(report["action_plan"], 1):
        st.markdown(f"{i}. {step}")

    if api_key:
        st.divider()
        section("💬", "Ask a Question")
        llm = LLMClient(api_key)
        q = st.text_input("Ask about this dataset", key="chat_q")
        if q and st.button("Send"):
            with st.spinner("Thinking…"):
                answer = llm.chat(q, clean_df, report["executive_summary"])
            st.markdown(f'<div class="glass-panel">{answer}</div>', unsafe_allow_html=True)


def tab_export(df: pd.DataFrame):
    clean_df = st.session_state.get("clean_df", df)
    section("📤", "Export")
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("⬇️ CSV", df_to_csv_bytes(clean_df), "dataset.csv", "text/csv", use_container_width=True)
    c2.download_button("⬇️ JSON", df_to_json_bytes(clean_df), "dataset.json", "application/json", use_container_width=True)

    try:
        profile = cached_profile(clean_df)
        excel_bytes = to_excel_bytes({"Data": clean_df, "Schema": pd.DataFrame(
            [{"Column": n, "Type": p.inferred_type.value, "Missing%": p.missing_pct} for n, p in profile.schema.items()])})
        c3.download_button("⬇️ Excel", excel_bytes, "report.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
    except Exception as e:
        c3.error("Excel export failed")

    try:
        domain = cached_domain(clean_df)
        report = full_report(clean_df, profile, domain)
        pdf_bytes = to_pdf_summary(UI.page_title, [
            ("Executive Summary", [report["executive_summary"]]),
            ("Key Findings", report["key_findings"]),
            ("Risks", report["risks"]),
            ("Recommendations", report["recommendations"]),
        ])
        c4.download_button("⬇️ PDF", pdf_bytes, "report.pdf", "application/pdf", use_container_width=True)
    except Exception:
        c4.error("PDF export failed")

    st.divider()
    if st.button("Generate HTML Dashboard"):
        try:
            figs = charts.auto_charts(clean_df)
            sections = [(title, fig.to_html(full_html=False, include_plotlyjs="cdn")) for title, fig, _ in figs]
            html_bytes = to_html_dashboard(UI.page_title, sections)
            st.download_button("⬇️ Download HTML Dashboard", html_bytes, "dashboard.html", "text/html")
        except Exception as e:
            st.error(f"HTML export failed: {e}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    apply_theme_styles()

    runtime_info = python_runtime_support()
    if not runtime_info["supported"]:
        st.warning(
            f"Python {runtime_info['version']} is outside the validated range for this app. "
            f"Use Python {runtime_info['recommended']} for the most reliable local and deployment experience."
        )

    df_raw, api_key = build_sidebar()

    if df_raw is None:
        st.markdown("""
        <div class="sticky-header">
          <div class="header-title">{UI.page_title}</div>
          <div class="header-subtitle">Ingest → Profile → Clean → Analyze → Visualize → Model → Explain → Export</div>
        </div>
        """.format(UI=UI), unsafe_allow_html=True)
        render_landing_page()
        return

    with st.spinner("Preparing dashboard insights…"):
        profile = cached_profile(df_raw)
        domain = cached_domain(df_raw)

    st.markdown("""
    <div class="sticky-header">
      <div style="display:flex; justify-content:space-between; gap:0.8rem; align-items:center; flex-wrap:wrap;">
        <div>
          <div class="header-title">{UI.page_title}</div>
          <div class="header-subtitle">{source_name} • {domain} • Quality {quality:.0f}/100</div>
        </div>
        <div style="font-size:0.85rem; color:var(--text-muted);">{rows:,} rows • {cols} columns</div>
      </div>
    </div>
    """.format(UI=UI, source_name=st.session_state.get("last_loaded_name", "Dataset"), domain=domain.domain, quality=profile.quality.overall, rows=profile.shape[0], cols=profile.shape[1]), unsafe_allow_html=True)

    render_breadcrumbs(["Workspace", domain.domain, "Overview"])
    render_search_input()

    metric_row([
        ("Rows", f"{profile.shape[0]:,}", None),
        ("Columns", str(profile.shape[1]), None),
        ("Memory", profile.memory_human, None),
        ("Quality", f"{profile.quality.overall:.0f}/100", f"{domain.domain}"),
    ])

    st.markdown(f"""
    <div class="glass-panel page-enter">
      <div style="display:flex; justify-content:space-between; align-items:center; gap:0.8rem; flex-wrap:wrap;">
        <div>
          <div style="font-weight:700; color:var(--text-primary);">Dashboard health</div>
          <div style="color:var(--text-muted); font-size:0.92rem;">Score reflects completeness, uniqueness, consistency, and validity.</div>
        </div>
        <div style="font-weight:700; color:var(--accent);">{profile.quality.overall:.0f}%</div>
      </div>
      <div class="progress-shell"><div style="width:{profile.quality.overall:.0f}%;"></div></div>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["📥 Overview", "🧹 Clean", "📊 Statistics", "🎨 Visualize",
                    "🔍 Analytics", "🤖 ML", "💡 Insights", "📤 Export"])
    tab_fns = [tab_overview, tab_clean, tab_statistics, tab_visualize, tab_analytics, tab_ml]
    for tab, fn in zip(tabs[:6], tab_fns):
        with tab:
            try:
                fn(df_raw)
            except Exception as e:
                error_box(f"{fn.__name__} failed", e)
    with tabs[6]:
        try:
            tab_insights(df_raw, api_key)
        except Exception as e:
            error_box("Insights failed", e)
    with tabs[7]:
        try:
            tab_export(df_raw)
        except Exception as e:
            error_box("Export failed", e)


if __name__ == "__main__":
    main()
