"""
GrokClient — xAI Grok API wrapper (OpenAI-compatible SDK).

Provides:
  - generate_summary()      → executive BI summary from data stats
  - chat()                  → conversational NLQ against the dataset
  - explain_anomalies()     → plain-English root-cause for flagged rows
  - explain_forecast()      → business narrative for forecast output
"""
from __future__ import annotations

import json
from typing import Optional

import pandas as pd

try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False


DEFAULT_MODEL = "grok-3"
FALLBACK_MODEL = "grok-3-mini"
BASE_URL = "https://api.x.ai/v1"
MAX_TOKENS = 2048
SAMPLE_ROWS = 8


SYSTEM_PROMPT = """You are a Senior Automated Data Analyst and Business Intelligence Expert embedded inside an enterprise data platform.

Your primary directives:
1. Deliver precise, actionable, executive-grade insights — no fluff, no boilerplate.
2. When analysing data, always identify the top 3 KPIs that matter most to business outcomes.
3. When explaining anomalies, provide a plausible root-cause hypothesis AND a recommended corrective action.
4. When interpreting forecasts, contextualize the trend (growth, decline, seasonality) and quantify the expected business impact.
5. When answering conversational queries, be concise but data-specific — reference column names, values, and statistics.
6. Format responses with markdown: use **bold** for key numbers, bullet lists for findings, and `code` for column names.
7. Always end analytical responses with a "💡 Recommended Action" section."""


def _truncate_df_to_str(df: pd.DataFrame, n: int = SAMPLE_ROWS, max_chars: int = 3000) -> str:
    """
    Serialize a DataFrame sample to a compact string.
    Uses markdown table if tabulate is available, otherwise falls back to
    a clean pipe-delimited text table — never raises ImportError.
    """
    sample = df.head(n)
    # Try markdown (requires tabulate)
    try:
        import tabulate as _tab  # noqa: F401
        md = sample.to_markdown(index=False)
        return md[:max_chars]
    except (ImportError, Exception):
        pass
    # Manual fallback: pipe-delimited table
    try:
        cols = sample.columns.tolist()
        header = " | ".join(str(c) for c in cols)
        sep    = " | ".join(["---"] * len(cols))
        rows   = []
        for _, row in sample.iterrows():
            rows.append(" | ".join(str(v)[:20] for v in row.values))
        md = "\n".join([header, sep] + rows)
        return md[:max_chars]
    except Exception:
        return sample.to_string(index=False)[:max_chars]


def _stats_summary(df: pd.DataFrame) -> str:
    """Build a structured stats string to feed into Grok."""
    parts = [f"Shape: {df.shape[0]} rows × {df.shape[1]} columns"]
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    dt_cols = df.select_dtypes(include="datetime64").columns.tolist()

    parts.append(f"Numeric columns ({len(num_cols)}): {', '.join(num_cols[:10])}")
    parts.append(f"Categorical columns ({len(cat_cols)}): {', '.join(cat_cols[:10])}")
    if dt_cols:
        parts.append(f"Datetime columns: {', '.join(dt_cols)}")

    if num_cols:
        try:
            desc = df[num_cols].describe().round(2).to_string()
            parts.append(f"\nDescriptive Statistics:\n{desc}")
        except Exception:
            pass

    null_info = df.isnull().sum()
    null_info = null_info[null_info > 0]
    if len(null_info):
        parts.append(f"\nMissing values: {null_info.to_dict()}")

    return "\n".join(parts)


class GrokClient:
    """
    Thin wrapper around the xAI Grok API (OpenAI-compatible).
    All methods return plain strings for rendering in Streamlit.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client: Optional[object] = None
        self._model = DEFAULT_MODEL
        self._available = False
        self._init_client()

    def _init_client(self):
        if not OPENAI_SDK_AVAILABLE:
            return
        if not self.api_key or not self.api_key.strip():
            return
        try:
            self._client = OpenAI(api_key=self.api_key, base_url=BASE_URL)
            self._available = True
        except Exception as e:
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available and self._client is not None

    def _call(self, messages: list[dict], max_tokens: int = MAX_TOKENS) -> str:
        """Core API call with model fallback."""
        if not self.is_available:
            return "⚠️ Grok AI is not configured. Please provide a valid API key in the sidebar."
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e)
            # Try fallback model once
            if "model" in err_str.lower() and self._model != FALLBACK_MODEL:
                try:
                    self._model = FALLBACK_MODEL
                    resp = self._client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=0.3,
                    )
                    return resp.choices[0].message.content.strip()
                except Exception as e2:
                    return f"⚠️ Grok API error: {e2}"
            return f"⚠️ Grok API error: {err_str}"

    # ── Public methods ────────────────────────────────────────────────────────

    def generate_summary(self, df: pd.DataFrame) -> str:
        """Generate an executive BI summary from the dataset."""
        stats = _stats_summary(df)
        sample = _truncate_df_to_str(df)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Analyse this dataset and produce a comprehensive executive summary.\n\n"
                f"**Dataset Statistics:**\n{stats}\n\n"
                f"**Sample Data (first {SAMPLE_ROWS} rows):**\n{sample}\n\n"
                f"Provide:\n"
                f"1. Dataset overview (what it represents, key dimensions)\n"
                f"2. Top 3 KPI insights with specific numbers\n"
                f"3. Data quality observations\n"
                f"4. Business patterns or trends visible in the data\n"
                f"5. 💡 Recommended Action"
            )},
        ]
        return self._call(messages, max_tokens=1500)

    def chat(
        self,
        user_message: str,
        df: pd.DataFrame,
        history: Optional[list[dict]] = None,
    ) -> str:
        """Conversational NLQ interface."""
        stats = _stats_summary(df)
        sample = _truncate_df_to_str(df, n=5)
        context_msg = {
            "role": "system",
            "content": (
                SYSTEM_PROMPT + "\n\n"
                f"**Current Dataset Context:**\n{stats}\n\n"
                f"**Sample:**\n{sample}"
            ),
        }
        messages = [context_msg]
        if history:
            messages.extend(history[-8:])  # last 4 exchanges
        messages.append({"role": "user", "content": user_message})
        return self._call(messages, max_tokens=MAX_TOKENS)

    def explain_anomalies(self, anomaly_df: pd.DataFrame, full_df: pd.DataFrame) -> str:
        """Translate anomaly flags into plain-English root-cause analysis."""
        if anomaly_df.empty:
            return "✅ No anomalies were detected in the dataset."

        n_anomalies = len(anomaly_df)
        sample = _truncate_df_to_str(anomaly_df, n=min(10, n_anomalies))
        stats = _stats_summary(full_df)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"The anomaly detection engine flagged **{n_anomalies} records** in this dataset.\n\n"
                f"**Dataset Statistics:**\n{stats}\n\n"
                f"**Flagged Anomaly Records:**\n{sample}\n\n"
                f"Provide:\n"
                f"1. A plain-English explanation of what makes these records anomalous\n"
                f"2. Most likely root causes (data error vs genuine business event)\n"
                f"3. Business impact assessment\n"
                f"4. 💡 Recommended Action for each anomaly type"
            )},
        ]
        return self._call(messages, max_tokens=1200)

    def explain_forecast(
        self,
        forecast_df: pd.DataFrame,
        value_col: str,
        method: str,
    ) -> str:
        """Translate a forecast output into a business narrative."""
        try:
            future = forecast_df.tail(30)
            trend = "upward" if future["yhat"].iloc[-1] > future["yhat"].iloc[0] else "downward"
            pct_change = (
                (future["yhat"].iloc[-1] - future["yhat"].iloc[0])
                / abs(future["yhat"].iloc[0]) * 100
            ) if future["yhat"].iloc[0] != 0 else 0
            forecast_summary = (
                f"Method: {method}\n"
                f"30-day forecast for `{value_col}`:\n"
                f"  - Start: {future['yhat'].iloc[0]:.2f}\n"
                f"  - End:   {future['yhat'].iloc[-1]:.2f}\n"
                f"  - Trend: {trend} ({pct_change:+.1f}%)\n"
                f"  - Avg CI width: {(future['yhat_upper'] - future['yhat_lower']).mean():.2f}"
            )
        except Exception as e:
            forecast_summary = f"Forecast summary unavailable: {e}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Interpret this time-series forecast for a business audience.\n\n"
                f"**Forecast Output:**\n{forecast_summary}\n\n"
                f"Provide:\n"
                f"1. Plain-English trend narrative (what is happening and why it might be)\n"
                f"2. Confidence interpretation (is the CI wide or tight, and what does that mean?)\n"
                f"3. Key risks or opportunities in the next 30 days\n"
                f"4. 💡 Recommended Action"
            )},
        ]
        return self._call(messages, max_tokens=800)
