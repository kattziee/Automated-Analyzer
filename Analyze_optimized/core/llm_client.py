"""
llm_client.py — Optional, OPT-IN LLM enrichment layer.

The application is fully functional without this module (see core/insights.py
for the deterministic rule-based engine). If a user supplies their own API
key at runtime (never hardcoded, never committed), this wraps an
OpenAI-compatible chat endpoint to add richer narrative and conversational
Q&A on top of the computed statistics.

Security notes:
  - No default/hardcoded API key is ever embedded in source.
  - Keys are only read from the value passed in by the caller (which the UI
    sources from st.secrets or a runtime text input), never from source code.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False

MAX_TOKENS = 1200
SAMPLE_ROWS = 8

SYSTEM_PROMPT = (
    "You are a senior data analyst. Be precise, concise, and business-focused. "
    "Reference actual column names and numbers from the provided context. "
    "Use markdown with bold key figures and bullet points."
)


def _sample_table(df: pd.DataFrame, n: int = SAMPLE_ROWS, max_chars: int = 2500) -> str:
    try:
        sample = df.head(n)
        cols = sample.columns.tolist()
        header = " | ".join(str(c) for c in cols)
        sep = " | ".join(["---"] * len(cols))
        rows = [" | ".join(str(v)[:20] for v in row.values) for _, row in sample.iterrows()]
        return "\n".join([header, sep] + rows)[:max_chars]
    except Exception:
        return ""


class LLMClient:
    """Thin, defensive wrapper around an OpenAI-compatible chat endpoint."""

    def __init__(self, api_key: Optional[str], base_url: str = "https://api.x.ai/v1", model: str = "grok-3"):
        self.api_key = (api_key or "").strip()
        self.model = model
        self.base_url = base_url
        self._client = None
        self._available = False
        if OPENAI_SDK_AVAILABLE and self.api_key:
            try:
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                self._available = True
            except Exception:
                self._available = False

    @property
    def is_available(self) -> bool:
        return self._available and self._client is not None

    def _call(self, messages: list[dict], max_tokens: int = MAX_TOKENS) -> str:
        if not self.is_available:
            return "⚠️ AI enrichment is not configured. Provide an API key in the sidebar to enable it."
        try:
            resp = self._client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=max_tokens, temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️ AI request failed: {e}"

    def enrich_summary(self, df: pd.DataFrame, base_summary: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Here is a rule-based summary of a dataset:\n{base_summary}\n\n"
                f"Sample rows:\n{_sample_table(df)}\n\n"
                "Expand this into a richer executive narrative (max 200 words), "
                "adding plausible business context and one recommended next step."
            )},
        ]
        return self._call(messages)

    def chat(self, user_message: str, df: pd.DataFrame, context: str, history: Optional[list[dict]] = None) -> str:
        messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nDataset context:\n{context}\n\n"
                                                    f"Sample:\n{_sample_table(df, 5)}"}]
        if history:
            messages.extend(history[-8:])
        messages.append({"role": "user", "content": user_message})
        return self._call(messages)
