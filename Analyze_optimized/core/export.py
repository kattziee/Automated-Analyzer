"""export.py — Multi-format export (CSV, JSON, Excel, HTML, PDF) — stdlib+existing deps only."""
from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
from utils.helpers import df_to_csv_bytes, df_to_json_bytes


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    """sheets: {sheet_name: dataframe}. Uses openpyxl (already a dependency)."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe_name = str(name)[:31] or "Sheet1"
            (df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)).to_excel(writer, sheet_name=safe_name, index=False)
    return buf.getvalue()


def to_html_dashboard(title: str, sections: list[tuple[str, str]]) -> bytes:
    """sections: list of (heading, html_body) — e.g. plotly fig.to_html(full_html=False)."""
    parts = [f"<html><head><meta charset='utf-8'><title>{title}</title>",
             "<style>body{font-family:Inter,Arial,sans-serif;background:#0b0b1f;color:#e2e8f0;padding:2rem;}"
             "h1{color:#a5b4fc;} h2{color:#a5b4fc;border-bottom:1px solid #333;padding-bottom:.3rem;}"
             ".section{margin-bottom:2rem;background:#14142f;padding:1.2rem;border-radius:10px;}</style></head><body>",
             f"<h1>{title}</h1><p>Generated {datetime.now():%Y-%m-%d %H:%M}</p>"]
    for heading, body in sections:
        parts.append(f"<div class='section'><h2>{heading}</h2>{body}</div>")
    parts.append("</body></html>")
    return "".join(parts).encode("utf-8")


def to_pdf_summary(title: str, text_sections: list[tuple[str, list[str]]]) -> bytes:
    """Lightweight PDF via matplotlib (already a dependency) — no extra libs needed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.95, title, fontsize=18, weight="bold")
        fig.text(0.08, 0.92, f"Generated {datetime.now():%Y-%m-%d %H:%M}", fontsize=9, color="gray")
        y = 0.87
        for heading, lines in text_sections:
            if y < 0.08:
                pdf.savefig(fig); plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69)); y = 0.95
            fig.text(0.08, y, heading, fontsize=13, weight="bold"); y -= 0.03
            for line in lines:
                wrapped = _wrap(line, 95)
                for w in wrapped:
                    if y < 0.05:
                        pdf.savefig(fig); plt.close(fig)
                        fig = plt.figure(figsize=(8.27, 11.69)); y = 0.95
                    fig.text(0.1, y, w, fontsize=9); y -= 0.022
                y -= 0.01
        pdf.savefig(fig); plt.close(fig)
    return buf.getvalue()


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = f"{cur} {w}".strip()
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines or [""]
