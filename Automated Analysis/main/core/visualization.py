"""
ChartFactory — Schema-aware Plotly visualization engine.

Generates:
  - Auto-detected charts (bar, line, scatter) based on column types
  - Correlation heatmap
  - Anomaly scatter plot
  - Prophet / ETS forecast chart with confidence bands
  - Distribution histograms

All charts share a consistent dark-mode theme with brand colours.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Theme constants ───────────────────────────────────────────────────────────

PALETTE = [
    "#6366f1",  # indigo
    "#22d3ee",  # cyan
    "#a78bfa",  # violet
    "#34d399",  # emerald
    "#f97316",  # orange
    "#f43f5e",  # rose
    "#facc15",  # yellow
    "#60a5fa",  # blue
]

TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="rgba(15,15,35,0.0)",
        plot_bgcolor="rgba(15,15,35,0.0)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0", size=13),
        colorway=PALETTE,
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.07)",
            linecolor="rgba(255,255,255,0.15)",
            tickcolor="rgba(255,255,255,0.3)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.07)",
            linecolor="rgba(255,255,255,0.15)",
            tickcolor="rgba(255,255,255,0.3)",
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.05)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
        ),
        hoverlabel=dict(
            bgcolor="#1e1b4b",
            bordercolor="#6366f1",
            font=dict(color="#e2e8f0"),
        ),
        margin=dict(l=40, r=20, t=50, b=40),
    )
)

_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(15,15,35,0.0)",
    plot_bgcolor="rgba(15,15,35,0.0)",
    font=dict(family="Inter, sans-serif", color="#e2e8f0", size=13),
    colorway=PALETTE,
    legend=dict(
        bgcolor="rgba(255,255,255,0.05)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1,
    ),
    hoverlabel=dict(
        bgcolor="#1e1b4b",
        bordercolor="#6366f1",
        font=dict(color="#e2e8f0"),
    ),
    margin=dict(l=40, r=20, t=55, b=40),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.07)",
        linecolor="rgba(255,255,255,0.15)",
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.07)",
        linecolor="rgba(255,255,255,0.15)",
    ),
)


def _apply_theme(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(**_LAYOUT_DEFAULTS)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=16, color="#a5b4fc")))
    return fig


class ChartFactory:
    """Generate all dashboard charts."""

    # ── Auto-chart dispatcher ─────────────────────────────────────────────────

    def auto_charts(self, df: pd.DataFrame) -> list[tuple[str, go.Figure]]:
        """
        Returns a list of (title, figure) tuples based on schema inference.
        Generates up to 6 auto-selected charts.
        """
        charts = []
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        dt_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

        if not num_cols:
            return charts

        primary_num = num_cols[0]
        secondary_num = num_cols[1] if len(num_cols) > 1 else None

        # 1) Bar chart: best categorical × top numeric
        if cat_cols:
            best_cat = min(cat_cols, key=lambda c: df[c].nunique() if df[c].nunique() > 1 else 999)
            fig = self.bar_chart(df, x=best_cat, y=primary_num)
            charts.append((f"{primary_num} by {best_cat}", fig))

        # 2) Line chart: datetime × numeric
        if dt_cols:
            fig = self.line_chart(df, x=dt_cols[0], y=primary_num)
            charts.append((f"{primary_num} over Time", fig))

        # 3) Scatter: two numerics
        if secondary_num:
            color = cat_cols[0] if cat_cols else None
            fig = self.scatter_chart(df, x=primary_num, y=secondary_num, color=color)
            charts.append((f"{primary_num} vs {secondary_num}", fig))

        # 4) Distribution histogram of primary numeric
        fig = self.histogram(df, col=primary_num)
        charts.append((f"Distribution of {primary_num}", fig))

        # 5) Correlation heatmap (if ≥ 3 numeric columns)
        if len(num_cols) >= 3:
            fig = self.correlation_heatmap(df)
            charts.append(("Correlation Heatmap", fig))

        # 6) Box plot: numeric by category
        if cat_cols and secondary_num:
            best_cat = cat_cols[0]
            fig = self.box_chart(df, x=best_cat, y=secondary_num)
            charts.append((f"{secondary_num} Distribution by {best_cat}", fig))

        return charts

    # ── Individual chart methods ──────────────────────────────────────────────

    def bar_chart(self, df: pd.DataFrame, x: str, y: str, title: str = "") -> go.Figure:
        try:
            agg = df.groupby(x, as_index=False)[y].mean().sort_values(y, ascending=False)
            fig = px.bar(
                agg, x=x, y=y,
                color=x,
                color_discrete_sequence=PALETTE,
                labels={y: y, x: x},
            )
            fig.update_traces(marker_line_width=0)
            return _apply_theme(fig, title or f"Average {y} by {x}")
        except Exception as e:
            return self._error_fig(str(e))

    def line_chart(self, df: pd.DataFrame, x: str, y: str, title: str = "") -> go.Figure:
        try:
            ts = df[[x, y]].dropna().sort_values(x)
            fig = px.line(
                ts, x=x, y=y,
                color_discrete_sequence=PALETTE,
            )
            fig.update_traces(line=dict(width=2.5))
            return _apply_theme(fig, title or f"{y} over {x}")
        except Exception as e:
            return self._error_fig(str(e))

    def scatter_chart(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        title: str = "",
    ) -> go.Figure:
        try:
            fig = px.scatter(
                df.dropna(subset=[x, y]),
                x=x, y=y,
                color=color,
                color_discrete_sequence=PALETTE,
                trendline="ols",
                trendline_color_override="#f43f5e",
                opacity=0.75,
            )
            return _apply_theme(fig, title or f"{x} vs {y}")
        except Exception as e:
            return self._error_fig(str(e))

    def histogram(self, df: pd.DataFrame, col: str, title: str = "") -> go.Figure:
        try:
            fig = px.histogram(
                df.dropna(subset=[col]),
                x=col,
                nbins=40,
                color_discrete_sequence=[PALETTE[0]],
                marginal="box",
                opacity=0.85,
            )
            fig.update_traces(marker_line_width=0.3, marker_line_color="#1e1b4b")
            return _apply_theme(fig, title or f"Distribution of {col}")
        except Exception as e:
            return self._error_fig(str(e))

    def box_chart(self, df: pd.DataFrame, x: str, y: str, title: str = "") -> go.Figure:
        try:
            fig = px.box(
                df.dropna(subset=[x, y]),
                x=x, y=y,
                color=x,
                color_discrete_sequence=PALETTE,
                notched=True,
            )
            return _apply_theme(fig, title or f"{y} by {x}")
        except Exception as e:
            return self._error_fig(str(e))

    def correlation_heatmap(self, df: pd.DataFrame, title: str = "") -> go.Figure:
        try:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(num_cols) < 2:
                return self._error_fig("Need ≥ 2 numeric columns for correlation heatmap.")
            corr = df[num_cols].corr().round(2)
            fig = go.Figure(
                data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns.tolist(),
                    y=corr.index.tolist(),
                    colorscale=[
                        [0, "#f43f5e"],
                        [0.5, "#1e1b4b"],
                        [1, "#6366f1"],
                    ],
                    zmid=0,
                    text=corr.values,
                    texttemplate="%{text}",
                    hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z}<extra></extra>",
                    showscale=True,
                )
            )
            return _apply_theme(fig, title or "Correlation Matrix")
        except Exception as e:
            return self._error_fig(str(e))

    def anomaly_chart(self, df: pd.DataFrame, x: str, y: str) -> go.Figure:
        """Scatter chart coloring anomalies in red."""
        try:
            normal = df[~df["is_anomaly"]]
            anomalous = df[df["is_anomaly"]]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=normal[x], y=normal[y],
                mode="markers",
                name="Normal",
                marker=dict(color=PALETTE[1], size=6, opacity=0.7),
                hovertemplate=f"<b>{x}</b>: %{{x}}<br><b>{y}</b>: %{{y}}<extra>Normal</extra>",
            ))
            fig.add_trace(go.Scatter(
                x=anomalous[x], y=anomalous[y],
                mode="markers",
                name="Anomaly",
                marker=dict(color="#f43f5e", size=10, symbol="x", opacity=0.9,
                            line=dict(width=1.5, color="#fff")),
                hovertemplate=(
                    f"<b>{x}</b>: %{{x}}<br><b>{y}</b>: %{{y}}<br>"
                    "<b>Reason</b>: %{customdata}<extra>⚠ Anomaly</extra>"
                ),
                customdata=anomalous["anomaly_reason"].values,
            ))
            return _apply_theme(fig, f"Anomaly Detection: {y} vs {x}")
        except Exception as e:
            return self._error_fig(str(e))

    def forecast_chart(
        self, df: pd.DataFrame, forecast_df: pd.DataFrame, date_col: str, value_col: str
    ) -> go.Figure:
        """Prophet/ETS forecast with actual data and confidence bands."""
        try:
            last_actual_date = pd.to_datetime(df[date_col]).max()

            historical = forecast_df[forecast_df["ds"] <= last_actual_date]
            future = forecast_df[forecast_df["ds"] > last_actual_date]

            fig = go.Figure()

            # Confidence band
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast_df["ds"], forecast_df["ds"][::-1]]),
                y=pd.concat([forecast_df["yhat_upper"], forecast_df["yhat_lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(99,102,241,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="95% CI",
                hoverinfo="skip",
            ))

            # Historical fit
            actual = (
                df[[date_col, value_col]]
                .dropna()
                .sort_values(date_col)
                .groupby(date_col, as_index=False)[value_col].mean()
            )
            fig.add_trace(go.Scatter(
                x=actual[date_col], y=actual[value_col],
                mode="lines+markers",
                name="Actual",
                line=dict(color=PALETTE[1], width=2),
                marker=dict(size=4),
            ))

            # Forecast line
            fig.add_trace(go.Scatter(
                x=future["ds"], y=future["yhat"],
                mode="lines",
                name="Forecast",
                line=dict(color=PALETTE[0], width=2.5, dash="dash"),
            ))

            # Divider line
            fig.add_vline(
                x=last_actual_date, line_dash="dot",
                line_color="rgba(255,255,255,0.3)", line_width=1,
                annotation_text=" Forecast →",
                annotation_font_color="#a5b4fc",
            )

            return _apply_theme(fig, f"30-Day Forecast: {value_col}")
        except Exception as e:
            return self._error_fig(str(e))

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _error_fig(msg: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(
            text=f"⚠️ Chart unavailable<br><sub>{msg}</sub>",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="#f87171"),
        )
        return _apply_theme(fig)
