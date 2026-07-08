"""
visualization.py — Schema-aware Plotly visualization engine.

Every chart method returns a `go.Figure`. A companion `insight_for(...)`
function generates a short natural-language interpretation for a given
chart so the UI can show "what am I looking at" text without needing an
external LLM call.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from utils.helpers import get_numeric_cols, get_categorical_cols, get_datetime_cols

PALETTE = ["#6366f1", "#22d3ee", "#a78bfa", "#34d399", "#f97316",
           "#f43f5e", "#facc15", "#60a5fa"]

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#e2e8f0", size=13),
    colorway=PALETTE,
    legend=dict(bgcolor="rgba(255,255,255,0.05)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
    hoverlabel=dict(bgcolor="#1e1b4b", bordercolor="#6366f1", font=dict(color="#e2e8f0")),
    margin=dict(l=40, r=20, t=55, b=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.07)", linecolor="rgba(255,255,255,0.15)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.07)", linecolor="rgba(255,255,255,0.15)"),
    dragmode="zoom",
    hovermode="x unified",
)


def _theme(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(**_LAYOUT)
    fig.update_traces(marker=dict(line=dict(width=0)))
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=16, color="#a5b4fc")))
    return fig


def _err_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=f"⚠️ Chart unavailable<br><sub>{msg}</sub>", xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="#f87171"))
    return _theme(fig)


class ChartFactory:
    """Central factory for all supported chart types."""

    # ── Distribution ─────────────────────────────────────────────────────

    def histogram(self, df: pd.DataFrame, col: str, bins: int = 40) -> go.Figure:
        try:
            sample = df.dropna(subset=[col]).sample(min(len(df.dropna(subset=[col])), 20000), random_state=42)
            fig = px.histogram(sample, x=col, nbins=min(bins, 60),
                                color_discrete_sequence=[PALETTE[0]], marginal="box", opacity=0.85)
            return _theme(fig, f"Distribution of {col}")
        except Exception as e:
            return _err_fig(str(e))

    def kde(self, df: pd.DataFrame, col: str) -> go.Figure:
        try:
            series = df[col].dropna()
            if len(series) < 3:
                return _err_fig("Need ≥ 3 values for a density curve.")
            kde = stats.gaussian_kde(series)
            xs = np.linspace(series.min(), series.max(), 200)
            fig = go.Figure(go.Scatter(x=xs, y=kde(xs), mode="lines", fill="tozeroy",
                                        line=dict(color=PALETTE[1], width=2)))
            return _theme(fig, f"Density (KDE) of {col}")
        except Exception as e:
            return _err_fig(str(e))

    def box_chart(self, df: pd.DataFrame, x: str, y: str) -> go.Figure:
        try:
            fig = px.box(df.dropna(subset=[x, y]), x=x, y=y, color=x,
                         color_discrete_sequence=PALETTE, notched=True)
            return _theme(fig, f"{y} by {x}")
        except Exception as e:
            return _err_fig(str(e))

    def violin(self, df: pd.DataFrame, x: str, y: str) -> go.Figure:
        try:
            fig = px.violin(df.dropna(subset=[x, y]), x=x, y=y, color=x, box=True, points=False,
                             color_discrete_sequence=PALETTE)
            return _theme(fig, f"{y} distribution by {x}")
        except Exception as e:
            return _err_fig(str(e))

    def ecdf(self, df: pd.DataFrame, col: str) -> go.Figure:
        try:
            fig = px.ecdf(df.dropna(subset=[col]), x=col, color_discrete_sequence=[PALETTE[2]])
            return _theme(fig, f"Empirical CDF of {col}")
        except Exception as e:
            return _err_fig(str(e))

    def qq_plot(self, df: pd.DataFrame, col: str) -> go.Figure:
        try:
            series = df[col].dropna()
            if len(series) < 5:
                return _err_fig("Need ≥ 5 values for a Q-Q plot.")
            osm, osr = stats.probplot(series, dist="norm", fit=False)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=osm[0], y=osm[1], mode="markers",
                                      marker=dict(color=PALETTE[0], size=6), name="Observed"))
            line_x = np.array([osm[0].min(), osm[0].max()])
            slope, intercept = np.polyfit(osm[0], osm[1], 1)
            fig.add_trace(go.Scatter(x=line_x, y=slope * line_x + intercept, mode="lines",
                                      line=dict(color=PALETTE[5], dash="dash"), name="Theoretical"))
            return _theme(fig, f"Q-Q Plot: {col} vs Normal")
        except Exception as e:
            return _err_fig(str(e))

    # ── Relationship ─────────────────────────────────────────────────────

    def scatter_chart(self, df: pd.DataFrame, x: str, y: str, color: Optional[str] = None) -> go.Figure:
        try:
            sample = df.dropna(subset=[x, y]).sample(min(len(df.dropna(subset=[x, y])), 20000), random_state=42)
            fig = px.scatter(sample, x=x, y=y, color=color,
                              color_discrete_sequence=PALETTE, trendline="ols",
                              trendline_color_override="#f43f5e", opacity=0.75)
            return _theme(fig, f"{x} vs {y}")
        except Exception as e:
            return _err_fig(str(e))

    def bubble_chart(self, df: pd.DataFrame, x: str, y: str, size: str, color: Optional[str] = None) -> go.Figure:
        try:
            data = df.dropna(subset=[x, y, size])
            fig = px.scatter(data, x=x, y=y, size=size, color=color, size_max=40,
                              color_discrete_sequence=PALETTE, opacity=0.75)
            return _theme(fig, f"{x} vs {y} (bubble size = {size})")
        except Exception as e:
            return _err_fig(str(e))

    def hexbin(self, df: pd.DataFrame, x: str, y: str, nbins: int = 30) -> go.Figure:
        try:
            fig = px.density_heatmap(df.dropna(subset=[x, y]), x=x, y=y, nbinsx=nbins, nbinsy=nbins,
                                      color_continuous_scale=[[0, "#1e1b4b"], [1, "#6366f1"]])
            return _theme(fig, f"Density Heatmap: {x} vs {y}")
        except Exception as e:
            return _err_fig(str(e))

    def pair_plot(self, df: pd.DataFrame, cols: list[str], color: Optional[str] = None) -> go.Figure:
        try:
            cols = cols[:5]
            fig = px.scatter_matrix(df, dimensions=cols, color=color, color_discrete_sequence=PALETTE, opacity=0.6)
            fig.update_traces(diagonal_visible=False, showupperhalf=False)
            return _theme(fig, "Pair Plot")
        except Exception as e:
            return _err_fig(str(e))

    def correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        try:
            num_cols = get_numeric_cols(df)
            if len(num_cols) < 2:
                return _err_fig("Need ≥ 2 numeric columns.")
            corr = df[num_cols].corr().round(2)
            fig = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                colorscale=[[0, "#f43f5e"], [0.5, "#1e1b4b"], [1, "#6366f1"]], zmid=0,
                text=corr.values, texttemplate="%{text}", showscale=True,
                hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z}<extra></extra>",
            ))
            return _theme(fig, "Correlation Matrix")
        except Exception as e:
            return _err_fig(str(e))

    def parallel_coordinates(self, df: pd.DataFrame, cols: list[str], color: Optional[str] = None) -> go.Figure:
        try:
            data = df[cols].dropna()
            color_series = df.loc[data.index, color] if color and color in df.columns else None
            fig = px.parallel_coordinates(data, color=color_series, color_continuous_scale=px.colors.sequential.Plasma)
            return _theme(fig, "Parallel Coordinates")
        except Exception as e:
            return _err_fig(str(e))

    def radar_chart(self, df: pd.DataFrame, category_col: str, value_cols: list[str]) -> go.Figure:
        try:
            agg = df.groupby(category_col)[value_cols].mean().reset_index()
            fig = go.Figure()
            for _, row in agg.head(6).iterrows():
                fig.add_trace(go.Scatterpolar(r=[row[c] for c in value_cols] + [row[value_cols[0]]],
                                               theta=value_cols + [value_cols[0]], fill="toself",
                                               name=str(row[category_col])))
            return _theme(fig, f"Radar Chart by {category_col}")
        except Exception as e:
            return _err_fig(str(e))

    # ── Categorical / composition ────────────────────────────────────────

    def bar_chart(self, df: pd.DataFrame, x: str, y: str, agg: str = "mean") -> go.Figure:
        try:
            grouped = df.groupby(x, as_index=False)[y].agg(agg).sort_values(y, ascending=False).head(30)
            fig = px.bar(grouped, x=x, y=y, color=x, color_discrete_sequence=PALETTE)
            fig.update_traces(marker_line_width=0)
            return _theme(fig, f"{agg.title()} {y} by {x}")
        except Exception as e:
            return _err_fig(str(e))

    def grouped_bar(self, df: pd.DataFrame, x: str, y: str, group: str) -> go.Figure:
        try:
            agg = df.groupby([x, group], as_index=False)[y].mean()
            fig = px.bar(agg, x=x, y=y, color=group, barmode="group", color_discrete_sequence=PALETTE)
            return _theme(fig, f"{y} by {x}, grouped by {group}")
        except Exception as e:
            return _err_fig(str(e))

    def stacked_bar(self, df: pd.DataFrame, x: str, y: str, group: str) -> go.Figure:
        try:
            agg = df.groupby([x, group], as_index=False)[y].sum()
            fig = px.bar(agg, x=x, y=y, color=group, barmode="stack", color_discrete_sequence=PALETTE)
            return _theme(fig, f"{y} by {x}, stacked by {group}")
        except Exception as e:
            return _err_fig(str(e))

    def count_plot(self, df: pd.DataFrame, col: str, top_n: int = 20) -> go.Figure:
        try:
            counts = df[col].value_counts().head(top_n).reset_index()
            counts.columns = [col, "count"]
            fig = px.bar(counts, x=col, y="count", color=col, color_discrete_sequence=PALETTE)
            return _theme(fig, f"Count of {col}")
        except Exception as e:
            return _err_fig(str(e))

    def pie_chart(self, df: pd.DataFrame, names: str, values: Optional[str] = None) -> go.Figure:
        try:
            if values:
                agg = df.groupby(names, as_index=False)[values].sum()
                fig = px.pie(agg, names=names, values=values, color_discrete_sequence=PALETTE)
            else:
                counts = df[names].value_counts().reset_index()
                counts.columns = [names, "count"]
                fig = px.pie(counts, names=names, values="count", color_discrete_sequence=PALETTE)
            return _theme(fig, f"Share of {values or 'records'} by {names}")
        except Exception as e:
            return _err_fig(str(e))

    def donut_chart(self, df: pd.DataFrame, names: str, values: Optional[str] = None) -> go.Figure:
        fig = self.pie_chart(df, names, values)
        if fig.data:
            fig.update_traces(hole=0.55)
        return fig

    def treemap(self, df: pd.DataFrame, path: list[str], values: str) -> go.Figure:
        try:
            agg = df.groupby(path, as_index=False)[values].sum()
            fig = px.treemap(agg, path=path, values=values, color_discrete_sequence=PALETTE)
            return _theme(fig, f"Treemap of {values} by {' → '.join(path)}")
        except Exception as e:
            return _err_fig(str(e))

    def sunburst(self, df: pd.DataFrame, path: list[str], values: str) -> go.Figure:
        try:
            agg = df.groupby(path, as_index=False)[values].sum()
            fig = px.sunburst(agg, path=path, values=values, color_discrete_sequence=PALETTE)
            return _theme(fig, f"Sunburst of {values} by {' → '.join(path)}")
        except Exception as e:
            return _err_fig(str(e))

    # ── Time series ───────────────────────────────────────────────────────

    def line_chart(self, df: pd.DataFrame, x: str, y: str, color: Optional[str] = None) -> go.Figure:
        try:
            data = df[[x, y] + ([color] if color else [])].dropna().sort_values(x)
            if len(data) > 5000:
                data = data.sample(5000, random_state=42)
            fig = px.line(data, x=x, y=y, color=color, color_discrete_sequence=PALETTE)
            fig.update_traces(line=dict(width=2.5))
            return _theme(fig, f"{y} over {x}")
        except Exception as e:
            return _err_fig(str(e))

    def area_chart(self, df: pd.DataFrame, x: str, y: str, color: Optional[str] = None) -> go.Figure:
        try:
            data = df[[x, y] + ([color] if color else [])].dropna().sort_values(x)
            fig = px.area(data, x=x, y=y, color=color, color_discrete_sequence=PALETTE)
            return _theme(fig, f"{y} over {x} (area)")
        except Exception as e:
            return _err_fig(str(e))

    def rolling_average(self, df: pd.DataFrame, x: str, y: str, window: int = 7) -> go.Figure:
        try:
            data = df[[x, y]].dropna().sort_values(x)
            data["rolling"] = data[y].rolling(window, min_periods=1).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data[x], y=data[y], mode="lines", name=y,
                                      line=dict(color="rgba(99,102,241,0.35)", width=1.5)))
            fig.add_trace(go.Scatter(x=data[x], y=data["rolling"], mode="lines",
                                      name=f"{window}-period MA", line=dict(color=PALETTE[1], width=2.5)))
            return _theme(fig, f"{y} with {window}-period Rolling Average")
        except Exception as e:
            return _err_fig(str(e))

    def seasonal_trend(self, seasonal_df: pd.DataFrame) -> go.Figure:
        try:
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                 subplot_titles=("Trend", "Seasonal", "Residual"))
            fig.add_trace(go.Scatter(x=seasonal_df["ds"], y=seasonal_df["trend"],
                                      line=dict(color=PALETTE[0])), row=1, col=1)
            fig.add_trace(go.Scatter(x=seasonal_df["ds"], y=seasonal_df["seasonal"],
                                      line=dict(color=PALETTE[1])), row=2, col=1)
            fig.add_trace(go.Scatter(x=seasonal_df["ds"], y=seasonal_df["resid"], mode="markers",
                                      marker=dict(color=PALETTE[5], size=4)), row=3, col=1)
            fig.update_layout(showlegend=False, height=550)
            return _theme(fig, "Seasonal Decomposition")
        except Exception as e:
            return _err_fig(str(e))

    def forecast_chart(self, df: pd.DataFrame, forecast_df: pd.DataFrame, date_col: str, value_col: str) -> go.Figure:
        try:
            last_actual = pd.to_datetime(df[date_col]).max()
            future = forecast_df[forecast_df["ds"] > last_actual]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast_df["ds"], forecast_df["ds"][::-1]]),
                y=pd.concat([forecast_df["yhat_upper"], forecast_df["yhat_lower"][::-1]]),
                fill="toself", fillcolor="rgba(99,102,241,0.15)", line=dict(color="rgba(0,0,0,0)"),
                name="95% CI", hoverinfo="skip",
            ))
            actual = df[[date_col, value_col]].dropna().sort_values(date_col).groupby(date_col, as_index=False)[value_col].mean()
            fig.add_trace(go.Scatter(x=actual[date_col], y=actual[value_col], mode="lines+markers",
                                      name="Actual", line=dict(color=PALETTE[1], width=2), marker=dict(size=4)))
            fig.add_trace(go.Scatter(x=future["ds"], y=future["yhat"], mode="lines", name="Forecast",
                                      line=dict(color=PALETTE[0], width=2.5, dash="dash")))
            fig.add_vline(x=last_actual, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                          annotation_text=" Forecast →", annotation_font_color="#a5b4fc")
            return _theme(fig, f"Forecast: {value_col}")
        except Exception as e:
            return _err_fig(str(e))

    # ── ML-related visuals ────────────────────────────────────────────────

    def missing_value_heatmap(self, df: pd.DataFrame) -> go.Figure:
        try:
            mask = df.isna().astype(int)
            fig = go.Figure(go.Heatmap(z=mask.T.values, x=list(range(len(df))), y=mask.columns.tolist(),
                                        colorscale=[[0, "#1e1b4b"], [1, "#f43f5e"]], showscale=False))
            fig.update_yaxes(autorange="reversed")
            return _theme(fig, "Missing Value Map (red = missing)")
        except Exception as e:
            return _err_fig(str(e))

    def feature_importance_chart(self, fi_df: pd.DataFrame) -> go.Figure:
        try:
            fig = px.bar(fi_df.sort_values("importance"), x="importance", y="feature", orientation="h",
                         color_discrete_sequence=[PALETTE[0]])
            return _theme(fig, "Feature Importance")
        except Exception as e:
            return _err_fig(str(e))

    def pca_scatter(self, components: pd.DataFrame, color: Optional[pd.Series] = None) -> go.Figure:
        try:
            if components.shape[1] < 2:
                return _err_fig("Need ≥ 2 principal components.")
            fig = px.scatter(components, x="PC1", y="PC2", color=color, color_discrete_sequence=PALETTE, opacity=0.75)
            return _theme(fig, "PCA Projection (PC1 vs PC2)")
        except Exception as e:
            return _err_fig(str(e))

    def cluster_plot(self, df: pd.DataFrame, x: str, y: str, labels: np.ndarray) -> go.Figure:
        try:
            data = df[[x, y]].copy()
            data["cluster"] = [f"Cluster {c}" for c in labels]
            fig = px.scatter(data, x=x, y=y, color="cluster", color_discrete_sequence=PALETTE, opacity=0.8)
            return _theme(fig, f"Cluster Assignment: {x} vs {y}")
        except Exception as e:
            return _err_fig(str(e))

    def anomaly_chart(self, df: pd.DataFrame, x: str, y: str) -> go.Figure:
        try:
            normal, anomalous = df[~df["is_anomaly"]], df[df["is_anomaly"]]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=normal[x], y=normal[y], mode="markers", name="Normal",
                                      marker=dict(color=PALETTE[1], size=6, opacity=0.6)))
            fig.add_trace(go.Scatter(x=anomalous[x], y=anomalous[y], mode="markers", name="Anomaly",
                                      marker=dict(color="#f43f5e", size=10, symbol="x", opacity=0.9,
                                                  line=dict(width=1.5, color="#fff")),
                                      customdata=anomalous.get("anomaly_reason", pd.Series(dtype=str)).values,
                                      hovertemplate=f"{x}: %{{x}}<br>{y}: %{{y}}<br>%{{customdata}}<extra></extra>"))
            return _theme(fig, f"Anomaly Detection: {y} vs {x}")
        except Exception as e:
            return _err_fig(str(e))

    # ── Auto-selection ───────────────────────────────────────────────────

    def auto_charts(self, df: pd.DataFrame, max_charts: int = 8) -> list[tuple[str, go.Figure, str]]:
        """Returns (title, figure, chart_type) tuples chosen from the schema."""
        charts = []
        num_cols = get_numeric_cols(df)
        cat_cols = [c for c in get_categorical_cols(df) if 1 < df[c].nunique() <= 30]
        dt_cols = get_datetime_cols(df)

        if not num_cols:
            return charts

        primary, secondary = num_cols[0], (num_cols[1] if len(num_cols) > 1 else None)

        if cat_cols:
            best_cat = min(cat_cols, key=lambda c: df[c].nunique())
            charts.append((f"{primary} by {best_cat}", self.bar_chart(df, best_cat, primary), "bar"))
            charts.append((f"Count of {best_cat}", self.count_plot(df, best_cat), "count"))

        if dt_cols:
            charts.append((f"{primary} over Time", self.line_chart(df, dt_cols[0], primary), "line"))

        if secondary:
            color = cat_cols[0] if cat_cols else None
            charts.append((f"{primary} vs {secondary}", self.scatter_chart(df, primary, secondary, color), "scatter"))

        charts.append((f"Distribution of {primary}", self.histogram(df, primary), "histogram"))

        if len(num_cols) >= 3:
            charts.append(("Correlation Heatmap", self.correlation_heatmap(df), "heatmap"))

        if cat_cols and secondary:
            charts.append((f"{secondary} by {cat_cols[0]}", self.box_chart(df, cat_cols[0], secondary), "box"))

        if df.isna().sum().sum() > 0:
            charts.append(("Missing Value Map", self.missing_value_heatmap(df), "missing"))

        return charts[:max_charts]


# ── Natural-language chart interpretation ───────────────────────────────

def insight_for(chart_type: str, df: pd.DataFrame, **cols) -> str:
    """Cheap, deterministic natural-language interpretation for a chart —
    no LLM call required. Falls back to a generic note on any failure."""
    try:
        if chart_type == "histogram":
            col = cols["col"]
            s = df[col].dropna()
            skew = s.skew()
            shape = "right-skewed (long tail of high values)" if skew > 0.5 else \
                    "left-skewed (long tail of low values)" if skew < -0.5 else "roughly symmetric"
            return (f"**{col}** ranges from {s.min():.2f} to {s.max():.2f} with a mean of {s.mean():.2f}. "
                    f"The distribution is {shape} (skew = {skew:.2f}).")
        if chart_type == "bar":
            x, y = cols["x"], cols["y"]
            agg = df.groupby(x)[y].mean().sort_values(ascending=False)
            if len(agg):
                return (f"**{agg.index[0]}** leads on average {y} ({agg.iloc[0]:.2f}), "
                        f"{(agg.iloc[0] - agg.iloc[-1]):.2f} higher than the lowest ({agg.index[-1]}).")
        if chart_type == "scatter":
            x, y = cols["x"], cols["y"]
            pair = df[[x, y]].dropna()
            if len(pair) >= 5:
                r = pair[x].corr(pair[y])
                strength = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"
                direction = "positive" if r > 0 else "negative"
                return f"There is a {strength} {direction} relationship between **{x}** and **{y}** (r = {r:.2f})."
        if chart_type == "correlation":
            return "Blue cells indicate positive correlation, red indicates negative. Values close to ±1 signal strong linear relationships worth investigating for redundancy or causal drivers."
        if chart_type == "missing":
            pct = round(100 * df.isna().sum().sum() / df.size, 1) if df.size else 0
            return f"{pct}% of cells are missing overall. Columns with dense red bands may need targeted collection or imputation review."
    except Exception:
        pass
    return "Review the chart for patterns, outliers, or shifts that stand out relative to the rest of the data."
