"""
insights.py — Deterministic, rule-based insight generator.

Produces an executive summary, key findings, risks, opportunities, and
recommendations purely from computed statistics — no external API required,
so the app is fully functional offline. An optional LLM layer (llm_client.py)
can enrich this narrative if the user supplies their own API key.
"""
from __future__ import annotations

import pandas as pd

from core.profiling import DatasetProfile
from core.domain import DomainResult
from core.analytics import AnalyticsEngine
from core.statistics_engine import significant_correlations
from core.schema import ColumnType
from utils.helpers import get_numeric_cols


def executive_summary(df: pd.DataFrame, profile: DatasetProfile, domain: DomainResult) -> str:
    rows, cols = profile.shape
    domain_txt = f"a **{domain.domain}**-style dataset" if domain.domain != "Generic" else "a general-purpose dataset"
    quality_txt = ("excellent" if profile.quality.overall >= 85 else
                   "good" if profile.quality.overall >= 70 else
                   "fair" if profile.quality.overall >= 50 else "poor")
    return (
        f"This dataset contains **{rows:,} records** across **{cols} columns** and appears to be {domain_txt} "
        f"(confidence {domain.confidence*100:.0f}%). Overall data quality is **{quality_txt}** "
        f"({profile.quality.overall}/100), driven by {profile.quality.completeness}% completeness, "
        f"{profile.quality.uniqueness}% row uniqueness, {profile.quality.consistency}% type consistency, "
        f"and {profile.quality.validity}% value validity. "
        f"There are {len(profile.numeric_cols)} numeric, {len(profile.categorical_cols)} categorical, and "
        f"{len(profile.datetime_cols)} datetime column(s) available for analysis."
    )


def key_findings(df: pd.DataFrame, profile: DatasetProfile) -> list[str]:
    findings = []
    corrs = significant_correlations(df)
    for c in corrs[:3]:
        findings.append(f"**{c['feature_1']}** and **{c['feature_2']}** show a {c['strength']} {c['direction']} "
                         f"correlation (r = {c['r']}).")

    for col in profile.numeric_cols[:8]:
        s = df[col].dropna()
        if len(s) < 5:
            continue
        skew = s.skew()
        if abs(skew) > 1:
            direction = "right" if skew > 0 else "left"
            findings.append(f"**{col}** is heavily {direction}-skewed (skew = {skew:.2f}); "
                             f"consider a log transform before modeling.")

    if profile.n_duplicate_rows > 0:
        findings.append(f"Found **{profile.n_duplicate_rows:,} duplicate row(s)** "
                         f"({100*profile.n_duplicate_rows/max(len(df),1):.1f}% of the data).")

    high_null_cols = [name for name, p in profile.schema.items() if 20 < p.missing_pct <= 60]
    if high_null_cols:
        findings.append(f"Columns with substantial missingness (20-60%): {', '.join(high_null_cols[:5])}.")

    return findings[:8] or ["No strong patterns stood out beyond baseline distributions — the data looks fairly uniform."]


def business_risks(df: pd.DataFrame, profile: DatasetProfile) -> list[str]:
    risks = []
    if profile.quality.completeness < 80:
        risks.append(f"Data completeness is only {profile.quality.completeness}% — downstream models or "
                      f"reports may be biased toward complete records.")
    if profile.n_duplicate_rows / max(len(df), 1) > 0.02:
        risks.append("A non-trivial share of duplicate rows could inflate totals or double-count entities.")

    engine = AnalyticsEngine()
    if len(df) >= 10 and profile.numeric_cols:
        outliers = engine.outlier_summary(df)
        if not outliers.empty:
            worst = outliers.iloc[0]
            if worst["pct"] > 5:
                risks.append(f"**{worst['column']}** has {worst['pct']}% statistical outliers — "
                              f"verify for data-entry errors or genuine extreme events.")

    high_card = [n for n, p in profile.schema.items() if p.inferred_type == ColumnType.MIXED]
    if high_card:
        risks.append(f"Columns with inconsistent/mixed value formats detected: {', '.join(high_card[:5])} — "
                      f"these may silently break aggregations.")

    return risks or ["No major data-quality risks detected at this scan depth."]


def opportunities(df: pd.DataFrame, profile: DatasetProfile, domain: DomainResult) -> list[str]:
    opp = []
    if len(profile.numeric_cols) >= 2:
        opp.append("Multiple numeric features are available — a regression or driver-analysis model "
                    "could quantify what most influences your key metric.")
    if profile.datetime_cols:
        opp.append("A datetime column is present — trend and seasonality analysis (or forecasting) "
                    "can reveal cyclical patterns to plan around.")
    if profile.categorical_cols:
        opp.append("Categorical segments are available for cohort or group comparisons "
                    "(e.g., ANOVA or grouped visualizations) to find over/under-performing segments.")
    if domain.domain != "Generic":
        opp.append(f"Because this looks like {domain.domain} data, domain-specific KPIs "
                    f"(e.g., {', '.join(domain.matched_keywords[:3])}) are good candidates for deeper drill-down.")
    return opp or ["Explore feature engineering from existing columns to surface additional signal."]


def recommendations(profile: DatasetProfile) -> list[str]:
    recs = []
    if profile.quality.completeness < 90:
        recs.append("Prioritize closing data-collection gaps in columns with the highest missingness before modeling.")
    if profile.n_duplicate_rows > 0:
        recs.append("Deduplicate records upstream, or confirm duplicates are intentional (e.g. re-orders).")
    if profile.n_high_cardinality_cols > 0:
        recs.append("Review high-cardinality columns — they may be identifiers that should be excluded from ML features.")
    recs.append("Re-run this analysis after each data refresh to track quality and trend drift over time.")
    return recs


def action_plan(profile: DatasetProfile) -> list[str]:
    steps = ["Review the Data Quality tab and resolve flagged missingness / duplicates.",
             "Inspect flagged anomalies in the Analytics tab for genuine business events vs. errors."]
    if profile.numeric_cols and profile.categorical_cols:
        steps.append("Use the Custom Chart Builder to compare key metrics across your top categorical segments.")
    if profile.datetime_cols:
        steps.append("Run the forecasting module to project the next period and validate against business plans.")
    steps.append("Export the cleaned dataset and share the executive summary with stakeholders.")
    return steps


def full_report(df: pd.DataFrame, profile: DatasetProfile, domain: DomainResult) -> dict:
    return {
        "executive_summary": executive_summary(df, profile, domain),
        "key_findings": key_findings(df, profile),
        "risks": business_risks(df, profile),
        "opportunities": opportunities(df, profile, domain),
        "recommendations": recommendations(profile),
        "action_plan": action_plan(profile),
    }
