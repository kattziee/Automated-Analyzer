"""
domain.py — Heuristic dataset-domain detection.

Scores each known business domain by counting how many of its keyword
tokens appear (as substrings) in the dataset's column names, then returns
the best match plus a confidence score. Falls back to "Generic" when no
domain scores meaningfully above the others.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import DOMAINS


@dataclass
class DomainResult:
    domain: str
    confidence: float
    matched_keywords: list[str]
    runner_up: str | None = None


def detect_domain(df: pd.DataFrame) -> DomainResult:
    cols_lower = " | ".join(str(c).lower().replace("_", " ") for c in df.columns)
    scores: dict[str, tuple[int, list[str]]] = {}

    for domain, keywords in DOMAINS.keywords.items():
        hits = [kw for kw in keywords if kw.replace("_", " ") in cols_lower]
        scores[domain] = (len(hits), hits)

    ranked = sorted(scores.items(), key=lambda kv: kv[1][0], reverse=True)
    best_domain, (best_count, best_hits) = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 and ranked[1][1][0] > 0 else None

    if best_count == 0:
        return DomainResult("Generic", 0.0, [], None)

    total_cols = max(len(df.columns), 1)
    confidence = min(1.0, best_count / max(total_cols * 0.3, 1))
    return DomainResult(best_domain, round(confidence, 2), best_hits, runner_up)
