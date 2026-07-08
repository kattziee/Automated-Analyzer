"""
ml_engine.py — Lightweight AutoML layer.

Automatically:
  - selects task type (regression / classification) from target dtype & cardinality
  - builds a preprocessing pipeline (impute + scale + one-hot encode)
  - trains and cross-validates a small candidate model set
  - reports metrics, feature importance, and the best model
  - offers standalone PCA and KMeans clustering with auto-k selection

Data leakage is avoided by fitting all preprocessing exclusively on the
training fold within a scikit-learn Pipeline / cross_val_score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, silhouette_score,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from config import THRESH
from utils.helpers import get_numeric_cols, get_categorical_cols, is_textual


@dataclass
class ModelResult:
    name: str
    metrics: dict
    cv_mean: float
    cv_std: float


@dataclass
class MLReport:
    task: str  # "regression" | "classification" | "unavailable"
    target: str
    results: list = field(default_factory=list)
    best_model_name: str = ""
    feature_importance: Optional[pd.DataFrame] = None
    message: str = ""


def _build_preprocessor(df: pd.DataFrame, feature_cols: list[str]) -> ColumnTransformer:
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=20))])

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    return ColumnTransformer(transformers, remainder="drop")


def detect_task(df: pd.DataFrame, target: str) -> str:
    if target not in df.columns:
        return "unavailable"
    series = df[target].dropna()
    if len(series) < THRESH.min_rows_ml:
        return "unavailable"
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 15:
        return "regression"
    return "classification"


class MLEngine:
    def auto_train(self, df: pd.DataFrame, target: str, feature_cols: Optional[list[str]] = None) -> MLReport:
        task = detect_task(df, target)
        if task == "unavailable":
            return MLReport(task, target, message="Target column unsuitable or insufficient rows (need ≥ "
                             f"{THRESH.min_rows_ml}).")

        data = df.dropna(subset=[target]).copy()
        feature_cols = feature_cols or [c for c in data.columns if c != target]
        feature_cols = [c for c in feature_cols if c in data.columns and c != target]
        if not feature_cols:
            return MLReport(task, target, message="No usable feature columns.")

        # Drop very high-cardinality object columns (likely IDs) to avoid explosion
        feature_cols = [c for c in feature_cols if not (
            is_textual(data[c]) and data[c].nunique() > 0.9 * len(data)
        )]
        if not feature_cols:
            return MLReport(task, target, message="All candidate features look like identifiers; skipped.")

        X = data[feature_cols]
        y = data[target]

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=THRESH.test_size, random_state=THRESH.random_state,
                stratify=y if task == "classification" and y.nunique() > 1 else None,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=THRESH.test_size, random_state=THRESH.random_state
            )

        preprocessor = _build_preprocessor(data, feature_cols)
        candidates = self._candidates(task)

        results = []
        best_pipe, best_score, best_name = None, -np.inf, ""
        cv = min(THRESH.cv_folds, max(2, len(X_train) // 10))

        for name, model in candidates:
            pipe = Pipeline([("prep", preprocessor), ("model", model)])
            try:
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                if task == "regression":
                    metrics = {
                        "R²": round(float(r2_score(y_test, preds)), 4),
                        "MAE": round(float(mean_absolute_error(y_test, preds)), 4),
                        "RMSE": round(float(mean_squared_error(y_test, preds) ** 0.5), 4),
                    }
                    score = metrics["R²"]
                    scoring = "r2"
                else:
                    metrics = {
                        "Accuracy": round(float(accuracy_score(y_test, preds)), 4),
                        "F1 (weighted)": round(float(f1_score(y_test, preds, average="weighted", zero_division=0)), 4),
                    }
                    score = metrics["Accuracy"]
                    scoring = "accuracy"

                cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                results.append(ModelResult(name, metrics, round(float(cv_scores.mean()), 4),
                                            round(float(cv_scores.std()), 4)))
                if score > best_score:
                    best_score, best_pipe, best_name = score, pipe, name
            except Exception as e:
                results.append(ModelResult(name, {"error": str(e)}, float("nan"), float("nan")))

        fi_df = self._feature_importance(best_pipe, feature_cols, data) if best_pipe is not None else None

        return MLReport(task=task, target=target, results=results, best_model_name=best_name,
                         feature_importance=fi_df,
                         message=f"Trained {len(candidates)} model(s); best = {best_name}." if best_pipe else
                                 "All candidate models failed to fit.")

    @staticmethod
    def _candidates(task: str) -> list[tuple[str, object]]:
        rs = THRESH.random_state
        if task == "regression":
            return [
                ("Linear Regression", LinearRegression()),
                ("Random Forest Regressor", RandomForestRegressor(n_estimators=150, random_state=rs, n_jobs=-1)),
            ]
        return [
            ("Logistic Regression", LogisticRegression(max_iter=500, random_state=rs)),
            ("Random Forest Classifier", RandomForestClassifier(n_estimators=150, random_state=rs, n_jobs=-1)),
        ]

    @staticmethod
    def _feature_importance(pipe: Pipeline, feature_cols: list[str], df: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            model = pipe.named_steps["model"]
            prep = pipe.named_steps["prep"]
            names = prep.get_feature_names_out()
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                coef = model.coef_
                importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
            else:
                return None
            fi = pd.DataFrame({"feature": names, "importance": importances})
            fi = fi.sort_values("importance", ascending=False).head(20).reset_index(drop=True)
            fi["importance"] = fi["importance"].round(4)
            return fi
        except Exception:
            return None

    # ── Unsupervised: PCA + Clustering ──────────────────────────────────

    def run_pca(self, df: pd.DataFrame, n_components: int = 2) -> dict:
        num_cols = get_numeric_cols(df)
        if len(num_cols) < 2:
            return {"error": "Need ≥ 2 numeric columns for PCA."}
        X = df[num_cols].fillna(df[num_cols].median())
        if len(X) < 3:
            return {"error": "Need ≥ 3 rows for PCA."}
        n_components = min(n_components, len(num_cols), len(X))
        try:
            scaled = StandardScaler().fit_transform(X)
            pca = PCA(n_components=n_components, random_state=THRESH.random_state)
            components = pca.fit_transform(scaled)
            comp_df = pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(n_components)], index=X.index)
            return {
                "components": comp_df,
                "explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
                "loadings": pd.DataFrame(pca.components_.T, index=num_cols,
                                          columns=[f"PC{i+1}" for i in range(n_components)]).round(3),
            }
        except Exception as e:
            return {"error": str(e)}

    def run_clustering(self, df: pd.DataFrame, k: Optional[int] = None) -> dict:
        num_cols = get_numeric_cols(df)
        if len(num_cols) < 2:
            return {"error": "Need ≥ 2 numeric columns for clustering."}
        X = df[num_cols].fillna(df[num_cols].median())
        if len(X) < 10:
            return {"error": "Need ≥ 10 rows for clustering."}
        scaled = StandardScaler().fit_transform(X)

        try:
            if k is None:
                k = self._auto_k(scaled)
            model = KMeans(n_clusters=k, random_state=THRESH.random_state, n_init=10)
            labels = model.fit_predict(scaled)
            sil = silhouette_score(scaled, labels) if k > 1 and len(set(labels)) > 1 else float("nan")
            return {"labels": labels, "k": k, "silhouette": round(float(sil), 4) if not np.isnan(sil) else None,
                    "columns_used": num_cols}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _auto_k(X: np.ndarray) -> int:
        best_k, best_score = 2, -1
        max_k = min(THRESH.max_kmeans_k, len(X) - 1, 8)
        for k in range(2, max(3, max_k + 1)):
            try:
                labels = KMeans(n_clusters=k, random_state=THRESH.random_state, n_init=10).fit_predict(X)
                if len(set(labels)) < 2:
                    continue
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score, best_k = score, k
            except Exception:
                continue
        return best_k
