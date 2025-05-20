"""
model_training.py – Build leakage-free pipelines with optional PCA.
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from src.cleaning import DataCleaner
from src.feature_selection import RFSelect

# ───────────────────────────────────────────────────────────────
def _build_pipeline(clf, *, use_pca: bool = False, n_components: int = 2) -> Pipeline:
    """Cleaner ➜ RFSelect(DataFrame) ➜ StandardScaler ➜ (PCA) ➜ Classifier"""
    steps = [
        ("cleaner", DataCleaner(add_missing_indicator=True)),
        ("selector", RFSelect(top_k=10)),
        ("scaler",  StandardScaler()),
    ]

    if use_pca:
        steps.append(("pca", PCA(n_components=n_components)))
    else:
        steps.append(("identity", "passthrough"))

    steps.append(("classifier", clf))
    return Pipeline(steps, verbose=False)

# ───────────────────────────────────────────────────────────────
def compare_models_with_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    cv: int = 5,
    use_pca: bool = False,
    n_components: int = 2,
) -> tuple[pd.DataFrame, dict]:

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42,
        ),
    }

    scoring = {
        "Accuracy":  "accuracy",
        "ROC AUC":   "roc_auc",
        "Precision": "precision",
        "Recall":    "recall",
    }

    results, pipeline_dict = {}, {}

    for name, clf in models.items():
        try:
            pipe = _build_pipeline(clf, use_pca=use_pca, n_components=n_components)

            cv_res = cross_validate(
                pipe, X_train, y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                error_score="raise",
                return_train_score=False,
            )

            results[name] = {m: cv_res[f"test_{m}"].mean() for m in scoring}
            pipeline_dict[name] = pipe
        except Exception as e:
            st.error(f"Error with {name}: {e}")

    df = (
        pd.DataFrame(results).T.sort_values("Accuracy", ascending=False)
        if results else pd.DataFrame()
    )
    return df, pipeline_dict

# ───────────────────────────────────────────────────────────────
def plot_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Return a heat-map figure or a placeholder if df 为空"""
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No results to display", ha="center", va="center")
        ax.axis("off")
        return fig

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        df.T,
        annot=True, fmt=".3f",
        cmap="Blues",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Cross-Validation Metrics (Mean Scores)")
    ax.set_xlabel("Models")
    ax.set_ylabel("Metrics")
    return fig
