"""
model_training.py  – build leakage-free pipelines, run CV, and
                     plot a heatmap of CV metrics.
"""

from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.transform import _build_preprocessor
from src.feature_selection import RFSelect
from sklearn.decomposition import PCA



# ──────────────────────────────────────────────────────────────────────
def _build_pipeline(clf, use_pca=False, n_components=2):
    prep = _build_preprocessor()
    prep.set_output(transform="pandas")

    steps = [
        ("prep", prep),
        ("select", RFSelect(top_k=10)),
    ]

    if use_pca:
        steps.append(("pca", PCA(n_components=n_components)))

    steps.append(("clf", clf))

    return Pipeline(steps)



# ──────────────────────────────────────────────────────────────────────
def compare_models_with_cv(
    X_train: pd.DataFrame,
    y_train,
    cv: int = 5,
    use_pca: bool = False,
    n_components: int = 2,
) -> tuple[pd.DataFrame, dict]:
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    }

    scoring = {
        "Accuracy":  "accuracy",
        "ROC AUC":   "roc_auc",
        "Precision": "precision",
        "Recall":    "recall",
    }

    results, pipeline_dict = {}, {}

    for name, clf in models.items():
        pipe = _build_pipeline(clf, use_pca=use_pca, n_components=n_components)
        cv_res = cross_validate(
            pipe, X_train, y_train,
            cv=cv, scoring=scoring, n_jobs=-1, error_score="raise"
        )

        results[name] = {
            k: round(cv_res[f"test_{k}"].mean(), 4) for k in scoring.keys()
        }

        pipeline_dict[name] = _build_pipeline(clf, use_pca=use_pca, n_components=n_components)

    return pd.DataFrame(results).T, pipeline_dict


# ──────────────────────────────────────────────────────────────────────
def plot_heatmap(df: pd.DataFrame):
    """Return a matplotlib Figure containing a heatmap of the CV metrics."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df, annot=True, cmap="Blues", fmt=".3f", ax=ax)
    ax.set_title("Cross-validation metrics (mean scores)")
    fig.tight_layout()
    return fig
