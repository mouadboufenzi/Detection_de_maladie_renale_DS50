"""
model_evaluation.py – Test-set metrics & visualisations
"""
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def evaluate_model_on_test(model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = np.nan

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "ROC AUC":  roc_auc,
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "Classification Report": classification_report(
            y_test, y_pred, output_dict=True
        ),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
    }


def plot_confusion(cm):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return fig
