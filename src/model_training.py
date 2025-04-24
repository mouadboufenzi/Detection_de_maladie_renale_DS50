import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

def compare_models_with_cv(X_train, y_train):
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }

    metrics = ["accuracy", "roc_auc", "precision", "recall"]
    results = {}

    for name, model in models.items():
        results[name] = {}
        for metric in metrics:
            score = cross_val_score(model, X_train, y_train, cv=5, scoring=metric).mean()
            results[name][metric.capitalize()] = round(score, 4)

    return pd.DataFrame(results).T

def plot_heatmap(results_df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(results_df, annot=True, cmap="YlGnBu", fmt=".4f")
    plt.title("Model Comparison on Training Set (CV=5)")
    plt.tight_layout()
    return plt
