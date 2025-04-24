"""
feature_selection.py
--------------------
Random-Forest Embedded Feature Selector (sklearn-compatible).

Yangran Notes
---------------
1. **No Data Leakage**: Implemented as a Transformer to fit within cross-validation folds and transform only on validation folds.
2. **Serializable**: Can be saved as part of a Pipeline, ensuring consistent column selection during online inference.
3. **Class Imbalance**: Enable `class_weight='balanced'` in the forest.
"""

from __future__ import annotations

from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted


class RFSelect(BaseEstimator, TransformerMixin):
    """
    Random-Forest Embedded Feature Selector

    Parameters
    ----------
    top_k : int or None, default 10
        Retain the top_k most important features; if None, retain all features with importance > 0.
    n_estimators : int, default 500
        Number of trees in the random forest.
    random_state : int or None
    """

    def __init__(
        self,
        top_k: Optional[int] = 10,
        n_estimators: int = 500,
        random_state: Optional[int] = 42,
    ):
        self.top_k = top_k
        self.n_estimators = n_estimators
        self.random_state = random_state

    # ------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "RFSelect expects a **pandas DataFrame** so it can access column names."
            )

        self.rf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            n_jobs=-1,
            class_weight="balanced",
            random_state=self.random_state,
            oob_score=True,
        )
        self.rf_.fit(X, y)

        importances = pd.Series(self.rf_.feature_importances_, index=X.columns)

        if self.top_k is None:
            self.selected_cols_: List[str] = importances[importances > 0].index.tolist()
        else:
            self.selected_cols_ = importances.nlargest(self.top_k).index.tolist()

        return self

    # ------------------------------------------------------------ #
    def transform(self, X):
        check_is_fitted(self, "selected_cols_")
        if isinstance(X, pd.DataFrame):
            return X[self.selected_cols_]
        # If upstream converts features to ndarray, column names are required
        raise TypeError("Input X must be a pandas DataFrame with named columns.")

    # Convenience method for ColumnTransformer / Pipeline to get column names
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "selected_cols_")
        return np.array(self.selected_cols_)

    # ------------------------------------------------------------ #
    # Convenience save/load wrapper
    def save(self, filepath: str):
        """joblib.dump(self, filepath)"""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str):
        """joblib.load(filepath)"""
        return joblib.load(filepath)