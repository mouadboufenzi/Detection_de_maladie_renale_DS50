"""
cleaning.py – Data cleaning transformers compatible with scikit-learn pipelines.
"""
from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

NUMERIC_COLS: List[str] = [
    "age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wc", "rc",
]
BINARY_YESNO   = ["htn", "dm", "cad", "pe", "ane"]
BINARY_NORMAL  = ["rbc", "pc", "pcc", "ba", "appet"]
HIGH_MISS_NUMERIC = ["rbc", "wc", "rc"]
TARGET_COL = "classification"


def _existing(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, add_missing_indicator: bool = True):
        self.add_missing_indicator = add_missing_indicator
        self.num_imp_, self.cat_imp_, self.fitted_columns_ = None, None, None

    # ────────────────────────────── internal helpers
    def _strip_and_lower(self, df, cols):
        cols = _existing(df, cols)
        if cols:
            df[cols] = (df[cols]
                        .apply(lambda s: (s.astype(str)
                                          .str.strip()
                                          .str.lower()
                                          .replace({"nan": np.nan}))))
        return df

    def _ensure_present(self, df, cols):
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        return df

    def _encode_target(self, df):
        if TARGET_COL in df.columns:
            df[TARGET_COL] = (df[TARGET_COL]
                              .astype(str).str.strip().str.lower()
                              .replace({"ckd": 1, "notckd": 0}))
            df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
        return df

    # ────────────────────────────── sklearn API
    def fit(self, X, y=None):
        X = X.copy().drop(columns=["id"], errors="ignore")
        X = self._ensure_present(X, NUMERIC_COLS + BINARY_YESNO + BINARY_NORMAL)
        X = self._strip_and_lower(X, BINARY_YESNO + BINARY_NORMAL)

        for col in NUMERIC_COLS:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        X = self._encode_target(X)

        if self.add_missing_indicator:
            for col in HIGH_MISS_NUMERIC:
                X[f"{col}_missing"] = X[col].isna().astype(int)

        self.num_imp_ = SimpleImputer(strategy="median").fit(X[NUMERIC_COLS])
        self.cat_imp_ = SimpleImputer(strategy="most_frequent").fit(
            X[BINARY_YESNO + BINARY_NORMAL]
        )
        self.fitted_columns_ = X.columns.tolist()
        return self

    def transform(self, X):
        X = X.copy().drop(columns=["id"], errors="ignore")
        X = self._ensure_present(X, NUMERIC_COLS + BINARY_YESNO + BINARY_NORMAL)
        X = self._strip_and_lower(X, BINARY_YESNO + BINARY_NORMAL)

        for col in NUMERIC_COLS:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        X = self._encode_target(X)                       # ← 新增

        if self.add_missing_indicator:
            for col in HIGH_MISS_NUMERIC:
                X[f"{col}_missing"] = X[col].isna().astype(int)

        X[NUMERIC_COLS] = self.num_imp_.transform(X[NUMERIC_COLS])
        X[BINARY_YESNO + BINARY_NORMAL] = self.cat_imp_.transform(
            X[BINARY_YESNO + BINARY_NORMAL]
        )

        # binary encodings
        X[BINARY_YESNO] = X[BINARY_YESNO].replace({"yes": 1, "no": 0})
        X[BINARY_NORMAL] = X[BINARY_NORMAL].replace({
            "normal": 0, "abnormal": 1,
            "present": 1, "notpresent": 0,
            "good": 0, "poor": 1,
        })

        return X.reindex(columns=self.fitted_columns_, fill_value=0)
