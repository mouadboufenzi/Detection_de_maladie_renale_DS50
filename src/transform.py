"""
Transform the "clean DataFrame" output by clean_data() into a machine learning feature matrix.

Features
--------
1. Apply RobustScaler to continuous/ordinal numeric columns to reduce the impact of outliers.
2. Pass through binary columns (0/1) directly.
3. Return (X, y, preprocessor) for convenient training and reuse in production.
4. Provide save/load utility functions.

Usage
-----
from src.load_data import load_dataset
from src.cleaning import clean_data
from src.transform import prepare_features, save_preprocessor

df = clean_data(load_dataset("data/kidney_disease.csv"))
X, y, prep = prepare_features(df, fit=True)
save_preprocessor(prep, "artifacts/preprocess.joblib")
"""

from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


# ----- Column grouping: consistent with cleaning.py -----------------------------
NUMERIC_COLS = [
    "age",
    "bp",
    "sg",
    "al",
    "su",
    "bgr",
    "bu",
    "sc",
    "sod",
    "pot",
    "hemo",
    "pcv",
    "wc",
    "rc",
]

# Binary columns encoded as 0/1 (no scaling)
BINARY_COLS = [
    "rbc",
    "pc",
    "pcc",
    "ba",
    "htn",
    "dm",
    "cad",
    "appet",
    "pe",
    "ane",
    # Missing indicators
    "rbc_missing",
    "wc_missing",
    "rc_missing",
]

TARGET_COL = "classification"


# ------------------------------------------------------------------
def _build_preprocessor() -> ColumnTransformer:
    """Build ColumnTransformer —— Numeric: RobustScaler, Binary: passthrough"""

    numeric_pipe = Pipeline(
        steps=[
            ("scaler", RobustScaler(with_centering=True, with_scaling=True)),
        ]
    )

    # remainder='drop' to discard columns not explicitly listed
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_COLS),
            ("bin", "passthrough", BINARY_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


# ------------------------------------------------------------------
def prepare_features(
    df: pd.DataFrame,
    *,
    fit: bool = True,
    preprocessor: Optional[ColumnTransformer] = None,
) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
    """Split DataFrame into X/y and (fit)-transform
    Parameters
    ----------
    df : pd.DataFrame
        Input should come from cleaning.clean_data()
    fit : bool, default True
        Whether to fit the scaler on this batch of data (True during training, False during inference)
    preprocessor : ColumnTransformer or None
        If an existing object is provided, reuse it; otherwise, create a new one

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)
    preprocessor : ColumnTransformer
    """
    if preprocessor is None:
        preprocessor = _build_preprocessor()

    X_df = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].to_numpy(dtype=int)

    if fit:
        X = preprocessor.fit_transform(X_df)
    else:
        X = preprocessor.transform(X_df)

    return X, y, preprocessor


# ------------------------------------------------------------------
def save_preprocessor(preprocessor: ColumnTransformer, filepath: str | Path) -> None:
    """Serialize ColumnTransformer to disk"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, filepath)


def load_preprocessor(filepath: str | Path) -> ColumnTransformer:
    """Deserialize ColumnTransformer"""
    return joblib.load(filepath)