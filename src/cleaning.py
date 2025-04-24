"""
Based on the DataFrame obtained from load_dataset():
1. Identify numeric columns → convert to float.
2. Add missing indicators for high-missing columns.
3. Impute numeric columns with median; categorical columns with mode.
4. Standardize yes/no, normal/abnormal to 0/1.
5. Encode target column 'classification' to 0/1.
6. Drop 'id' column.
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


# ----------- Field grouping (fixed according to data dictionary/business rules) -----------
NUMERIC_COLS: List[str] = [
    "age", "bp", "sg", "al", "su",
    "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wc", "rc",
]

BINARY_YESNO: List[str] = ["htn", "dm", "cad", "pe", "ane"]
BINARY_NORMAL: List[str] = ["rbc", "pc", "pcc", "ba", "appet"]

# High-missing rate columns that need both imputation and tracking
HIGH_MISS_NUMERIC: List[str] = ["rbc", "wc", "rc"]


def _strip_and_lower(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Strip whitespace and convert column values to lowercase"""
    df[cols] = (
        df[cols]
        .apply(lambda s: s.astype(str).str.strip().str.lower().replace({"nan": np.nan}))
    )
    return df


def clean_data(df: pd.DataFrame, add_missing_indicator: bool = True) -> pd.DataFrame:
    """Complete missing value imputation + encoding, return a clean DataFrame"""

    df = df.copy()

    # 1️⃣  Standardize character format ------------------------------------------
    cat_cols = BINARY_YESNO + BINARY_NORMAL + ["classification"]
    df = _strip_and_lower(df, cat_cols)

    # 2️⃣  Convert to numeric (set unconvertible values to NaN) ----------------------------
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3️⃣  Add missing indicators for high-missing columns ------------------------
    if add_missing_indicator:
        for col in HIGH_MISS_NUMERIC:
            df[f"{col}_missing"] = df[col].isna().astype(int)

    # 4️⃣  Impute missing values --------------------------------------------
    num_imputer = SimpleImputer(strategy="median")
    df[NUMERIC_COLS] = num_imputer.fit_transform(df[NUMERIC_COLS])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[BINARY_YESNO + BINARY_NORMAL] = cat_imputer.fit_transform(
        df[BINARY_YESNO + BINARY_NORMAL]
    )

    # 5️⃣  Encode categories ----------------------------------------------
    yes_no_map = {"yes": 1, "no": 0}
    df[BINARY_YESNO] = df[BINARY_YESNO].replace(yes_no_map)

    normal_map = {
        "normal": 0,
        "abnormal": 1,
        "present": 1,
        "notpresent": 0,
        "good": 0,
        "poor": 1,
    }
    df[BINARY_NORMAL] = df[BINARY_NORMAL].replace(normal_map)

    # 6️⃣  Process target column -------------------------------------------
    if "classification" in df.columns:
        df["classification"] = (
            df["classification"]
            .str.replace(r"[^a-z]", "", regex=True)  # Remove extra characters
            .map({"ckd": 1, "notckd": 0})
        )

    # 7️⃣  Drop uninformative columns -----------------------------------------
    if "id" in df.columns:
        df.drop(columns="id", inplace=True)

    return df