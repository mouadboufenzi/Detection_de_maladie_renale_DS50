"""
Load kidney_disease.csv and perform the **lightest** cleaning:
1. Recognize '?' as missing values.
2. Convert all column names to lowercase and remove leading/trailing whitespace.
3. Remove leading/trailing whitespace and tabs from object columns.
"""

from pathlib import Path
from typing import Union, Sequence

import numpy as np
import pandas as pd


def load_dataset(
    filepath: Union[str, Path],
    na_values: Sequence[str] = ("?", "\\t", " "),
) -> pd.DataFrame:
    """Load dataset and return DataFrame (basic standardization only, **no imputation**)

    Parameters
    ----------
    filepath : Path to the CSV file.
    na_values : Characters to be considered as missing values.

    Returns
    -------
    pd.DataFrame
    """
    # 1. Read the CSV and mark '?' as NaN
    df = pd.read_csv(filepath, na_values=na_values)

    # 2. Standardize column names: remove whitespace and convert to lowercase
    df.columns = df.columns.str.strip().str.lower()

    # 3. Remove tabs and whitespace from object columns
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = (
        df[obj_cols]
        .apply(lambda s: s.str.strip().str.replace(r"\t", "", regex=True))
    )

    return df