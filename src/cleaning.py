import pandas as pd
import numpy as np

def clean_data(df):
    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)
    
    # Convert all columns to appropriate types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    
    # Drop rows with any NaN
    df.dropna(inplace=True)

    # Clean and normalize the target
    if 'classification' in df.columns:
        df['classification'] = df['classification'].astype(str).str.strip().str.lower()
        df['classification'] = df['classification'].replace({'ckd': 1, 'notckd': 0, 'ckd\t': 1})

    return df

