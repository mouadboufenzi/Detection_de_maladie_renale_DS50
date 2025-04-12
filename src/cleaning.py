# cleaning.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_dataset(df):
    df.fillna(df.mode().iloc[0], inplace=True)

    le = LabelEncoder()
    df['classification'] = le.fit_transform(df['classification'])

    for col in ['pcv', 'wc', 'rc']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    num_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_columns] = df[num_columns].fillna(df[num_columns].mean())

    cat_columns = df.select_dtypes(include=['object']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.fillna(x.mode()[0]))

    return df
