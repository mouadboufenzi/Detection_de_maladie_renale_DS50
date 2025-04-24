import pandas as pd
import numpy as np

def load_dataset(file):
    df = pd.read_csv(file)
    df.replace('?', np.nan, inplace=True)
    df.columns = [col.strip().lower() for col in df.columns]
    return df
