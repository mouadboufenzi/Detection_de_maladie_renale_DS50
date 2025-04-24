from sklearn.preprocessing import LabelEncoder, StandardScaler

def encode_and_scale(df):
    df = df.copy()
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('classification')
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, label_encoders, scaler

