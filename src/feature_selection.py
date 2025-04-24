import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def select_important_features(df: pd.DataFrame, top_k: int = 10):
    """
    Selects the top_k most important features based on feature importance from a Random Forest model.
    
    Parameters:
        df (pd.DataFrame): The preprocessed DataFrame (including the target column).
        top_k (int): Number of top features to retain.

    Returns:
        pd.DataFrame: A new DataFrame with only the selected features and the target column.
    """
    # Remove the 'id' feature (if present)
    df = df.drop(columns=["id"], errors='ignore')

    # Separate features and target
    X = df.drop(columns=["classification"])
    y = df["classification"]

    # Fit Random Forest to get feature importances
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Get top_k important features
    feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = feature_importances.nlargest(top_k).index.tolist()

    # Return DataFrame with top features + target
    df_selected = df[top_features + ["classification"]]
    return df_selected
