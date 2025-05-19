import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def show_missing_data(df):
    st.markdown("### 🧼 Missing Data Overview")
    st.write("This step shows how complete the dataset is. Missing values might indicate data collection issues or potential fields to impute or drop.")
    st.write(df.isnull().sum())

    st.markdown("#### 🔥 Heatmap of Missing Values")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    st.pyplot(fig)

def plot_distributions(df, cols_per_row=4):
    st.markdown("### 📈 Feature Distributions")
    st.write("This helps us understand the spread of numeric features and identify skewness, anomalies, or transformation needs.")

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    total = len(numeric_cols)
    
    for i in range(0, total, cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            if i + j < total:
                col = numeric_cols[i + j]
                with cols[j]:
                    fig, ax = plt.subplots()
                    sns.histplot(df[col].dropna(), kde=True, ax=ax)
                    ax.set_title(f'Distribution: {col}')
                    st.pyplot(fig)


import seaborn as sns
import matplotlib.pyplot as plt

def show_pairplot(df, max_features=5):
    st.markdown("### 🔍 Pairplot of Numeric Features")
    st.write("""
        Pairplots allow you to visualize pairwise relationships between features.
        The colors represent different classes (e.g., presence or absence of CKD), 
        which helps you spot clusters, correlations, and separations.
    """)

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if "classification" not in df.columns:
        st.warning("Target column 'classification' not found in the dataset.")
        return

    if len(numeric_cols) > max_features:
        selected_cols = st.multiselect(
            f"Select up to {max_features} features for pairplot (currently {len(numeric_cols)} numeric features):",
            numeric_cols,
            default=numeric_cols[:max_features]
        )
    else:
        selected_cols = numeric_cols

    if len(selected_cols) < 2:
        st.warning("Please select at least 2 features.")
        return

    with st.spinner("Generating pairplot..."):
        try:
            fig = sns.pairplot(df[selected_cols + ["classification"]].dropna(), hue="classification", palette="husl", diag_kind="kde")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating pairplot: {e}")
