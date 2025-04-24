import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def plot_boxplots(df, cols_per_row=4):
    st.markdown("### 📦 Outlier Detection via Boxplots")
    st.write("Boxplots help us visually detect outliers—extremely high or low values that could distort training or influence model bias.")

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    total = len(numeric_cols)
    
    for i in range(0, total, cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            if i + j < total:
                col = numeric_cols[i + j]
                with cols[j]:
                    fig, ax = plt.subplots()
                    sns.boxplot(y=df[col], ax=ax)
                    ax.set_title(f'Boxplot: {col}')
                    st.pyplot(fig)

def show_correlation_matrix(df):
    st.markdown("### 🔗 Feature Correlation Matrix")
    st.write("This helps identify relationships between features. Highly correlated features might be redundant. This can also hint at multicollinearity problems.")

    # ✅ Filter only numeric columns before correlation
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    if numeric_df.shape[1] < 2:
        st.warning("Not enough numeric features for correlation matrix.")
        return

    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    st.pyplot(fig)



