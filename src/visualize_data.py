import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd


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
    fig, ax = plt.subplots(figsize=(12, 10))  # Increase height for better spacing
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True,
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})

    # Rotate the x and y axis labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Add tight layout
    plt.tight_layout()

    st.pyplot(fig)



def plot_pca_projection(df: pd.DataFrame, n_components: int = 2):
    if "classification" not in df.columns:
        return None

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if len(numeric_cols) < n_components:
        return None
    df_dropna = df[numeric_cols + ["classification"]].dropna()
    if df_dropna.empty:
        return None
    X = df[numeric_cols]
    y = df["classification"]

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)

    comp_df = pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(n_components)])
    comp_df["classification"] = y.values

    fig = plt.figure(figsize=(8, 6))
    if n_components == 2:
        sns.scatterplot(data=comp_df, x="PC1", y="PC2", hue="classification", palette="Set2")
        plt.title("PCA - 2D Projection")
    else:
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')
        for label in comp_df["classification"].unique():
            subset = comp_df[comp_df["classification"] == label]
            ax.scatter(subset["PC1"], subset["PC2"], subset["PC3"], label=label)
        ax.set_title("PCA - 3D Projection")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.legend()

    return fig