import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def show_missing_values(df):
    plt.figure(figsize=(12,6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False)
    plt.title('Valeurs manquantes par colonne')
    st.pyplot(plt.gcf()) 
    plt.clf()

def plot_class_distribution(df):
    plt.figure(figsize=(6,4))
    sns.countplot(x='classification', data=df)
    plt.title('Distribution des cas de maladie rénale')
    st.pyplot(plt.gcf()) 
    plt.clf()

def plot_histograms(df):
    fig = plt.figure(figsize=(15,10))
    df.hist(ax=fig.gca())
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

def plot_correlation_matrix(df):
    # Filter only numeric columns for correlation matrix
    df_numeric = df.select_dtypes(include=['number'])

    # Plot correlation matrix for numeric columns only
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_numeric.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Matrice de corrélation (numeric columns only)')
    st.pyplot(plt.gcf())
    plt.clf()

    try:
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Matrice de corrélation (entire dataframe)')
        st.pyplot(plt.gcf())
        plt.clf()
    except ValueError:
        st.warning("The correlation matrix cannot be calculated for non-numeric columns.")


def plot_boxplots(df):
    numeric_columns = df.select_dtypes(include='number').columns
    for col in numeric_columns:
        plt.figure(figsize=(6,3))
        sns.boxplot(x=df[col])
        plt.title(f'Distribution de {col}')
        st.pyplot(plt.gcf())
        plt.clf()
