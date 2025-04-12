import streamlit as st
import pandas as pd
import subprocess
from src import load_data, cleaning, explore_data, visualize_data, predict

st.set_page_config(page_title="Détection Maladie Rénale", layout="wide")
st.title("🧪 Détection de la Maladie Rénale")

# File uploader
uploaded_file = st.file_uploader("📁 Importer un fichier CSV", type=["csv"])

# Charger les données dans la session si uploadé
if uploaded_file is not None:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.success("Données chargées avec succès !")

# Vérification si les données sont disponibles
if 'df' in st.session_state and st.session_state.df is not None:
    df = st.session_state.df
    st.subheader("🔍 Aperçu des données")
    st.dataframe(df.head())

    # Nettoyage simple
    if st.button("🧼 Nettoyage simple (dropna)"):
        df = cleaning.clean_dataset(df)
        st.session_state.df = df
        st.success("Nettoyage simple effectué.")

    # Nettoyage avancé
    if st.button("🔧 Nettoyage avancé"):
        df = cleaning.clean_dataset(df)
        st.session_state.df = df
        st.success("Nettoyage avancé effectué.")

    # Exploration
    if st.button("📊 Explorer les données"):
        st.write("Données du DataFrame")
        st.dataframe(df)

    # Visualisations
    if st.button("📊 Valeurs manquantes"):
        visualize_data.show_missing_values(df)

    if st.button("📈 Distribution des classes"):
        visualize_data.plot_class_distribution(df)

    if st.button("📉 Histogrammes"):
        visualize_data.plot_histograms(df)

    if st.button("🧮 Matrice de corrélation"):
        visualize_data.plot_correlation_matrix(df)

    if st.button("📦 Boxplots des variables"):
        visualize_data.plot_boxplots(df)
    if st.button("🔧 Entraîner le modèle"):
    # Utiliser subprocess pour appeler le script Python d'entraînement
        try:
            st.info("L'entraînement du modèle commence...")
            result = subprocess.run(["python", "train_model.py"], capture_output=True, text=True)
            st.success("Modèle entraîné et sauvegardé avec succès !")
            st.text(result.stdout)  # Afficher les logs de sortie
        except Exception as e:
            st.error(f"Erreur lors de l'entraînement : {e}")
    
    # Prédiction
    if st.button("🔮 Prédire la maladie rénale"):
        # On peut ici créer un sous-ensemble des données (une ligne à la fois ou un groupe de données)
        # Par exemple, ici on prend la première ligne du DataFrame pour prédiction
        prediction_result = predict.predict_result(df.iloc[0:1])  # Prédiction sur la première ligne
        st.write(f"Résultat de la prédiction: {prediction_result}")
else:
    st.info("➡️ Importez un fichier CSV pour commencer.")
