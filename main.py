import streamlit as st
import pandas as pd
from io import StringIO, BytesIO
import joblib

from src.load_data import load_dataset
from src.cleaning import clean_data
# Yangran modified
# from src.transform import encode_and_scale
# from src.feature_selection import select_important_features
from src.transform import prepare_features
from src.feature_selection import RFSelect
from src.explore_data import show_missing_data, plot_distributions
from src.visualize_data import show_correlation_matrix, plot_boxplots
from src.model_training import compare_models_with_cv, plot_heatmap
from sklearn.model_selection import train_test_split
from src.model_evaluation import evaluate_model_on_test, plot_confusion
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

st.set_page_config(page_title="CKD Detection", layout="wide")
st.title("🔬 Chronic Kidney Disease Detection App")

st.markdown("### 📁 Step 1: Upload Dataset")
uploaded_file = st.file_uploader("Upload your CKD dataset CSV", type=["csv"])

def download_buffer(df):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

if uploaded_file:
    df = load_dataset(uploaded_file)
    st.session_state.raw_df = df
    st.success("✅ Dataset loaded successfully!")
    st.write(df.head())

    st.markdown("### 🧹 Step 2: Data Cleaning")
    df_cleaned = clean_data(df)
    st.session_state.df_cleaned = df_cleaned
    st.success("✅ Cleaning completed!")
    st.write(df_cleaned.head())
    show_missing_data(df_cleaned)

    st.markdown("### 📊 Step 3: Exploratory Data Analysis")
    plot_distributions(df_cleaned, cols_per_row=4)

    st.markdown("### 📦 Step 4: Outlier Detection")
    plot_boxplots(df_cleaned, cols_per_row=4)

    st.markdown("### 🔗 Step 5: Correlation Matrix")
    show_correlation_matrix(df_cleaned)

    # Yangran modified
    # st.markdown("### 🧬 Step 6: Encoding and Scaling")
    # df_transformed, encoders, scaler = encode_and_scale(df_cleaned)
    # st.session_state.df_transformed = df_transformed
    # st.success("✅ Data encoded and scaled!")
    # st.write(df_transformed.head())
    
    # st.markdown("### 🔎 Step 7: Feature Selection")
    # df_selected = select_important_features(df_transformed, top_k=10)
    # st.session_state.df_selected = df_selected
    # st.success("✅ 10 important features selected!")
    # st.write(df_selected.head())

    # 🧬 Step 6: Preprocessing  (RobustScaler etc.)
    st.markdown("### 🧬 Step 6: Preprocessing")
    # prepare_features() 只在训练阶段 fit=True
    X_np, y_np, preproc = prepare_features(df_cleaned, fit=True)
    
    # ndarray ↔ DataFrame（colom from preproc）
    feature_names = preproc.get_feature_names_out()
    df_transformed = pd.DataFrame(X_np, columns=feature_names)
    df_transformed["classification"] = y_np
    
    st.session_state.df_transformed = df_transformed
    st.session_state.preprocessor = preproc          # can reuse further
    
    st.success("✅ Preprocessing completed!")
    st.write(df_transformed.head())

    # 🔎 Step 7: Feature Selection  (RF embedded)
    st.markdown("### 🔎 Step 7: Feature Selection")

    X_fs = df_transformed.drop(columns=["classification"])
    y_fs = df_transformed["classification"]

    selector = RFSelect(top_k=10)
    X_sel = selector.fit_transform(X_fs, y_fs)        # returns DataFrame

    df_selected = pd.concat([X_sel, y_fs], axis=1)

    st.session_state.df_selected = df_selected
    st.session_state.selector = selector
    st.success("✅ 10 important features selected!")
    st.write(df_selected.head())

    st.markdown("### 💾 Step 8: Download Clean Data")

    # Split the selected data
    X = st.session_state.df_selected.drop(columns=["classification"])
    y = st.session_state.df_selected["classification"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save them to session state
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

    # Prepare files for download
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = X_test  # 'classification' intentionally removed

    st.download_button("⬇️ Download Train Dataset", data=download_buffer(train_df), file_name="train_data.csv")
    st.download_button("⬇️ Download Test Dataset", data=download_buffer(test_df), file_name="test_data.csv")

    # ─────────────────────────────────────────────
    # STEP 9: MODEL TRAINING AND EVALUATION (TRAINING SET)
    # ─────────────────────────────────────────────
    st.markdown("### 🤖 Step 9: Model Training and Evaluation")

    # Retrieve training set from session state
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train

    # Train and compare models
    results_df = compare_models_with_cv(X_train, y_train)

    st.subheader("📊 Model Comparison on Training Set (5-fold CV)")
    st.dataframe(results_df)

    # Plot heatmap
    st.pyplot(plot_heatmap(results_df).gcf())

    # 🧪 Step 10: Evaluate Best Model on Test Set
    st.markdown("### 🧪 Step 10: Final Evaluation on Test Set")

    # Re-train models and pick the one with best CV score
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }

    best_model_name = results_df["Accuracy"].idxmax()
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)

    # Evaluate on test data
    eval_results = evaluate_model_on_test(best_model, X_test, y_test)

    st.subheader(f"✅ Evaluation of {best_model_name} on Test Data")
    st.write(f"**Accuracy**: {eval_results['Accuracy']:.4f}")
    st.write(f"**ROC AUC**: {eval_results['ROC AUC']:.4f}")
    st.write(f"**Precision**: {eval_results['Precision']:.4f}")
    st.write(f"**Recall**: {eval_results['Recall']:.4f}")
    st.json(eval_results["Classification Report"])

    # Display confusion matrix
    st.subheader("🔍 Confusion Matrix")
    st.pyplot(plot_confusion(eval_results["Confusion Matrix"]))

    # 💾 Step 11: Save and Export Best Model
    st.markdown("### 💾 Step 11: Save and Export Best Model")

    # Save the model to a temporary buffer
    model_buffer = BytesIO()
    joblib.dump(best_model, model_buffer)
    model_buffer.seek(0)

    st.download_button(
        label="⬇️ Download Trained Model (.joblib)",
        data=model_buffer,
        file_name=f"{best_model_name.replace(' ', '_').lower()}_ckd_model.joblib",
        mime="application/octet-stream"
    )

