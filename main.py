import streamlit as st
import pandas as pd
from io import StringIO, BytesIO
import joblib
from sklearn.model_selection import train_test_split

from src.load_data import load_dataset
from src.cleaning import DataCleaner
from src.explore_data import show_missing_data, plot_distributions, show_pairplot
from src.visualize_data import show_correlation_matrix, plot_boxplots, plot_pca_projection
from src.model_training import compare_models_with_cv, plot_heatmap
from src.model_evaluation import evaluate_model_on_test, plot_confusion

st.set_page_config(page_title="CKD Detection", layout="wide")
st.title("🔬 Chronic Kidney Disease Detection")

st.write(
    "This interactive app walks through the entire pipeline of predicting Chronic "
    "Kidney Disease (CKD) - from raw CSV to a downloadable trained model."
)

# 1 ─ Upload raw data --------------------------------------------------------
st.markdown("### 📁 Step 1 – Upload CKD dataset (CSV)")
uploaded_file = st.file_uploader("Upload the dataset", type=["csv"])

def _df_to_csv(df: pd.DataFrame) -> str:
    buf = StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()

if uploaded_file:
    # 1-A • Load raw data
    raw_df = load_dataset(uploaded_file)
    st.success("✅ Dataset loaded")
    st.dataframe(raw_df.head())

    st.markdown("#### Missing-value overview (raw)")
    show_missing_data(raw_df)

    # 2 ─ Train / Test split --------------------------------------------------
    st.markdown("### ✂️ Step 2 – Train / Test split (80 / 20)")
    if "classification" not in raw_df.columns:
        st.error("Dataset must contain 'classification' column")
        st.stop()

    X_full = raw_df.drop(columns=["classification"])
    y_full = raw_df["classification"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_full,
        y_full,
        test_size=0.20,
        stratify=y_full,
        random_state=42,
    )

    # create train_df and test_df
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    st.write(f"Raw training set : {train_df.shape}  Raw test set : {test_df.shape}")

    # 3 ─ Quick EDA on Raw Data (Before Cleaning) ----------------------------
    st.markdown("### 📊 Step 3 – EDA on Raw Training Data (Pre-Cleaning)")
    plot_distributions(train_df, cols_per_row=4)
    plot_boxplots(train_df, cols_per_row=4)
    show_pairplot(train_df)
    show_correlation_matrix(train_df)

    # 4 ─ Data cleaning (no leakage) -----------------------------------------
    st.markdown("### 🧹 Step 4 – Data cleaning (no leakage)")
    cleaner = DataCleaner()
    clean_train = cleaner.fit_transform(train_df)
    clean_test = cleaner.transform(test_df)

    st.success("✅ Cleaning completed")
    st.dataframe(clean_train.head())

    # # 5 ─ Optional: EDA on Cleaned Data --------------------------------------
    # st.markdown("### 📊 Step 5 – EDA on Cleaned Training Data (Post-Cleaning)")
    # plot_distributions(clean_train, cols_per_row=4)
    # plot_boxplots(clean_train, cols_per_row=4)
    # show_pairplot(clean_train)
    # show_correlation_matrix(clean_train)

    st.markdown("#### PCA Projection (Cleaned Data)")
    pca_fig = plot_pca_projection(clean_train)
    if pca_fig:
        st.pyplot(pca_fig)

    # 5 ─ Prepare ML Features -------------------------------------------------
    st.markdown("### ✂️ Step 5 – Prepare ML Features")
    X_train_clean = clean_train.drop(columns=["classification"])
    y_train_clean = clean_train["classification"]
    X_test_clean  = clean_test.drop(columns=["classification"])
    y_test_clean  = clean_test["classification"]

    # 6 ─ Model comparison ----------------------------------------------------
    st.markdown("### 🤖 Step 6 – Model comparison (5-fold CV on training set)")
    use_pca     = st.checkbox("Enable PCA Dimensionality Reduction", value=False)
    n_components = st.slider("PCA Components", 2, 5, 2) if use_pca else 2

    results_df, model_dict = compare_models_with_cv(
        X_train_clean, y_train_clean,
        use_pca=use_pca, n_components=n_components,
    )

    st.dataframe(results_df.style.format(precision=4))
    st.pyplot(plot_heatmap(results_df))

    # 7 ─ Test set evaluation -------------------------------------------------
    if results_df.empty:
        st.warning("No model produced valid cross-validation results – "
                "please inspect the previous error messages.")
        st.stop()
    st.markdown("### 🔍 Step 7 – Confusion matrices on the 20% test set")
    for name, pipe in model_dict.items():
        pipe.fit(X_train_clean, y_train_clean)
        eval_res = evaluate_model_on_test(pipe, X_test_clean, y_test_clean)
        st.subheader(f"Confusion Matrix – {name}")
        st.pyplot(plot_confusion(eval_res["Confusion Matrix"]))

    # 8 ─ Cross-validation top performer metrics ---------------------------------
    st.markdown("### 🏆 Step 8 – Performance of the best generalizing model (CV Top Performer)")

    # Obtain the best performing model name from the cross validation results
    best_model_name = results_df["Accuracy"].idxmax()
    st.write(f"🔍 Based on cross-validation, the best generalizing model is: **{best_model_name}**")

    best_pipe = model_dict[best_model_name].fit(X_train_clean, y_train_clean)

    best_eval = evaluate_model_on_test(best_pipe, X_test_clean, y_test_clean)

    for k in ["Accuracy", "ROC AUC", "Precision", "Recall"]:
        st.write(f"**{k}** : {best_eval[k]:.4f}")

    # JSON
    st.markdown("#### 📋 Detailed Classification Report")
    st.json(best_eval["Classification Report"])


    # 9 ─ Download trained pipeline ------------------------------------------
    st.markdown("### 💾 Step 9 – Download trained pipeline")
    bytes_buf = BytesIO()
    joblib.dump(best_pipe, bytes_buf)
    bytes_buf.seek(0)
    st.download_button(
        label="⬇️ Download pipeline (.joblib)",
        data=bytes_buf,
        file_name=f"ckd_{best_model_name.lower().replace(' ', '_')}_pipeline.joblib",
        mime="application/octet-stream",
    )

    # 10 ─ Optional data download --------------------------------------------
    st.markdown("### 📄 Optional – Download cleaned splits")
    st.download_button("⬇️ Train CSV", _df_to_csv(clean_train),
                       file_name="train_clean.csv")
    st.download_button("⬇️ Test CSV", _df_to_csv(clean_test),
                       file_name="test_clean.csv")
