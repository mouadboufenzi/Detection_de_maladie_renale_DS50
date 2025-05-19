import streamlit as st
import pandas as pd
from io import StringIO, BytesIO
import joblib
from sklearn.model_selection import train_test_split

from src.load_data      import load_dataset                 # raw CSV loader
from src.cleaning       import clean_data                   # full cleaning / imputation
from src.explore_data   import show_missing_data, plot_distributions, show_pairplot
from src.visualize_data import show_correlation_matrix, plot_boxplots, plot_pca_projection
from src.model_training import compare_models_with_cv, plot_heatmap
from src.model_evaluation import evaluate_model_on_test, plot_confusion

st.set_page_config(page_title="CKD Detection", layout="wide")
st.title("🔬 Chronic Kidney Disease Detection")

st.write(
    "This interactive app walks through the entire pipeline of predicting Chronic "
    "Kidney Disease (CKD) — from raw CSV to a downloadable trained model."
)

# 1. Upload raw data 
st.markdown("### 📁 Step 1 – Upload CKD dataset (CSV)")
uploaded_file = st.file_uploader("Upload the dataset", type=["csv"])


# helper: convert DataFrame to CSV string (for download button)
def _df_to_csv(df: pd.DataFrame) -> str:
    buf = StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


# Main workflow starts after file upload
if uploaded_file:

    # 1-A • Load raw data
    raw_df = load_dataset(uploaded_file)
    st.success("✅ Dataset loaded")
    st.dataframe(raw_df.head())

    st.markdown("#### Missing-value overview (raw)")
    show_missing_data(raw_df)

    # 2 • Cleaning / Imputation
    st.markdown("### 🧹 Step 2 – Data cleaning")
    clean_df = clean_data(raw_df)
    st.success("✅ Cleaning completed")
    st.dataframe(clean_df.head())

    # 3 • Quick EDA  (still on cleaned data)
    st.markdown("### 📊 Step 3 – Exploratory Data Analysis")
    plot_distributions(clean_df, cols_per_row=4)
    plot_boxplots(clean_df, cols_per_row=4)
    show_pairplot(clean_df)
    show_correlation_matrix(clean_df)
    st.markdown("#### 🔻 PCA Visualization (optional)")
    n_pca = st.slider("Select number of PCA components", 2, 3, 2)
    fig_pca = plot_pca_projection(clean_df, n_components=n_pca)
    if fig_pca:
        st.pyplot(fig_pca)
    else:
        st.warning("PCA visualization could not be generated.")

    # 4 • Train / test split  (NO transformation done yet)
    st.markdown("### ✂️ Step 4 – Train / Test split (80 / 20)")
    X_full = clean_df.drop(columns=["classification"])
    y_full = clean_df["classification"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.20, stratify=y_full, random_state=42
    )
    st.write(f"Training set : {X_train.shape}  Test set : {X_test.shape}")

    # 5 • Model comparison via leakage-free Pipelines (CV)
    st.markdown("### 🤖 Step 5a – Model comparison WITHOUT PCA (5-fold CV on training set)")
    results_no_pca, model_dict_no_pca = compare_models_with_cv(X_train, y_train, use_pca=False)
    st.dataframe(results_no_pca.style.format(precision=4))
    st.pyplot(plot_heatmap(results_no_pca))

    st.markdown("### 🤖 Step 5b – Model comparison WITH PCA (5-fold CV on training set)")
    results_pca, model_dict_pca = compare_models_with_cv(X_train, y_train, use_pca=True, n_components=2)
    st.dataframe(results_pca.style.format(precision=4))
    st.pyplot(plot_heatmap(results_pca))


    # Assume you have:
    # results_no_pca, model_dict_no_pca
    # results_pca, model_dict_pca

    # Let user pick which model dict to evaluate
    eval_choice = st.selectbox("Choose pipeline to evaluate", ["No PCA", "With PCA"])

    if eval_choice == "No PCA":
        results_df = results_no_pca
        model_dict = model_dict_no_pca
    else:
        results_df = results_pca
        model_dict = model_dict_pca

    # Step 6: Evaluate all models in chosen dict
    st.markdown("### 🔍 Step 6 – Confusion matrices on the 20 % test set")

    for name, pipe in model_dict.items():
        pipe.fit(X_train, y_train)
        eval_res = evaluate_model_on_test(pipe, X_test, y_test)
        st.subheader(f"Confusion Matrix — {name}")
        st.pyplot(plot_confusion(eval_res["Confusion Matrix"]))

    # Step 7: Highlight best model in chosen dict
    best_model_name = results_df["Accuracy"].idxmax()
    best_pipe = model_dict[best_model_name].fit(X_train, y_train)
    best_eval = evaluate_model_on_test(best_pipe, X_test, y_test)

    st.markdown("### 🏆 Step 7 – Best model performance on Test set")
    for k in ["Accuracy", "ROC AUC", "Precision", "Recall"]:
        st.write(f"**{k}** : {best_eval[k]:.4f}")
    st.json(best_eval["Classification Report"])

    # 8 • Download trained pipeline
    st.markdown("### 💾 Step 8 – Download trained pipeline")
    bytes_buf = BytesIO()
    joblib.dump(best_pipe, bytes_buf)
    bytes_buf.seek(0)
    st.download_button(
        label="⬇️ Download pipeline (.joblib)",
        data=bytes_buf,
        file_name=f"ckd_{best_model_name.lower().replace(' ', '_')}_pipeline.joblib",
        mime="application/octet-stream",
    )

    # 9 • Optional – download the cleaned train / test splits
    st.markdown("### 📄 Optional – Download cleaned splits")
    st.download_button("⬇️ Train CSV", _df_to_csv(pd.concat([X_train, y_train], axis=1)),
                       file_name="train_clean.csv")
    st.download_button("⬇️ Test  CSV", _df_to_csv(pd.concat([X_test,  y_test],  axis=1)),
                       file_name="test_clean.csv")
