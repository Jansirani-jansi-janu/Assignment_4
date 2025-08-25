# app.py

import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from preprocess import load_and_preprocess
import base64

# ------------------ Streamlit Page Config ------------------
st.set_page_config(page_title="Voice Gender Classifier", layout="centered")

# ------------------ Background Image (Local File -> Base64) ------------------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Path to your local background image
img_path = "C:/Users/Admin/Desktop/Jansi/Project/Human Voice Classification and Clustering/background.jpg"
img_base64 = get_base64_of_bin_file(img_path)

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{img_base64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);  /* Transparent header */
}}

[data-testid="stToolbar"] {{ 
    right: 2rem;
}}

h1, h2, h3, h4, h5, h6, p, div, span {{
    color: white !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.7);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ------------------ App Title ------------------
st.title("🎙️ Human Voice Classification and Clustering")
st.caption("Built with Machine Learning + Streamlit")

# ------------------ Load trained model ------------------
@st.cache_resource
def load_model():
    return joblib.load("voice_classifier.pkl")

model = load_model()

# ------------------ File Uploader ------------------
uploaded_file = st.file_uploader("📂 Upload a CSV file with voice features", type=["csv"])

if uploaded_file:
    st.success("✅ File uploaded successfully!")

    try:
        # Load and preprocess uploaded data (skip PCA to match trained model)
        X_train, X_test, y_train, y_test, scaler, df, _ = load_and_preprocess(
            uploaded_file, n_pca_components=None
        )

        # Tabs for better UI
        tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "✅ Model Evaluation", "🔮 Predictions"])

        with tab1:
            st.subheader("🔍 Preview of Uploaded Data")
            st.dataframe(df.head())

            # ✨ Feature Engineering Concept Section
            st.markdown(
                """
                <div style="background-color: rgba(0,0,0,0.6); padding: 15px; border-radius: 10px; margin-top:15px;">
                    <h3 style="color: #FFD700;">✨ Feature Engineering Concept</h3>
                    <p style="color: #00FFCC; font-size:16px;">
                        In this project, we can enhance the model using <b>Feature Engineering</b>.  
                        This involves:
                        <ul>
                            <li style="color:#FFB6C1;">📊 Creating new features such as row mean, standard deviation, skewness, and kurtosis.</li>
                            <li style="color:#7FFFD4;">🎼 Applying <b>PCA (Principal Component Analysis)</b> to reduce noise and highlight important patterns.</li>
                            <li style="color:#FFA07A;">⚡ Selecting the most important features that improve classification accuracy.</li>
                        </ul>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with tab2:
            st.subheader("📈 Model Evaluation")
            predictions = model.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            st.metric(label="Accuracy", value=f"{acc*100:.2f}%")

            # Confusion Matrix Heatmap
            cm = confusion_matrix(y_test, predictions)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Male", "Female"], yticklabels=["Male", "Female"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        with tab3:
            st.subheader("🔮 Prediction Results")

            # Map numbers to labels
            label_map = {0: "Male", 1: "Female"}
            results_df = pd.DataFrame({
                "Actual": y_test.map(label_map).values,
                "Predicted": pd.Series(predictions).map(label_map).values
            })

            st.dataframe(results_df.head(10))

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
