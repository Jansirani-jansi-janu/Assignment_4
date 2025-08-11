# app.py

import streamlit as st
import pandas as pd
import joblib
from preprocess import load_and_preprocess
from train_model import evaluate_model

st.set_page_config(page_title="Voice Gender Classifier", layout="centered")

st.title("🎙️ Human Voice Gender Classification App")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("voice_classifier.pkl")

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file with voice features", type=["csv"])

if uploaded_file:
    st.success(" File uploaded successfully!")
    
    # Load and preprocess uploaded data
    try:
        X_train, X_test, y_train, y_test, scaler, df = load_and_preprocess(uploaded_file)
        st.write("🔍 Preview of uploaded data:")
        st.dataframe(df.head())

        # Evaluate on uploaded data
        st.subheader(" Model Evaluation on Uploaded Data")
        evaluate_model(model, X_test, y_test)

        # Predict and show result
        predictions = model.predict(X_test)
        results_df = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": predictions
        })
        st.subheader(" Prediction Results")
        st.dataframe(results_df.head(10))

    except Exception as e:
        st.error(f" Error processing file: {e}")
