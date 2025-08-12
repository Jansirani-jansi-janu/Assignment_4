# Assignment_4
# Human Voice Gender Classification and Clustering

## 📌 Project Overview
This project classifies human voices as **Male** or **Female** using a machine learning model (Random Forest Classifier) 
and also provides an option to perform **unsupervised clustering** (K-Means) on voice features.

The project includes:
1. **Classification** – Supervised ML model to predict gender from voice features.
2. **Clustering** – Grouping similar voices without labels.
3. **Streamlit App** – Interactive interface for predictions.

---

## 📂 Project Structure
1. **preprocess.py**
   - Loads dataset from CSV
   - Cleans missing data
   - Scales features using StandardScaler
   - Splits into training and testing sets

2. **train_model.py**
   - Trains a Random Forest Classifier
   - Evaluates model (Accuracy, Classification Report, Confusion Matrix)
   - Saves trained model as `voice_classifier.pkl`

3. **cluster_model.py**
   - Performs K-Means clustering
   - Calculates silhouette score
   - Option to plot clusters

4. **app.py**
   - Streamlit app for interactive predictions
   - Upload CSV with voice features
   - Preprocess and predict using saved model
   - Displays results in a table

5. **voice_classifier.pkl**
   - Pre-trained Random Forest model saved with joblib

---

## 🛠 Requirements
Install dependencies before running:
