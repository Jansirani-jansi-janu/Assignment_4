# Assignment_4
# Human Voice Gender Classification and Clustering

## 📌 Project Overview
This project classifies human voices as **Male** or **Female** using a machine learning model (Random Forest Classifier) 
and also provides an option to perform **unsupervised clustering** (K-Means) on voice features.

🎤 Human Voice Classification and Clustering

This project classifies human voices as Male or Female using Machine Learning and also performs clustering to find natural groupings in the data. A Streamlit web app is built for interactive use.

📌 Features

Upload a CSV file with voice acoustic features
Data Preview: See uploaded data
Model Evaluation: Accuracy, Confusion Matrix, Metrics
Predictions: View actual vs predicted results
Clustering: Group voices using KMeans

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
pip install pandas scikit-learn streamlit matplotlib joblib

- Trains the classifier
- Saves model to `voice_classifier.pkl`

### 2. Launch the Streamlit App
Run:
- Opens app in your browser
- Upload CSV with the same feature columns used for training

---

## 📊 Dataset

Contains acoustic features of voice samples (43 features).
Labels: Male (0), Female (1).
Example features:
Mean Spectral Centroid
Spectral Bandwidth
Zero Crossing Rate
Roll-off frequencies

⚙️ Preprocessing
Data Cleaning (remove missing values)
Standardization using StandardScaler
Train-Test Split (80% – 20%)
(Optional) Dimensionality Reduction (PCA)

🤖 Models Used
Random Forest Classifier (Supervised)
Accuracy: ~99.5%
Confusion Matrix shows very few misclassifications
KMeans Clustering (Unsupervised)
k = 2 (Male / Female)
Evaluated using Silhouette Score (~0.87)

🖥️ Streamlit App
The app has 3 main sections:
📂 Data Preview – view uploaded dataset
📊 Model Evaluation – accuracy & confusion matrix
🔮 Predictions – actual vs predicted labels

Example:
meanfreq,sd,median,Q25,Q75,label
0.059781,0.064241,0.056286,0.031302,0.082253,male
0.066009,0.067420,0.068275,0.043109,0.089885,female

