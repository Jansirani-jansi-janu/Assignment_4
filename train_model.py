# train_model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from preprocess import load_and_preprocess

def train_classification_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n Accuracy: {acc:.4f}")
    print("\n Classification Report:\n", classification_report(y_test, y_pred))
    print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def save_model(model, filename='voice_classifier.pkl'):
    joblib.dump(model, filename)
    print(f"\n Model saved as '{filename}'")

if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test, scaler, df = load_and_preprocess("D:/Human Voice Classification and Clustering/data/vocal_gender_features_new.csv")
    
    # Train model
    model = train_classification_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model)
