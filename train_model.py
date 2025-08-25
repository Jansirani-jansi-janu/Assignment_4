import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocess import load_and_preprocess

def train_classification_model(X_train, y_train, n_estimators=100):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n Accuracy: {acc:.4f}")
    print("\n Classification Report:\n", classification_report(y_test, y_pred))
    print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return acc

if __name__ == "__main__":
    # Load data with feature engineering + PCA
    X_train, X_test, y_train, y_test, scaler, df, pca = load_and_preprocess(
        "C:/Users/Admin/Desktop/Jansi/Project/Human Voice Classification and Clustering/data/vocal_gender_features_new.csv",
        n_pca_components=10
    )

    # Start MLflow experiment
    mlflow.set_experiment("Voice Gender Classification with Feature Engineering")

    with mlflow.start_run():
        # Train model
        model = train_classification_model(X_train, y_train, n_estimators=100)

        # Evaluate
        accuracy = evaluate_model(model, X_test, y_test)

        # Log parameters & metrics
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)

        # Log PCA explained variance
        for i, var in enumerate(pca.explained_variance_ratio_):
            mlflow.log_metric(f"pca_variance_component_{i+1}", var)

        # Log model
        mlflow.sklearn.log_model(model, "voice_classifier_model")

        print("\n Model logged with MLflow (with feature engineering + PCA)!")
