from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from preprocess import load_and_preprocess

def perform_kmeans(X, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    return kmeans, labels, score

def plot_clusters(X, labels):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title("KMeans Clustering")
    plt.xlabel("Feature 1 (PCA)")
    plt.ylabel("Feature 2 (PCA)")
    plt.show()

if __name__ == "__main__":
    # Load data with feature engineering + PCA
    X_train, X_test, y_train, y_test, scaler, df, pca = load_and_preprocess(
        "C:/Users/Admin/Desktop/Jansi/Project/Human Voice Classification and Clustering/data/vocal_gender_features_new.csv",
        n_pca_components=10
    )

    # Start MLflow experiment
    mlflow.set_experiment("Voice Clustering with Feature Engineering")

    with mlflow.start_run():
        # Run KMeans
        kmeans, labels, score = perform_kmeans(X_train, n_clusters=2)

        # Log params & metrics
        mlflow.log_param("n_clusters", 2)
        mlflow.log_metric("silhouette_score", score)

        # Log PCA explained variance
        for i, var in enumerate(pca.explained_variance_ratio_):
            mlflow.log_metric(f"pca_variance_component_{i+1}", var)

        # Log model
        mlflow.sklearn.log_model(kmeans, "kmeans_voice_clustering")

        print(f"Silhouette Score: {score:.4f}")
        print("KMeans model logged with MLflow (with feature engineering + PCA)!")
