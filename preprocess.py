# preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_and_preprocess(csv_path, n_pca_components=None):
    df = pd.read_csv(csv_path)

    # Drop missing values
    df.dropna(inplace=True)

    # Features and labels
    X = df.drop(columns=['label'])
    y = df['label']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA only if specified
    if n_pca_components is not None:
        n_pca = min(n_pca_components, X_scaled.shape[1])
        pca = PCA(n_components=n_pca)
        X_final = pca.fit_transform(X_scaled)
    else:
        X_final = X_scaled
        pca = None

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler, df, pca


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, df, pca = load_and_preprocess(
        "C:/Users/Admin/Desktop/Jansi/Project/Human Voice Classification and Clustering/data/vocal_gender_features_new.csv",
        n_pca_components=None  # keep full features for compatibility
    )
    print("Data loaded successfully. Shape:", df.shape)
    print("Train set:", X_train.shape)
    print("Test set:", X_test.shape)
