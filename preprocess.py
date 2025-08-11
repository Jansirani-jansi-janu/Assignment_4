# preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)

    # Drop any missing values
    df.dropna(inplace=True)

    # Separate features and label
    X = df.drop(columns=['label'])
    y = df['label']

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler, df

# Optional test code when running this file directly
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, df = load_and_preprocess("D:/Human Voice Classification and Clustering/data/vocal_gender_features_new.csv")
    print("Data loaded successfully. Shape:", df.shape)
    print("Train set:", X_train.shape)
    print("Test set:", X_test.shape)
