import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


def scale_data(test_size=0.2):
    """
    Loads latest_features.csv,
    performs time-series split,
    scales features safely,
    saves train/test scaled datasets and scaler.
    """

    # Load processed features
    filepath = os.path.join("data", "processed", "latest_features.csv")
    df = pd.read_csv(filepath)

    # Separate Date column
    dates = df["Date"]
    df = df.drop(columns=["Date"])

    # Time-series split (no shuffling)
    split_index = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    # Initialize scaler
    scaler = StandardScaler()

    # Fit ONLY on training data
    train_scaled = scaler.fit_transform(train_df)

    # Transform test data using same scaler
    test_scaled = scaler.transform(test_df)

    # Convert back to DataFrame
    train_scaled_df = pd.DataFrame(train_scaled, columns=train_df.columns)
    test_scaled_df = pd.DataFrame(test_scaled, columns=test_df.columns)

    # Save scaled datasets
    scaled_path = os.path.join("data", "scaled")
    os.makedirs(scaled_path, exist_ok=True)

    train_scaled_df.to_csv(os.path.join(scaled_path, "train_scaled.csv"), index=False)
    test_scaled_df.to_csv(os.path.join(scaled_path, "test_scaled.csv"), index=False)

    # Save scaler object
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, os.path.join("models", "scaler.pkl"))

    print("Scaling complete.")
    print("Train & Test scaled datasets saved.")
    print("Scaler saved to models/scaler.pkl")