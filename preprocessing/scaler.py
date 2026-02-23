import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler


def scale_data(test_size=0.2):

    input_path = os.path.join("data", "processed", "combined_features.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError("combined_features.csv not found. Run feature engineering first.")

    df = pd.read_csv(input_path)

    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)

    split_date = df["Date"].quantile(1 - test_size)

    train_df = df[df["Date"] <= split_date].copy()
    test_df = df[df["Date"] > split_date].copy()

    feature_cols = [
        col for col in df.columns
        if col not in ["Date", "Close", "Ticker"]
    ]

    # ---- FORCE FLOAT TYPE BEFORE SCALING ----
    train_df[feature_cols] = train_df[feature_cols].astype(float)
    test_df[feature_cols] = test_df[feature_cols].astype(float)

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    scaler = StandardScaler()
    scaler.fit(X_train)

    train_df[feature_cols] = scaler.transform(X_train)
    test_df[feature_cols] = scaler.transform(X_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    os.makedirs("data/scaled", exist_ok=True)

    train_df.to_csv("data/scaled/train_scaled.csv", index=False)
    test_df.to_csv("data/scaled/test_scaled.csv", index=False)

    print("Global scaling complete.")
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")