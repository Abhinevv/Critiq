import pandas as pd
import os
import ta


def get_latest_raw_file():
    raw_path = os.path.join("data", "raw")
    files = [f for f in os.listdir(raw_path) if f.endswith(".csv")]

    if not files:
        raise FileNotFoundError("No raw data files found.")

    latest_file = max(
        files,
        key=lambda x: os.path.getctime(os.path.join(raw_path, x))
    )

    return os.path.join(raw_path, latest_file)


def generate_features():
    # Load latest raw file
    filepath = get_latest_raw_file()

    df = pd.read_csv(filepath, index_col=0)
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d")

    # Sort by date
    df.sort_index(inplace=True)

    # Ensure numeric columns
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop missing values
    df.dropna(inplace=True)

    # ===== Add Technical Indicators =====

    # Simple Moving Average
    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)

    # Exponential Moving Average
    df["EMA_20"] = ta.trend.ema_indicator(df["Close"], window=20)

    # RSI
    df["RSI_14"] = ta.momentum.rsi(df["Close"], window=14)

    # MACD
    df["MACD"] = ta.trend.macd(df["Close"])

    # Bollinger Bands
    df["BB_upper"] = ta.volatility.bollinger_hband(df["Close"])
    df["BB_lower"] = ta.volatility.bollinger_lband(df["Close"])

    # Drop NaNs created by rolling indicators
    df.dropna(inplace=True)

    # ===== Save processed data (overwrite mode) =====

    processed_path = os.path.join("data", "processed")
    os.makedirs(processed_path, exist_ok=True)

    filename = "latest_features.csv"
    save_path = os.path.join(processed_path, filename)

    df.to_csv(save_path)

    print(f"Processed features saved to {save_path}")