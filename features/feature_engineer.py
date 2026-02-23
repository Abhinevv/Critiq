import pandas as pd
import os
import ta


def generate_features():

    raw_path = os.path.join("data", "raw")
    processed_path = os.path.join("data", "processed")
    os.makedirs(processed_path, exist_ok=True)

    files = [f for f in os.listdir(raw_path) if f.endswith(".csv")]

    if not files:
        raise FileNotFoundError("No raw data files found.")

    all_data = []

    for file in files:

        filepath = os.path.join(raw_path, file)

        # Extract ticker from filename
        # Assuming filename format like: AAPL_2026-02-22_23-29-47.csv
        ticker = file.split("_")[0]

        df = pd.read_csv(filepath)

        # Ensure Date column exists
        if "Date" not in df.columns:
            df.rename(columns={df.columns[0]: "Date"}, inplace=True)

        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)

        # Convert numeric columns
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        df.dropna(inplace=True)

        # === Technical Indicators ===
        df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
        df["EMA_20"] = ta.trend.ema_indicator(df["Close"], window=20)
        df["RSI_14"] = ta.momentum.rsi(df["Close"], window=14)
        df["MACD"] = ta.trend.macd(df["Close"])
        df["BB_upper"] = ta.volatility.bollinger_hband(df["Close"])
        df["BB_lower"] = ta.volatility.bollinger_lband(df["Close"])

        df.dropna(inplace=True)

        # Add ticker column
        df["Ticker"] = ticker

        all_data.append(df)

    # Combine all stocks
    combined_df = pd.concat(all_data, ignore_index=True)

    # Sort by Date (important for time split later)
    combined_df.sort_values(["Date", "Ticker"], inplace=True)

    output_file = os.path.join(processed_path, "combined_features.csv")
    combined_df.to_csv(output_file, index=False)

    print(f"Combined features saved to {output_file}")