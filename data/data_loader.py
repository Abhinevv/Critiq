import yfinance as yf
import pandas as pd
from datetime import datetime
import os


def download_stock_data(ticker: str, period: str = "1y", interval: str = "1d"):
    """
    Downloads stock data and saves it in data/raw with timestamp.
    Produces a clean CSV with Date as a normal column.
    """

    # Download data
    data = yf.download(ticker, period=period, interval=interval)

    if data.empty:
        print("No data downloaded.")
        return

    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Reset index so Date becomes a normal column
    data.reset_index(inplace=True)

    # Create timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Ensure raw directory exists
    raw_path = os.path.join("data", "raw")
    os.makedirs(raw_path, exist_ok=True)

    # Create filename
    filename = f"{ticker}_{timestamp}.csv"
    filepath = os.path.join(raw_path, filename)

    # Save file
    data.to_csv(filepath, index=False)

    print(f"Data saved to {filepath}")