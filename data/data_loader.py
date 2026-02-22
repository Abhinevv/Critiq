import yfinance as yf
import pandas as pd
from datetime import datetime
import os


def download_stock_data(ticker: str, period: str = "1y", interval: str = "1d"):
    """
    Downloads stock data and saves it in data/raw with timestamp.
    """

    # Download data
    data = yf.download(ticker, period=period, interval=interval)

    if data.empty:
        print("No data downloaded.")
        return

    # Create timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Ensure raw directory exists
    raw_path = os.path.join("data", "raw")
    os.makedirs(raw_path, exist_ok=True)

    # Create filename
    filename = f"{ticker}_{timestamp}.csv"
    filepath = os.path.join(raw_path, filename)

    # Save file
    data.to_csv(filepath)

    print(f"Data saved to {filepath}")