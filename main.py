from data.data_loader import download_stock_data
from features.feature_engineer import generate_features
from preprocessing.scaler import scale_data


if __name__ == "__main__":

    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]

    # Download data for multiple stocks
    for ticker in tickers:
        download_stock_data(ticker, period="1y", interval="1d")

    # Generate combined features
    generate_features()

    # Apply global scaling
    scale_data()