from data.data_loader import download_stock_data
from features.feature_engineer import generate_features


if __name__ == "__main__":

    DOWNLOAD = False  # Set True only when you want new raw data

    if DOWNLOAD:
        download_stock_data("AAPL", period="1y", interval="1d")

    generate_features()