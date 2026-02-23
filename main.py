from data.data_loader import download_stock_data
from features.feature_engineer import generate_features
from preprocessing.scaler import scale_data


if __name__ == "__main__":

    DOWNLOAD = False

    if DOWNLOAD:
        download_stock_data("AAPL", period="1y", interval="1d")

    generate_features()
    scale_data()