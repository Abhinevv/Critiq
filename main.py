from data.data_loader import download_stock_data


if __name__ == "__main__":
    download_stock_data("AAPL", period="1y", interval="1d")