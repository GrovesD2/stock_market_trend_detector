from utils import tickers

TICKERS = ['BTC-USD', 'MSFT']

if __name__ == '__main__':

    tickers.download_data(TICKERS)
    tickers.move_data_to_classified(TICKERS)