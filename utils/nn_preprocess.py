import os
import time
import itertools
import numpy as np
import pandas as pd
import multiprocessing as mp

# Global variables
DATA_LOC = 'data/classified/'
SAVE_DIR = 'data/nn_inputs/'
RETURN_COLS = ['Open', 'Low', 'High', 'Close', 'Volume']
DROP_COLS = ['Adj Close', 'Date']
MOVING_AVERAGES = [10, 20, 50]
TREND_REPLACEMENT = {
    'no trend': 0,
    'uptrend': 1,
    'downtrend': 2,
}

def time_series(df: pd.DataFrame,
                col: str,
                name: str,
                lags: range) -> pd.DataFrame:
    '''
    Form the lagged cols (i.e like transposing a handful of rows)
    '''
    return df.assign(**{
        f'{name}_t-{lag}': col.shift(lag)
        for lag in lags
    })


def get_lagged_returns(df: pd.DataFrame,
                       cols: list,
                       lags: range) -> pd.DataFrame:
    '''
    For each of the input cols, get corresponding cols for the lagged returns
    '''
    for col in cols:
        return_col = df[col]/df[col].shift(1)-1
        df = time_series(df, return_col, f'{col}_ret', lags)
        
    return df
    

def candle_thickness(df: pd.DataFrame,
                     lags: range) -> pd.DataFrame:
    '''
    Return the lagged time series for the candle thickness
    '''
    open_to_close = df['Open']/df['Close']-1
    high_to_low = df['High']/df['Low']-1
    
    df = time_series(df, open_to_close, 'open_to_close', lags)
    df = time_series(df, high_to_low, 'high_to_low', lags)

    return df


def get_ma_features(df: pd.DataFrame,
                    averages: list,
                    lags: range) -> pd.DataFrame:
    '''
    Get the lagged returns for the moving averages
    '''
    
    for avg in averages:
        ma = df['Close'].rolling(avg).mean()
        df = time_series(df, ma/ma.shift(1)-1, f'{avg}_ma', lags)
        
    return df


def reshape_rnn(x: np.array,
                n_lags: int) -> np.array:
    '''
    If an RNN-type network is wanted, reshape the input so that it is a 3D
    array of the form (sample, time series, feature).
    
    Parameters
    ----------
    x : np_arr
        The data to reshape.
    n_lags : int
        The number of time-lags used.
    Returns
    -------
    x_new : np_arr
        The reshaped x array for the RNN layers.
    '''
    
    # Calculate the number of features we have in the nn (assumes all features
    # are of the same length)
    num_feats = x.shape[1]//n_lags
    
    # Initialise the new x array with the correct size
    x_new = np.zeros((x.shape[0], n_lags, num_feats))
    
    # Populate this array through iteration
    for n in range(0, num_feats):
        x_new[:, :, n] = x[:, n*n_lags:(n+1)*n_lags]
    
    return x_new


def save_data(data: pd.DataFrame,
              train_split: float,
              max_lags: int):
    '''
    Given the processed data for the NN, split into training and testing sets
    and reshape ready for the LSTM. The results are then saved to the nn_inputs
    folder
    '''
    
    vals = data.values
    
    np.random.shuffle(vals)
    
    # Split into the training and testing arrays
    train_split = int(train_split*vals.shape[0])
    
    x_train = vals[:train_split, 1:]
    y_train = vals[:train_split, 0]
    
    x_test = vals[train_split:, 1:]
    y_test = vals[train_split:, 0]
    
    # Reshape for RNN type nets, and save
    np.save(
        SAVE_DIR + 'x_train.npy',
        reshape_rnn(x_train, max_lags),
    )
    np.save(
        SAVE_DIR + 'x_test.npy',
        reshape_rnn(x_test, max_lags),
    )
    
    np.save(SAVE_DIR + 'y_train.npy', y_train)
    np.save(SAVE_DIR + 'y_test.npy', y_test)
    
    # Save the csv file with the data (for visual inspection/modification)
    data.to_csv(SAVE_DIR + 'input_data.csv', index = False)
    
    return


def get_ticker_features(df: pd.DataFrame,
                        lags: range) -> pd.DataFrame:
    '''
    Get the NN features for a specific ticker
    '''

    df = get_lagged_returns(df, RETURN_COLS, lags)
    df = get_ma_features(df, MOVING_AVERAGES, lags)
    df = candle_thickness(df, lags)
    
    # This makes sure no ill processed data slips through
    return (
        df
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    

def get_nn_data_parallel(ticker: str,
                         lags: range):
    '''
    This function gets the processed NN data for a single ticker, so it can
    be called in a multi-processing way
    '''
    
    df = pd.read_csv(DATA_LOC + ticker + '.csv')
    df = df[df['classification'] != 'unclassified']
    
    if len(df) > 0:
        return (
            get_ticker_features(df, lags)
            .drop(columns=RETURN_COLS+DROP_COLS)
            .replace(TREND_REPLACEMENT)
        )

    else:
        return None


def get_nn_data(lags: range) -> pd.DataFrame:
    
    classified_tickers = [
        file.split('.csv')[0] for file in os.listdir(DATA_LOC)    
    ]
    
    dfs = []
    with mp.Pool(processes = mp.cpu_count()) as pool:
        dfs = pool.starmap(
            get_nn_data_parallel,
            itertools.product(
                *[
                    classified_tickers,
                    [lags],
                ]
            ),
        )

    return pd.concat(dfs)


def process_for_classifier(config: dict):
    
    lags = range(0, config['days back'] + 1)
    
    df = pd.read_csv(DATA_LOC + config['ticker'] + '.csv')
    data = (
        get_ticker_features(df, lags)
        .drop(columns=RETURN_COLS+DROP_COLS)
        .replace(TREND_REPLACEMENT)
    )
    
    return (
        df.iloc[data.index].reset_index(drop = True), 
        reshape_rnn(data.values[:, 1:], max(lags))
    )
    

def main(config: dict):
    
    t0 = time.time()
    
    lags = range(0, config['days back'] + 1)
    
    print('Starting to generate the NN data')

    save_data(
        get_nn_data(lags),
        config['train split'],
        max(lags)
    )
    
    print('Time taken to get all the NN data :', time.time()-t0)
    
    return
