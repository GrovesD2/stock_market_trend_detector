import os
import pandas as pd
import yfinance as yf

def get_downloaded_tickers():
    '''
    Get a list of all tickers that have been downloaded
    '''
    return [file.split('.csv')[0] for file in os.listdir('data/classified')]


def move_data_to_classified(tickers: list):
    '''
    For each downloaded ticker data, move to the classified folder. If the 
    data already exists in the classified folder, the data is appended with
    the most recent data (and labelled as unclassified)

    Parameters
    ----------
    tickers : list
        A list of tickers to search over

    Returns
    -------
    None
    '''
    
    for ticker in tickers:
            
        # This logic appends the newly downloaded data to the data in the 
        # classified folder, if the file exists
        if os.path.isfile(f'data/classified/{ticker}.csv'):
            
            df_master = pd.read_csv(f'data/master/{ticker}.csv')
            df_master['classification'] = 'unclassified'
            
            df_classified = pd.read_csv(f'data/classified/{ticker}.csv')
            
            df = pd.concat([
                df_classified,
                df_master[df_master['Date'] > df_classified['Date'].max()]
            ])
            
        else:
            df = pd.read_csv(f'data/master/{ticker}.csv')
            df['classification'] = 'unclassified'
        
        df.to_csv(f'data/classified/{ticker}.csv', index = False)
    

def download_data(tickers: list):
    '''
    Obtain historical daily price data for all tickers specified. The outcome
    is a csv file of price data for each ticker in the data folder.

    Parameters
    ----------
    tickers : list
        A list of the tickers to download the data for

    Returns
    -------
    None
    '''
    
    print('Downloading the data from yahoo finance')
        
    data = yf.download(
        tickers = tickers,
        interval = '1D',
        period = 'max',
        group_by = 'ticker',
        auto_adjust = False,
        prepost = False,
        threads = True,
        proxy = None
    )
            
    
    print('Data downloaded. Saving the csv files in the data directory.')
    if len(tickers) > 1:
        
        data = data.T
        
        for ticker in tickers:
            
            # Sometimes tickers can fail to download
            try:
                df = data.loc[ticker.upper(), :].T.sort_index().dropna()
                df.to_csv(f'data/master/{ticker}.csv', index = True)
            except Exception as e:
                print('Error occurred:', e)
        
    else:
        data.to_csv(f'data/master/{tickers[0]}.csv', index = True)
            
    return