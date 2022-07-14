import datetime
import pandas as pd
import plotly.graph_objects as go


def store_trend(ticker: str,
                first_date: datetime.date, 
                last_date: datetime.date,
                trend: str):
    '''
    Given a date range and the trend type, append the new classification to the
    ticker information and save the csv
    
    Parameters
    ----------
    ticker : str
        The stock identifier
    first_date : datetime.date
        The first date of the trend
    last_date : datetime.date
        The last date of the trend
    trend : str
        The type of trend (uptrend, downtrend, no trend)
    '''
    
    df = pd.read_csv(f'data/classified/{ticker}.csv')

    # Change the datetime objects to string so we can filter the dataframe
    first_date = first_date.strftime('%Y-%m-%d')
    last_date = last_date.strftime('%Y-%m-%d')
    
    # Filter for between the trend dates
    date_filter = (df['Date'] >= first_date) & (df['Date'] <= last_date)
    df.loc[date_filter, 'classification'] = trend
    
    # Save the updated csv
    df.to_csv(f'data/classified/{ticker}.csv', index = False)
    
    return


def get_candlestick_chart(df: pd.DataFrame,
                          ticker: str,
                          first_date: datetime.date,
                          last_date: datetime.date):
    '''
    Given the price dataset, return the plotly candlestick chart with the 
    selected area highlighted    

    Parameters
    ----------
    df : pd.DataFrame
        The stock price dataset
    ticker : str
        The ticker name for the title
    first_date : datetime.date
        First date for the selected region
    last_date : datetime.date
        Last date for the selected region

    Returns
    -------
    fig : The plotly candlestick chart
    '''
    
    df_selected = df[(df['Date'] >= first_date) & (df['Date'] <= last_date)]
    
    layout = go.Layout(
        title = f'{ticker} Chart',
        xaxis = {'title':'Date'},
        yaxis = {'title': 'Price'},
    )
    
    fig = go.Figure(
        layout=layout,
        data=[
            go.Candlestick(
                x = df['Date'],
                open = df['Open'], 
                high = df['High'],
                low = df['Low'],
                close = df['Close'],
                name = 'Candlestick chart'
            ),
            go.Line(x=df_selected['Date'],
                    y=df_selected['High'],
                    name='Selected area',
            ),
        ]
    )
    
    
    fig.update_xaxes(
            rangeslider_visible=False,
            rangebreaks=[{'bounds': ['sat', 'mon']}]
    )
    
    return fig