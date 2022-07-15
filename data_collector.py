import pandas as pd
import streamlit as st
from utils import dash, tickers
 

ticker = st.sidebar.selectbox(
    'Ticker', 
    tickers.get_downloaded_tickers(),
)
days_ahead = st.sidebar.number_input(
    'Trading days to plot',
    value = 60,
    min_value = 1,
    max_value = 120,
    step = 1,    
)

# Get the dataframe
df = pd.read_csv(f'data/classified/{ticker}.csv')

if 'unclassified' in df['classification'].unique():

    # Find the first unclassified index
    idx_unc = df[df['classification'] == 'unclassified'].index.values[0]
    
    # Produce a candlestick chart n months ahead 
    df_chart = df[idx_unc:(idx_unc + int(days_ahead))]
    df_chart['Date'] = pd.to_datetime(df_chart['Date']).dt.date
    
    # Get the other sidebar inputs 
    first_date = df_chart['Date'].min()
    
    last_date = st.sidebar.date_input(
        'Last date in the trend',
        value = df_chart['Date'].max(),
        min_value = df_chart['Date'].min(),
        max_value = df_chart['Date'].max(),
    )
    
    # Display the chart
    st.plotly_chart(
        dash.get_candlestick_chart(df_chart, ticker, first_date, last_date),
        use_container_width = False,
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.button(
            'Uptrend', 
            on_click = dash.store_trend, 
            args = (ticker, first_date, last_date, 'uptrend')
        )
    with col2:
        st.button(
            'Downtrend', 
            on_click = dash.store_trend, 
            args = (ticker, first_date, last_date, 'downtrend')
        )
    with col3:
        st.button(
            'No trend', 
            on_click = dash.store_trend, 
            args = (ticker, first_date, last_date, 'no trend')
        )
        
else:
    
    st.write(f'{ticker} has been completed :)')