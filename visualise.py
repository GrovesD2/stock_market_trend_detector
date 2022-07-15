import numpy as np
from utils import nn_preprocess
from tensorflow.keras.models import load_model


import plotly.io as pio
pio.renderers.default='browser'
import plotly.graph_objects as go

CONFIG = {
    'ticker': 'BTC-USD',
    'days back': 30, # This should match what the NN was trained on
    'model': 'nn_models/trend_detector'
}

if __name__ == '__main__':

    # Load in the NN classifier
    model = load_model(CONFIG['model'])
    
    # Get the dataframe to plot, and the processed data for the classifier
    df, data = nn_preprocess.process_for_classifier(CONFIG)
    
    # Make the predictions and append them to the dataframe
    predictions = model.predict(data)
    df.loc[:, 'classification'] = np.argmax(predictions, axis = 1)
    
    # Split into the different trends for individual plots
    no_trend = df[df['classification'] == 0]
    uptrend = df[df['classification'] == 1]
    downtrend = df[df['classification'] == 2]
    
    # Produce the plotly figure
    layout = go.Layout(
        title = CONFIG['ticker'] + ' Price',
        xaxis = {'title': 'Date'},
        yaxis = {'title': 'Price'},
    ) 
    
    fig = go.Figure(
        layout = layout,
        data=[
            go.Candlestick(
                x = df['Date'],
                open = df['Open'], 
                high = df['High'],
                low = df['Low'],
                close = df['Close'],
                name = 'Candlestick chart'
            ),
            go.Scatter(
                x = no_trend['Date'],
                y = no_trend['High'],
                name = 'No trend',
                mode = 'markers'
            ),
            go.Scatter(
                x = uptrend['Date'],
                y = uptrend['High'],
                name = 'Uptrend',
                mode = 'markers'
            ),
            go.Scatter(
                x = downtrend['Date'],
                y = downtrend['High'],
                name = 'Downtrend',
                mode = 'markers'
            ),
            
        ]
    )
    
    fig.update_xaxes(
            rangeslider_visible = False,
            rangebreaks=[{'bounds': ['sat', 'mon']}]
    )
    
    fig.show()
