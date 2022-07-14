
def get_candlestick_chart(df)
    layout = go.Layout(
        title = f'{ticker} Candlestick Chart',
        xaxis = {'title':'Date'},
        yaxis = {'title': 'Price'},
        # yaxis=dict(
        #     title="Price"
        # ) 
    )
    
    fig = go.Figure(
        layout=layout,
        data=[
            go.Candlestick(
                x=df['Date'],
                open=df['Open'], 
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name = 'Candlestick chart'
            ),
        ]
    )
    
    
    fig.update_xaxes(
            rangeslider_visible=False,
            rangebreaks=[{'bounds': ['sat', 'mon']}]
    )