import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np


# Load stock data
def load_data(ticker):
    data = yf.download(ticker, start="2020-01-01")
    data.reset_index(inplace=True)
    return data


# Forecasting model
def forecast(data, days=30):
    data['Days'] = np.arange(len(data))
    X = data[['Days']]
    y = data['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)

    future_days = np.arange(len(data), len(data) + days).reshape(-1, 1)
    predictions = model.predict(future_days)

    return future_days, predictions


app = dash.Dash(__name__)

# Load initial data
initial_ticker = 'AAPL'
df = load_data(initial_ticker)
future_days, predictions = forecast(df)

# List of stock tickers for the dropdown
stock_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

app.layout = html.Div([
    html.H1("Stock Price Visualization and Forecasting"),

    dcc.Dropdown(
        id='ticker-dropdown',
        options=[{'label': ticker, 'value': ticker} for ticker in stock_tickers],
        value=initial_ticker,
        clearable=False
    ),

    dcc.Graph(id='stock-graph'),

    dcc.Interval(
        id='interval-component',
        interval=60 * 60 * 1000,  # Refresh every hour
        n_intervals=0
    )
])


@app.callback(
    Output('stock-graph', 'figure'),
    [Input('ticker-dropdown', 'value')],
    [Input('interval-component', 'n_intervals')]
)
def update_graph(ticker, n_intervals):
    df = load_data(ticker)
    future_days, predictions = forecast(df)

    figure = {
        'data': [
            go.Scatter(
                x=df['Date'],
                y=df['Close'],
                mode='lines',
                name='Historical'
            ),
            go.Scatter(
                x=pd.date_range(start=df['Date'].iloc[-1], periods=len(predictions) + 1)[1:],
                y=predictions,
                mode='lines',
                name='Forecast'
            )
        ],
        'layout': go.Layout(
            title=f'Stock Prices for {ticker}',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Price'},
        )
    }

    return figure


if __name__ == '__main__':
    app.run_server(debug=True)
