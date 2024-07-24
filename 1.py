import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import math


# Load stock data
def load_data(ticker):
    data = yf.download(ticker, start="2020-01-01")
    data.reset_index(inplace=True)
    return data


# Preprocess data for LSTM
def preprocess_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler


# Forecasting model using LSTM
def forecast(data, days=30, look_back=60):
    X, y, scaler = preprocess_data(data, look_back)

    # Split the data into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Define the LSTM model
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)),
        LSTM(units=50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    # Calculate accuracy (using RMSE)
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_rmse = math.sqrt(mean_squared_error(y_train, train_predict))
    test_rmse = math.sqrt(mean_squared_error(y_test, test_predict))
    accuracy = 1 - (test_rmse / np.mean(scaler.inverse_transform(y_test.reshape(-1, 1))))

    # Predict future prices
    last_sequence = X[-1]
    future_predictions = []

    for _ in range(days):
        next_pred = model.predict(last_sequence.reshape(1, look_back, 1))[0]
        future_predictions.append(next_pred[0])
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = next_pred

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    future_days = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=days)

    return future_days, future_predictions, accuracy


app = dash.Dash(__name__)

# Load initial data
initial_ticker = 'AAPL'
df = load_data(initial_ticker)
future_days, predictions, accuracy = forecast(df)

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
    html.Div(id='accuracy-display'),
    dcc.Interval(
        id='interval-component',
        interval=60 * 60 * 1000,  # Refresh every hour
        n_intervals=0
    )
])


@app.callback(
    [Output('stock-graph', 'figure'),
     Output('accuracy-display', 'children')],
    [Input('ticker-dropdown', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_graph(ticker, n_intervals):
    df = load_data(ticker)
    future_days, predictions, accuracy = forecast(df)

    figure = {
        'data': [
            go.Scatter(
                x=df['Date'],
                y=df['Close'],
                mode='lines',
                name='Historical'
            ),
            go.Scatter(
                x=future_days,
                y=predictions.flatten(),
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

    accuracy_text = f"Model Accuracy: {accuracy:.2%}"

    return figure, accuracy_text


if __name__ == '__main__':
    app.run_server(debug=True)