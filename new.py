import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


# Load stock data
def load_data(ticker):
    data = yf.download(ticker, start="2020-01-01")
    data.reset_index(inplace=True)
    return data


# Feature engineering
def create_features(data):
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['RSI'] = calculate_rsi(data['Close'])
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['MA_200'] = data['Close'].rolling(window=200).mean()
    data['Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Return'].rolling(window=20).std() * np.sqrt(252)
    return data


# Calculate RSI
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# Forecasting model with accuracy prediction
def forecast_with_accuracy(data, days=30):
    data = create_features(data)
    data.dropna(inplace=True)

    features = ['EMA_12', 'EMA_26', 'MACD', 'RSI', 'MA_50', 'MA_200', 'Volatility']
    X = data[features]
    y = data['Close']

    tscv = TimeSeriesSplit(n_splits=5)
    mae_scores = []
    mse_scores = []
    accuracy_scores = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_predictions = rf_model.predict(X_test)

        # ARIMA model
        arima_model = ARIMA(y_train, order=(1, 1, 1))
        arima_results = arima_model.fit()
        arima_predictions = arima_results.forecast(steps=len(y_test))

        # Ensemble predictions
        predictions = (rf_predictions + arima_predictions) / 2

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        mae_scores.append(mae)
        mse_scores.append(mse)

        accuracy = np.mean(np.abs((y_test - predictions) / y_test) <= 0.05)
        accuracy_scores.append(accuracy)

    # Final predictions
    rf_model.fit(X, y)
    last_data = X.iloc[-1].values.reshape(1, -1)
    future_features = np.tile(last_data, (days, 1))
    rf_future = rf_model.predict(future_features)

    arima_model = ARIMA(y, order=(1, 1, 1))
    arima_results = arima_model.fit()
    arima_future = arima_results.forecast(steps=days)

    future_predictions = (rf_future + arima_future) / 2
    future_days = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=days)

    return future_days, future_predictions, np.mean(mae_scores), np.mean(mse_scores), np.mean(accuracy_scores)


# Initialize the Dash app
app = dash.Dash(__name__)

# Load initial data
initial_ticker = 'AAPL'
df = load_data(initial_ticker)
future_days, predictions, mae, mse, accuracy = forecast_with_accuracy(df)

# List of stock tickers for the dropdown
stock_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# App layout
app.layout = html.Div([
    html.H1("Stock Price Visualization and Forecasting"),
    dcc.Dropdown(
        id='ticker-dropdown',
        options=[{'label': ticker, 'value': ticker} for ticker in stock_tickers],
        value=initial_ticker,
        clearable=False
    ),
    dcc.Graph(id='stock-graph'),
    html.Div(id='accuracy-metrics'),
    dcc.Interval(
        id='interval-component',
        interval=60 * 60 * 1000,  # Refresh every hour
        n_intervals=0
    )
])


# Callback to update graph and metrics
@app.callback(
    [Output('stock-graph', 'figure'),
     Output('accuracy-metrics', 'children')],
    [Input('ticker-dropdown', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_graph(ticker, n_intervals):
    df = load_data(ticker)
    future_days, predictions, mae, mse, accuracy = forecast_with_accuracy(df)

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

    accuracy_metrics = html.Div([
        html.H3("Model Performance Metrics"),
        html.P(f"Mean Absolute Error: ${mae:.2f}"),
        html.P(f"Root Mean Squared Error: ${np.sqrt(mse):.2f}"),
        html.P(f"Accuracy (within 5% of actual): {accuracy * 100:.2f}%")
    ])

    return figure, accuracy_metrics


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)