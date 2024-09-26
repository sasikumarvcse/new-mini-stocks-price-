
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
from gnews import GNews
from textblob import TextBlob


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

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)),
        LSTM(units=50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_rmse = math.sqrt(mean_squared_error(y_train, train_predict))
    test_rmse = math.sqrt(mean_squared_error(y_test, test_predict))
    accuracy = 1 - (test_rmse / np.mean(scaler.inverse_transform(y_test.reshape(-1, 1))))

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


# Get stock description
def get_stock_description(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        description = f"{info['longName']} ({ticker})\n\n"
        description += f"Sector: {info.get('sector', 'N/A')}\n"
        description += f"Industry: {info.get('industry', 'N/A')}\n"
        description += f"Current Price: ${info.get('currentPrice', 'N/A')}\n"
        description += f"52 Week High: ${info.get('fiftyTwoWeekHigh', 'N/A')}\n"
        description += f"52 Week Low: ${info.get('fiftyTwoWeekLow', 'N/A')}\n"
        description += f"\nBusiness Summary: {info.get('longBusinessSummary', 'N/A')}"

        return description
    except:
        return f"Unable to fetch description for {ticker}. Please try again later."


# Get latest news
def get_latest_news(ticker):
    google_news = GNews(language='en', country='US', period='7d', max_results=5)
    news = google_news.get_news(ticker)
    if news:
        return news[0]  # Return the latest news
    return None


# Analyze sentiment
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0.1:
        return "Positive", "The stock may increase based on this news."
    elif sentiment < -0.1:
        return "Negative", "The stock may decrease based on this news."
    else:
        return "Neutral", "The stock may not be significantly affected by this news."


app = dash.Dash(__name__)

# List of stock tickers for the dropdown
stock_tickers = [
    {'label': 'Apple Inc. (AAPL)', 'value': 'AAPL'},
    {'label': 'Microsoft Corporation (MSFT)', 'value': 'MSFT'},
    {'label': 'Alphabet Inc. (GOOGL)', 'value': 'GOOGL'},
    {'label': 'Amazon.com, Inc. (AMZN)', 'value': 'AMZN'},
    {'label': 'Tesla, Inc. (TSLA)', 'value': 'TSLA'},
    {'label': 'JPMorgan Chase & Co. (JPM)', 'value': 'JPM'},
    {'label': 'Johnson & Johnson (JNJ)', 'value': 'JNJ'},
    {'label': 'Visa Inc. (V)', 'value': 'V'},
    {'label': 'Procter & Gamble Company (PG)', 'value': 'PG'},
    {'label': 'Walmart Inc. (WMT)', 'value': 'WMT'},
]

initial_ticker = 'AAPL'

app.layout = html.Div([
    html.H1("Stock Price Visualization and Forecasting"),
    dcc.Dropdown(
        id='ticker-dropdown',
        options=stock_tickers,
        value=initial_ticker,
        clearable=False
    ),
    html.Div(id='stock-description', style={'whiteSpace': 'pre-wrap'}),
    dcc.Graph(id='stock-graph'),
    html.Div(id='accuracy-display'),
    html.H2("Latest News"),
    html.Div(id='news-content'),
    html.Div(id='sentiment-alert'),
    dcc.Graph(id='sentiment-graph'),
    dcc.Interval(
        id='interval-component',
        interval=60 * 60 * 1000,  # Refresh every hour
        n_intervals=0
    )
])


@app.callback(
    [Output('stock-graph', 'figure'),
     Output('accuracy-display', 'children'),
     Output('stock-description', 'children'),
     Output('news-content', 'children'),
     Output('sentiment-alert', 'children'),
     Output('sentiment-graph', 'figure')],
    [Input('ticker-dropdown', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_dashboard(ticker, n_intervals):
    df = load_data(ticker)
    future_days, predictions, accuracy = forecast(df)

    figure = {
        'data': [
            go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Historical'),
            go.Scatter(x=future_days, y=predictions.flatten(), mode='lines', name='Forecast')
        ],
        'layout': go.Layout(
            title=f'Stock Prices for {ticker}',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Price'},
        )
    }

    accuracy_text = f"Model Accuracy: {accuracy:.2%}"
    description = get_stock_description(ticker)

    news = get_latest_news(ticker)
    if news:
        news_content = html.Div([
            html.H3(news['title']),
            html.P(news['description']),
            html.A("Read more", href=news['url'], target="_blank")
        ])
        sentiment, alert_message = analyze_sentiment(news['title'] + " " + news['description'])
        sentiment_alert = html.Div(f"Sentiment: {sentiment}. {alert_message}",
                                   style={
                                       'color': 'green' if sentiment == 'Positive' else 'red' if sentiment == 'Negative' else 'black'})

        sentiment_figure = {
            'data': [
                go.Bar(x=['Sentiment'], y=[TextBlob(news['title'] + " " + news['description']).sentiment.polarity])],
            'layout': go.Layout(
                title='News Sentiment',
                yaxis={'title': 'Sentiment Score', 'range': [-1, 1]}
            )
        }
    else:
        news_content = html.P("No recent news available.")
        sentiment_alert = html.P("No sentiment analysis available.")
        sentiment_figure = {}

    return figure, accuracy_text, description, news_content, sentiment_alert, sentiment_figure


if __name__ == '__main__':
    app.run_server(debug=True)
