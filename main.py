from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import ta
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------- LSTM Helper -----------
def prepare_lstm_data(df):
    df = df[['Close']].values
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    
    X, y = [], []
    for i in range(60, len(df_scaled)):
        X.append(df_scaled[i-60:i])
        y.append(df_scaled[i])
    
    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ----------- News Sentiment Scraper -----------
def get_sentiment(stock_name):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = f"https://www.moneycontrol.com/news/tags/{stock_name.lower()}.html"
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, 'html.parser')
        headlines = [x.text for x in soup.find_all('h2')[:5]]
        
        polarity = 0
        for headline in headlines:
            blob = TextBlob(headline)
            polarity += blob.sentiment.polarity
        avg_sentiment = polarity / len(headlines) if headlines else 0
        return round(avg_sentiment, 2)
    except Exception:
        return 0  # Neutral if fetch fails

# ----------- Main Prediction Endpoint -----------
@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    symbol = data["stock"].upper() + ".NS"
    
    df = yf.download(symbol, period="90d", interval="15m")
    df.dropna(inplace=True)

    # Add Technical Indicators
    df['EMA'] = ta.trend.ema_indicator(df['Close'], window=14)
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['MACD'] = ta.trend.macd_diff(df['Close'])

    df.dropna(inplace=True)

    # Prepare data for LSTM
    X, y, scaler = prepare_lstm_data(df)

    model = build_lstm_model((X.shape[1], 1))
    model.fit(X, y, epochs=3, batch_size=32, verbose=0)

    # Predict next step
    last_60 = scaler.transform(df[['Close']].values)[-60:]
    last_60 = np.expand_dims(last_60, axis=0)
    predicted_price = scaler.inverse_transform(model.predict(last_60))[0][0]

    # Get news sentiment
    sentiment_score = get_sentiment(data["stock"])

    return {
        "stock": symbol,
        "current_price": round(df['Close'].values[-1], 2),
        "predicted_price": round(predicted_price, 2),
        "EMA": round(df['EMA'].values[-1], 2),
        "RSI": round(df['RSI'].values[-1], 2),
        "MACD": round(df['MACD'].values[-1], 2),
        "sentiment": sentiment_score,
        "direction": "UP" if predicted_price > df['Close'].values[-1] else "DOWN"
    }
