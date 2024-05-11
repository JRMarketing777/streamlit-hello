import streamlit as st
import yfinance as yf
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

# Set up the page
st.title('Stock Price Prediction Dashboard')
st.sidebar.header('User Input Parameters')

def user_input_features():
    ticker = st.sidebar.text_input('Ticker', 'AAPL')
    return ticker

ticker = user_input_features()
data = yf.download(ticker, '2020-01-01', '2023-01-01')

st.line_chart(data.Close)

# Prepare data for ML model
data['SMA'] = data['Close'].rolling(window=14).mean()
data = data.dropna()
X = np.array(data.index).reshape(-1, 1)
y = data['SMA'].values

# Build and train the model
model = LinearRegression()
model.fit(X, y)

# Prediction and plot
prediction = model.predict(X)
st.write('Model Prediction of Moving Average Prices')
st.line_chart(data.index, prediction)

