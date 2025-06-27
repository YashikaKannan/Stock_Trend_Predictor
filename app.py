import streamlit as st
st.set_page_config( page_title="Stock Trend Predictor", page_icon="📈", layout="wide")
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        # header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from model_utils import prepare_data, build_lstm
from indicators import moving_average, calculate_rsi
from sklearn.metrics import mean_squared_error
import numpy as np
import datetime

st.markdown("""
    <div style="text-align:center;">
        <h1 style="color:#00BFFF; font-size:50px;">📈 Stock Price Trend Predictor</h1>
        <h4 style="color:gray;">Live Forecasting Dashboard Powered by LSTM</h4>
        <hr style="border-top: 2px solid #00BFFF; width:100%;">
    </div>
""", unsafe_allow_html=True)

st.markdown("### 🔍 What This System Does:")
feature1, feature2, feature3 = st.columns(3)
with feature1:
    st.markdown("📅 **Fetch Live Data**\n\nUsing Yahoo Finance")
with feature2:
    st.markdown("🧠 **Predict Trends**\n\nUsing LSTM Deep Learning")
with feature3:
    st.markdown("📉 **Visualize Insights**\n\nWith Moving Avg, RSI, RMSE")

# Title
st.title("Stock Price Trend Prediction using LSTM")
st.markdown("Predict future stock prices based on past trends using a trained LSTM model.")

# Sidebar Inputs
st.sidebar.header("🔧 Settings")
stocks_input = st.sidebar.text_input("Enter Stock Symbols (comma separated)", value="AAPL,GOOGL")
symbols = [s.strip().upper() for s in stocks_input.split(',') if s.strip()]
start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
prediction_days = st.sidebar.slider("Days to Predict", min_value=5, max_value=30, value=7)

# Fetch data

@st.cache_data
def get_data(symbol, start, end):
    return yf.download(symbol, start=start, end=end)

# Comparing datas

st.subheader("📊 Compare Stock Trends")

fig_compare, ax_compare = plt.subplots(figsize=(10, 5))

for symbol in symbols:
    compare_data = get_data(symbol, start_date, end_date)
    if compare_data.empty:
        continue
    compare_data.index = pd.to_datetime(compare_data.index)
    ax_compare.plot(compare_data['Close'], label=symbol)

ax_compare.set_title("Close Price Comparison")
ax_compare.set_xlabel("Date")
ax_compare.set_ylabel("Close Price")
ax_compare.legend()
st.pyplot(fig_compare)

# Loop through each stock symbol
for symbol in symbols:
    st.markdown(f"## 🔹 Stock: {symbol}")
    data = get_data(symbol, start_date, end_date)

    if data.empty:
        st.warning(f"No data found for {symbol}. Skipping. Please check the stock symbol or date range.")
        continue
    # Show raw data
    st.subheader(f"Raw Data for {symbol}")
    st.dataframe(data.tail(10))

    data.index = pd.to_datetime(data.index)

    # Add indicators
    data = moving_average(data)
    data = calculate_rsi(data)

    # Plot indicators
    st.subheader("Closing Price with Moving Average & RSI")
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    # Price + MA
    ax[0].plot(data['Close'], label="Close Price", color='blue')
    ax[0].plot(data['MA'], label="Moving Avg", color='orange')
    ax[0].legend()
    ax[0].set_ylabel("Price")
    # RSI
    ax[1].plot(data['RSI'], label="RSI", color='green')
    ax[1].axhline(70, color='red', linestyle='--')
    ax[1].axhline(30, color='red', linestyle='--')
    ax[1].set_ylabel("RSI")
    ax[1].set_xlabel("Date")

    st.pyplot(fig)

    # LSTM Model
    st.subheader("LSTM Model Prediction")
    window = 60
    X, y, scaler = prepare_data(data, window)

    if len(X) < 10:
        st.warning("Not enough data to train LSTM. Please increase date range.")
        continue

    X_train, y_train = X[:-prediction_days], y[:-prediction_days]
    X_test = X[-prediction_days:]

    model = build_lstm((X.shape[1], 1))
    model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)
    # Predict
    predicted_scaled = model.predict(X_test)
    predicted = scaler.inverse_transform(predicted_scaled)
    
    # Extract the last N actual values to compare (same as prediction length)
    actual_scaled = y[-prediction_days:]
    actual = scaler.inverse_transform(actual_scaled.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    # Forecast dates
    last_date = data.index[-1]
    future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(prediction_days)]
    # Create DataFrame for prediction results
    prediction_df = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
        "Predicted Close Price": predicted.flatten()
    })
    # Plot predictions
    st.subheader(f"📈 Prediction for next {prediction_days} days")
    fig2, ax2 = plt.subplots()
    # Plot recent actual prices
    ax2.plot(data.index[-100:], data['Close'].values[-100:], label="Actual Price", color='blue')

    # Plot future predicted prices

    ax2.plot(future_dates, predicted, label="Predicted Price", color='red')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig2.autofmt_xdate()
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Stock Price")
    ax2.legend()
    st.pyplot(fig2)

# Display Key Statistics

st.subheader(f"📊 Key Statistics for {symbol}")
stock_info = yf.Ticker(symbol).info

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Open Price", f"${stock_info.get('open', 'N/A')}")
    st.metric("Previous Close", f"${stock_info.get('previousClose', 'N/A')}")

with col2:
    st.metric("52-Week High", f"${stock_info.get('fiftyTwoWeekHigh', 'N/A')}")
    st.metric("52-Week Low", f"${stock_info.get('fiftyTwoWeekLow', 'N/A')}")

with col3:
    st.metric("Volume", stock_info.get('volume', 'N/A'))
    st.metric("Market Cap", stock_info.get('marketCap', 'N/A'))

# Summary

    st.success("Prediction complete.")
    st.info(f"📉 Model RMSE (Root Mean Squared Error): {rmse:.2f}")

# Allows the user to download predictions as CSV

    csv = prediction_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📅 Download Prediction CSV for {symbol}",
        data=csv,
        file_name=f'{symbol}_predicted_prices.csv',
        mime='text/csv'
    )

st.markdown("""
    <hr>
    <div style="text-align:center;">
        <p style="color:gray;">Made with ❤️ by <b>Yashika</b> • <a href="https://linkedin.com/in/yashika-kannan" target="_blank">LinkedIn</a></p>
    </div>
""", unsafe_allow_html=True)
