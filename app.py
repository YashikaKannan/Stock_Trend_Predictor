import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DateFormatter
from sklearn.metrics import mean_squared_error
import datetime
from model_utils import prepare_data, build_lstm
from indicators import moving_average, calculate_rsi

# Page Config
st.set_page_config(page_title="Stock Trend Predictor", page_icon="📈", layout="wide")
st.markdown("""
    <style>
       footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)
st.sidebar.markdown("## 📘 Predictor Menu")

# Header
st.markdown("""
    <div style="text-align:center;">
        <h1 style="color:#00BFFF;">📈 Stock Price Trend Predictor</h1>
        <h4 style="color:gray;">Live Forecasting Dashboard Powered by LSTM</h4>
        <hr style="border-top: 2px solid #00BFFF;">
    </div>
""", unsafe_allow_html=True)

# What the app does
feature1, feature2, feature3 = st.columns(3)
with feature1:
    st.markdown("📅 **Fetch Live Data**\n\nFrom Yahoo Finance")
with feature2:
    st.markdown("🧠 **Predict Trends**\n\nUsing LSTM Deep Learning")
with feature3:
    st.markdown("📉 **Visualize Insights**\n\nWith MA, RSI, RMSE")

# Sidebar
st.sidebar.header("🔧 Settings")
symbols_input = st.sidebar.text_input("Enter Stock Symbols (comma separated)", value="AAPL,GOOGL")
symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
prediction_days = st.sidebar.slider("Days to Predict", min_value=5, max_value=30, value=7)


# Cache data
@st.cache_data
def get_data(symbol, start, end):
    return yf.download(symbol, start=start, end=end)

# Compare trends
st.subheader("📊 Compare Stock Trends")
fig_compare, ax_compare = plt.subplots(figsize=(10, 5))
for symbol in symbols:
    data = get_data(symbol, start_date, end_date)
    if not data.empty:
        data.index = pd.to_datetime(data.index)
        ax_compare.plot(data['Close'], label=symbol)
ax_compare.set_title("Close Price Comparison")
ax_compare.set_xlabel("Year/Month")
ax_compare.set_ylabel("Close Price")
ax_compare.legend()
st.pyplot(fig_compare)

# Main loop
for symbol in symbols:
    st.markdown(f"## 🔹 Stock: {symbol}")
    data = get_data(symbol, start_date, end_date)
    if data.empty:
        st.warning(f"No data found for {symbol}")
        continue

    st.subheader("🔢 Raw Data")
    st.dataframe(data.tail(10))

    # Add Indicators
    data = moving_average(data)
    data = calculate_rsi(data)

    # Indicator Plot
    st.subheader("📉 Closing Price with Moving Avg & RSI")
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(data['Close'], label="Close Price", color='blue')
    ax[0].plot(data['MA'], label="Moving Avg", color='orange')
    ax[0].legend()
    ax[0].set_ylabel("Price")
    ax[1].plot(data['RSI'], label="RSI", color='green')
    ax[1].axhline(70, color='red', linestyle='--')
    ax[1].axhline(30, color='red', linestyle='--')
    ax[1].set_ylabel("RSI")
    ax[1].set_xlabel("Year/Month")
    st.pyplot(fig)

    # LSTM Prediction
    st.subheader("🧠 LSTM Prediction")
    window = 60
    X, y, scaler = prepare_data(data, window)

    if len(X) < 10:
        st.warning("Not enough data to train model.")
        continue

    X_train, y_train = X[:-prediction_days], y[:-prediction_days]
    X_test = X[-prediction_days:]

    model = build_lstm((X.shape[1], 1))
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

    predicted_scaled = model.predict(X_test)
    predicted = scaler.inverse_transform(predicted_scaled)
    actual_scaled = y[-prediction_days:]
    actual = scaler.inverse_transform(actual_scaled.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # Generate only weekday future dates
    def next_working_days(start_date, num_days):
        days = []
        current = start_date
        while len(days) < num_days:
            current += datetime.timedelta(days=1)
            if current.weekday() < 5:  # 0-4 are weekdays
                days.append(current)
        return days

    # last_date = data.index[-1].to_pydatetime()
    # future_dates = next_working_days(last_date, prediction_days)

    # Get the last trading date from data
    last_date = data.index[-1]

    # Start checking from the next calendar day
    next_day = last_date + pd.Timedelta(days=1)

    # Collect only weekdays
    future_dates = []
    while len(future_dates) < prediction_days:
        if next_day.weekday() < 5:  # Monday=0 to Friday=4
            future_dates.append(next_day)
        next_day += pd.Timedelta(days=1)


   # 📈 Improved Prediction Plot
    st.subheader(f"📈 Prediction for next {prediction_days} days")
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    # Concatenate actual + future dates for clean continuity
    combined_dates = list(data.index[-15:]) + list(pd.to_datetime(future_dates))
    combined_prices = list(data['Close'].values[-15:]) + [None] * len(future_dates)

    # Plot actual prices
    ax2.plot(data.index[-15:], data['Close'].values[-15:], label="Actual Price", color='blue')
    ax2.set_xticks(data.index[-15:].append(pd.to_datetime(future_dates)))

    # Plot predicted prices starting exactly after actual
    ax2.plot(pd.to_datetime(future_dates), predicted, label="Predicted Price", color='red', linestyle='--')

    # Format X-axis to show both parts cleanly
    ax2.set_xlim([combined_dates[0], combined_dates[-1]])
    ax2.xaxis.set_major_locator(AutoDateLocator())
    ax2.xaxis.set_major_formatter(DateFormatter('%d-%m-%Y'))
    fig2.autofmt_xdate(rotation=45)

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Stock Price")
    ax2.legend()
    st.pyplot(fig2)

    # Key Stats
    st.subheader("📊 Key Statistics")
    stock_info = yf.Ticker(symbol).info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Open Price", f"${stock_info.get('open', 'N/A')}")
        st.metric("Prev Close", f"${stock_info.get('previousClose', 'N/A')}")
    with col2:
        st.metric("52W High", f"${stock_info.get('fiftyTwoWeekHigh', 'N/A')}")
        st.metric("52W Low", f"${stock_info.get('fiftyTwoWeekLow', 'N/A')}")
    with col3:
        st.metric("Volume", stock_info.get('volume', 'N/A'))
        st.metric("Market Cap", stock_info.get('marketCap', 'N/A'))

    # RMSE + Download
    st.success("✅ Prediction complete.")
    st.info(f"📉 RMSE: {rmse:.2f}")

    # CSV Download
    prediction_df = pd.DataFrame({
        "Date": [d.strftime("%d-%m-%Y") for d in future_dates],
        "Predicted Close Price": predicted.flatten()
    })
    csv = prediction_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📁 Download {symbol} Prediction CSV",
        data=csv,
        file_name=f'{symbol}_predicted_prices.csv',
        mime='text/csv'
    )

# Footer
st.markdown("""
    <hr>
    <div style="text-align:center;">
        <p style="color:gray;">Made with ❤️ by <b>Yashika</b> • <a href="https://linkedin.com/in/yashika-kannan" target="_blank">LinkedIn</a></p>
    </div>
""", unsafe_allow_html=True)
