# Install necessary libraries
# Install necessary libraries
# Install necessary libraries
# Install necessary libraries
# Install necessary libraries
# Install necessary libraries
# Install necessary libraries
# Install necessary libraries
# Install necessary libraries
# Install necessary libraries
# Install necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Fetch stock data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="5y")
    df = df[['Close']]
    return df

# Prepare data for LSTM model
def prepare_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df)
    X, y = [], []
    for i in range(60, len(df_scaled)):
        X.append(df_scaled[i-60:i, 0])
        y.append(df_scaled[i, 0])
    return np.array(X), np.array(y), scaler

# Train LSTM model
def train_lstm_model(df):
    X, y, scaler = prepare_data(df)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    return model, scaler

# Predict next day's stock price
def predict_next_day_price(model, df, scaler):
    last_60_days = df[-60:].values.reshape(-1, 1)
    last_60_scaled = scaler.transform(last_60_days)
    X_test = np.reshape(last_60_scaled, (1, last_60_scaled.shape[0], 1))
    predicted_price = model.predict(X_test)
    return scaler.inverse_transform(predicted_price)[0, 0]

# Compare stock growth between user-defined months, including date and year
def compare_growth(df, start_month, start_year, end_month, end_year):
    start_data = df[(df.index.month == start_month) & (df.index.year == start_year)]
    end_data = df[(df.index.month == end_month) & (df.index.year == end_year)]
    if start_data.empty or end_data.empty:
        return None, None, None
    start_price = start_data.iloc[0]['Close']
    end_price = end_data.iloc[-1]['Close']
    growth = ((end_price - start_price) / start_price) * 100
    return start_price, end_price, growth

# Streamlit UI
def streamlit_ui():
    st.title(" 📈💰📊 Stock Analyzer 📈💰📊 ")
    tickers = st.text_input("Enter Stock Tickers (comma-separated, e.g., AAPL, TSLA):", "")
    ticker_list = [ticker.strip().upper() for ticker in tickers.split(',') if ticker.strip()]

    all_data = {}
    next_day_prices = {}
    for ticker in ticker_list:
        df = get_stock_data(ticker)
        if not df.empty:
            model, scaler = train_lstm_model(df)
            next_day_price = predict_next_day_price(model, df, scaler)
            all_data[ticker] = df
            next_day_prices[ticker] = f"${next_day_price:.2f}"

    if all_data:
        st.subheader("Stock Data & Prediction")
        fig = px.line()
        for ticker, df in all_data.items():
            fig.add_scatter(x=df.index, y=df['Close'], mode='lines', name=ticker)
        fig.update_layout(title="Stock Prices Comparison", xaxis_title="Date", yaxis_title="Stock Price ($)")
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig)

        st.subheader("Compare Stock Growth Between Months")
        start_month = st.number_input("Enter Start Month (1-12):", min_value=1, max_value=12, value=1)
        start_year = st.number_input("Enter Start Year:", min_value=2000, max_value=2025, value=2022)
        end_month = st.number_input("Enter End Month (1-12):", min_value=1, max_value=12, value=2)
        end_year = st.number_input("Enter End Year:", min_value=2000, max_value=2025, value=2025)

        growth_data = []
        for ticker, df in all_data.items():
            start_price, end_price, growth = compare_growth(df, start_month, start_year, end_month, end_year)
            if start_price and end_price:
                growth_data.append({"Ticker": ticker, "Start Price ($)": f"${start_price:.2f}", "End Price ($)": f"${end_price:.2f}", "Growth %": f"{growth:.2f}%"})
            else:
                st.error(f"Insufficient data for {ticker} in selected months and years.")

        if growth_data:
            growth_df = pd.DataFrame(growth_data)
            st.write(growth_df)
            st.bar_chart(growth_df.set_index("Ticker")['Growth %'])

        st.subheader("Predicted Next Day Price")
        prediction_df = pd.DataFrame({'Ticker': list(next_day_prices.keys()), 'Next Day Price ($)': list(next_day_prices.values())})
        st.write(prediction_df)
    else:
        st.error("Invalid Ticker(s) or No Data Available")

# Run Streamlit UI
if __name__ == "__main__":
    streamlit_ui()






