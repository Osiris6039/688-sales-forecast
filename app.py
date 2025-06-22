import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import base64
import os

# --- Authentication ---
def login():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if username == "admin" and password == "admin123":
        return True
    elif username and password:
        st.sidebar.error("Invalid credentials")
    return False

# --- Forecasting Function ---
def train_and_forecast(df, target_column):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    df = pd.get_dummies(df, columns=['weather'])

    features = [col for col in df.columns if col not in ['date', 'sales', 'customers']]
    X = df[features]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    
    # Forecasting for next 7 days
    last_date = df['date'].max()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'addons': df['addons'].mean(),
        'dayofweek': [d.dayofweek for d in future_dates],
        'month': [d.month for d in future_dates],
        'day': [d.day for d in future_dates],
    })
    # Reapply weather dummy columns
    for col in df.columns:
        if col.startswith('weather_'):
            forecast_df[col] = 0
    forecast = model.predict(forecast_df.drop(columns=['date']))
    return future_dates, forecast

# --- App ---
if login():
    st.title("Smart Sales & Customer Forecaster")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data")
        st.dataframe(df.head())

        # Forecasting Section
        st.subheader("Forecasting")
        sales_dates, sales_forecast = train_and_forecast(df, "sales")
        cust_dates, cust_forecast = train_and_forecast(df, "customers")

        forecast_df = pd.DataFrame({
            "date": sales_dates,
            "forecasted_sales": sales_forecast.astype(int),
            "forecasted_customers": cust_forecast.astype(int)
        })

        st.line_chart(forecast_df.set_index("date"))

        st.dataframe(forecast_df)

        # Download Button
        csv = forecast_df.to_csv(index=False).encode()
        st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv")

        # Manual data input
        st.subheader("Manual Data Input")
        with st.form("data_input"):
            input_date = st.date_input("Date")
            input_sales = st.number_input("Sales", value=0)
            input_customers = st.number_input("Customers", value=0)
            input_weather = st.selectbox("Weather", ["sunny", "rainy", "cloudy"])
            input_addons = st.number_input("Add-on Sales", value=0)
            submitted = st.form_submit_button("Save Entry")

            if submitted:
                new_row = pd.DataFrame([[input_date, input_sales, input_customers, input_weather, input_addons]],
                                       columns=['date', 'sales', 'customers', 'weather', 'addons'])
                if os.path.exists("historical_data.csv"):
                    historical_df = pd.read_csv("historical_data.csv")
                    historical_df = pd.concat([historical_df, new_row], ignore_index=True)
                else:
                    historical_df = new_row
                historical_df.to_csv("historical_data.csv", index=False)
                st.success("Data saved successfully!")