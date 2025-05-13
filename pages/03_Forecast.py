import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
from datetime import datetime
import calendar
from utils import load_data, filter_year_month, filter_year_month_depart, load_iata
from streamlit_card import card
import altair as alt
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os
import io

st.set_page_config(page_title="DataFrame Demo", page_icon="📊", layout="wide")

if 'df' not in st.session_state:
    st.session_state.df = load_data()
    st.session_state.df_code = load_iata()
    df = st.session_state.df
    df_code = st.session_state.df_code
else:
    df = st.session_state.df
    df_code = st.session_state.df_code
    
df_holiday = pd.read_excel('files/holidays_ID.xlsx')
    
df['Issued Date'] = pd.to_datetime(df['Issued Date'])
df['Segments/Departure Date'] = pd.to_datetime(df['Segments/Departure Date'])
start_year = df['Issued Date'].min().year
end_year = df['Issued Date'].max().year

df['Booking Date'] = pd.to_datetime(df['Booking Date'])
df['Issued Date'] = pd.to_datetime(df['Issued Date'])
df['Segments/Departure Date'] = pd.to_datetime(df['Segments/Departure Date'])

booking_pax = df.groupby(df['Booking Date'].dt.date)['Total Pax'].sum().reset_index()
issued_pax = df.groupby(df['Segments/Departure Date'].dt.date)['Total Pax'].sum().reset_index()
departure_pax = df.groupby(df['Segments/Departure Date'].dt.date)['Total Pax'].sum().reset_index()

df_holiday = pd.read_excel('files/holidays_ID.xlsx')
df_holiday['Date'] = pd.to_datetime(df_holiday['Date'])
df_holiday = df_holiday[['Date', 'Libur']].rename(columns={'Date': 'Segments/Departure Date', 'Libur': 'holiday'})

df['Segments/Departure Date'] = pd.to_datetime(df['Segments/Departure Date'])
# df['Total Pax'] = np.log1p(df['Total Pax'])
df_daily = df.groupby(df['Segments/Departure Date'].dt.date)['Total Pax'].sum().reset_index()
df_daily['Segments/Departure Date'] = pd.to_datetime(df_daily['Segments/Departure Date'])

df_daily = df_daily.merge(df_holiday, on='Segments/Departure Date', how='left')
df_daily['holiday'] = df_daily['holiday'].fillna(0).astype(int)

df_daily = df_daily[
    (df_daily['Segments/Departure Date'].dt.year >= start_year) &
    (df_daily['Segments/Departure Date'].dt.year <= end_year)
]

# st.dataframe(df_daily)

n = int(len(df_daily) * 0.8)
train = df_daily.iloc[:n]
test = df_daily.iloc[n:]

endog_train = train['Total Pax']
exog_train = train[['holiday']]

endog_test = test['Total Pax']
exog_test = test[['holiday']]

train_size = int(len(df_daily)*0.8)
train_data = df_daily['Total Pax'].iloc[:train_size].values.reshape(-1, 1)
test_data = df_daily['Total Pax'].iloc[train_size:].values.reshape(-1, 1)

scaler = MinMaxScaler().fit(train_data)
train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)

def create_dataset(X, look_back=1):
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        v = X[i:i+look_back]
        Xs.append(v)
        ys.append(X[i+look_back])
    return np.array(Xs), np.array(ys)

LOOK_BACK = 1
X_train, y_train = create_dataset(train_scaled, LOOK_BACK)
X_test, y_test = create_dataset(test_scaled, LOOK_BACK)

print('X_train.shape:', X_train.shape)
print('y_train.shape:', y_train.shape)
print('X_test.shape:', X_test.shape)

def create_gru(units):
    model = Sequential()
    model.add(GRU(units=units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(GRU(units=units))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')
    return model

MODEL_PATH = 'gruPredictDepart.keras'

if os.path.exists(MODEL_PATH):
    model_gru = load_model(MODEL_PATH)
    # model_gru.fit(X_train, y_train, epochs=1500, validation_split=0.2, batch_size=16, shuffle=False)
    # model_gru.save('gruPredictDepart.keras')
    
else:
    model_gru = create_gru(64)
    model_gru.fit(X_train, y_train, epochs=250, validation_split=0.2, batch_size=16, shuffle=False)
    model_gru.save('gruPredictDepart.keras')

y_test = scaler.inverse_transform(y_test)
y_train = scaler.inverse_transform(y_train)

def prediction(model):
    prediction = model.predict(X_test)
    prediction = scaler.inverse_transform(prediction)
    return prediction

prediction_gru = prediction(model_gru)

def plot_future(prediction, model_name, y_test):
    prediction = prediction.flatten()
    y_test = y_test.flatten()[-len(prediction):]

    data = pd.DataFrame({
        'Test Data': y_test,
        'Prediction': prediction
    })

    st.line_chart(data)

plot_future(prediction_gru, 'GRU', y_test)

def evaluate_prediction(predictions, actual, model_name):
    errors = predictions - actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()
    mape = mean_absolute_percentage_error(actual, predictions)

    print(model_name + ':')
    print('Mean Absolute Error: {:.4f}'.format(mae))
    print('Root Mean Square Error: {:.4f}'.format(rmse))
    print('Mean Absolute Percentage Error: {:.2f}%'.format(mape * 100))
    print('')

evaluate_prediction(prediction_gru, y_test, 'GRU')

def forecast_next_days(model, last_sequence, steps, scaler):
    forecast = []
    current_input = last_sequence.copy()

    for _ in range(steps):
        pred = model.predict(current_input.reshape(1, LOOK_BACK, 1))
        forecast.append(pred[0, 0])
        current_input = np.append(current_input[1:], pred, axis=0)

    forecast = np.array(forecast).reshape(-1, 1)
    forecast_inv = scaler.inverse_transform(forecast)
    return forecast_inv

full_scaled = scaler.transform(df_daily['Total Pax'].values.reshape(-1, 1))
last_sequence = full_scaled[-LOOK_BACK:]

future_pred = forecast_next_days(model_gru, last_sequence, steps=30, scaler=scaler)

last_date = df_daily.index[-1] if isinstance(df_daily.index, pd.DatetimeIndex) else pd.to_datetime(df_daily['Segments/Departure Date'].iloc[-1])
future_dates = pd.date_range(start=last_date, periods=31, freq='D')[1:]

future_df = pd.DataFrame({
    'Date': future_dates,
    'Forecasted Total Pax': future_pred.flatten()
})

st.line_chart(future_df.set_index('Date'))

class sarimax:
    def seasonality_test(data, column, period):
        decomposition_result = None

        try:
            decomposition = seasonal_decompose(data[column], model='multiplicative', period=period)
            seasonal_component = decomposition.seasonal
            decomposition_result = "Seasonal"
        except Exception:
            decomposition_result = "Decomposition failed"

        try:
            ljung_box_result = acorr_ljungbox(data[column], lags=[period], return_df=True)
            p_value = ljung_box_result['lb_pvalue'].values[0]
            lb_result = "Seasonal" if p_value < 0.05 else "Not Seasonal"
        except Exception:
            lb_result = "Ljung-Box failed"

        if decomposition_result == "Seasonal" or lb_result == "Seasonal":
            return "Seasonal"
        elif decomposition_result == "Decomposition failed" and lb_result == "Ljung-Box failed":
            return "Both Tests Failed"
        else:
            return "Not Seasonal"
        
    # Test seasonality
    bool_seasonal = seasonality_test(df_daily, column='Total Pax', period=1)
    st.write(f"Seasonality test result: {bool_seasonal}")

    # Grid Search for SARIMAX Parameters
    p = d = P = D = range(0, 3)
    q = Q = range(0, 2)
    s = [7]

    param_grid = list(itertools.product(p, d, q))
    seasonal_grid = list(itertools.product(P, D, Q, s)) if bool_seasonal == "Seasonal" else None

    best_mse = float("inf")
    best_mae = float("inf")
    best_r2 = float("inf")
    best_order = None
    best_seasonal_order = None
    best_model = None

    # try:
    #     model_fit = joblib.load('sarimax_model.pkl')
    #     st.write("Loaded pretrained model.")
        
    #     # Perform prediction with the loaded model
    #     predictions = model_fit.predict(start=0, end=len(endog_test) - 1, exog=exog_test)

    #     # Calculate the error metrics (MAE, MSE, R2)
    #     mse = mean_squared_error(endog_test, predictions)
    #     mae = mean_absolute_error(endog_test, predictions)
    #     r2 = r2_score(endog_test, predictions)

    #     # Display the error metrics
    #     st.write(f"MAE: {mae}")
    #     st.write(f"MSE: {mse}")
    #     st.write(f"R2: {r2}")
        
    try:
        # Coba untuk membaca file CSV yang sudah ada
        params_df = pd.read_csv('best_sarimax_params.csv')

        order = (params_df['order_p'][0], params_df['order_d'][0], params_df['order_q'][0])
        seasonal_order = (
            params_df['seasonal_order_P'][0],
            params_df['seasonal_order_D'][0],
            params_df['seasonal_order_Q'][0],
            params_df['seasonal_order_s'][0]
        )

        model = SARIMAX(endog_train, order=order, seasonal_order=seasonal_order, exog=exog_train)
        model_fit = model.fit(disp=False)

        st.write(f"Loaded SARIMAX parameters from CSV")
        st.write(f"Order: {order}")
        st.write(f"Seasonal Order: {seasonal_order}")

    except FileNotFoundError:
        st.write("No parameter CSV found, training from scratch.")

        for param in param_grid:
            for param_seasonal in seasonal_grid:
                try:
                    model = SARIMAX(endog_train, order=param, seasonal_order=param_seasonal, exog=exog_train)
                    model_fit = model.fit(disp=False)

                    predictions = model_fit.predict(start=0, end=len(endog_train) - 1)
                    mse = mean_squared_error(endog_train, predictions)
                    mae = mean_absolute_error(endog_train, predictions)
                    r2 = r2_score(endog_train, predictions)

                    if mse < best_mse:
                        best_mse = mse
                        best_mae = mae
                        best_r2 = r2
                        best_order = param
                        best_seasonal_order = param_seasonal
                        best_model = model_fit

                except Exception:
                    continue

        best_model_info = pd.DataFrame([{
            'order_p': best_order[0],
            'order_d': best_order[1],
            'order_q': best_order[2],
            'seasonal_order_P': best_seasonal_order[0],
            'seasonal_order_D': best_seasonal_order[1],
            'seasonal_order_Q': best_seasonal_order[2],
            'seasonal_order_s': best_seasonal_order[3],
            'MSE': best_mse,
            'MAE': best_mae,
            'R2': best_r2
        }])

        best_model_info.to_csv('best_sarimax_params.csv', index=False)
        st.write("Best model parameters saved to CSV.")
        st.dataframe(best_model_info)

        # Auto-download CSV
        csv_buffer = io.StringIO()
        best_model_info.to_csv(csv_buffer, index=False)
        
        # Here we simulate an automatic download by writing the CSV data directly
        st.download_button(
            label="Download SARIMAX Parameters CSV",
            data=csv_buffer.getvalue(),
            file_name='best_sarimax_params.csv',
            mime='text/csv'
        )

    # Perform prediction with the best model
    predictions = model_fit.predict(start=0, end=len(endog_test) - 1, exog=exog_test)