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
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import Sequential, layers, callbacks
# from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
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

model_option = st.sidebar.selectbox(
    'Choose Prediction Model:',
    ('GRU', 'SARIMAX')
)

df_holiday = pd.read_excel('files/holidays_ID.xlsx')
df['Issued Date'] = pd.to_datetime(df['Issued Date'])
df['Segments/Departure Date'] = pd.to_datetime(df['Segments/Departure Date'])
start_year = df['Issued Date'].min().year
end_year = df['Issued Date'].max().year

df['Booking Date'] = pd.to_datetime(df['Booking Date'])
df_daily = df.groupby(
    df['Segments/Departure Date'].dt.date)['Total Pax'].sum().reset_index()
df_daily['Segments/Departure Date'] = pd.to_datetime(
    df_daily['Segments/Departure Date'])

df_holiday['Date'] = pd.to_datetime(df_holiday['Date'])
df_holiday = df_holiday[['Date', 'Libur']].rename(
    columns={'Date': 'Segments/Departure Date', 'Libur': 'holiday'})
df_daily = df_daily.merge(df_holiday, on='Segments/Departure Date', how='left')
df_daily['holiday'] = df_daily['holiday'].fillna(0).astype(int)

# today = pd.Timestamp('today').normalize()

# df_daily = df_daily[
#     (df_daily['Segments/Departure Date'].dt.year >= start_year) &
#     (df_daily['Segments/Departure Date'].dt.year <= today)
# ]

df_daily = df_daily[
    (df_daily['Segments/Departure Date'].dt.year >= start_year) &
    (df_daily['Segments/Departure Date'].dt.year <= end_year)
]

if model_option == 'GRU':
    lookback_option = st.sidebar.selectbox(
        'Choose Time Window (in Days):',
        (3, 7, 30)
    )

    st.subheader(
        f"GRU Forecast - Actual Vs Predicted ({lookback_option} Day{'s)' if lookback_option > 1 else ')'}")

    n = int(len(df_daily) * 0.8)
    train_data = df_daily['Total Pax'].iloc[:n].values.reshape(-1, 1)
    test_data = df_daily['Total Pax'].iloc[n:].values.reshape(-1, 1)

    scaler = MinMaxScaler().fit(train_data)
    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)

    def create_dataset(X, look_back):
        Xs, ys = [], []
        for i in range(len(X) - look_back):
            Xs.append(X[i:i+look_back])
            ys.append(X[i+look_back])
        return np.array(Xs), np.array(ys)

    LOOK_BACK = lookback_option
    X_train, y_train = create_dataset(train_scaled, LOOK_BACK)
    X_test, y_test = create_dataset(test_scaled, LOOK_BACK)

    def create_gru(units):
        model = Sequential()
        model.add(GRU(units=units, return_sequences=True,
                  input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(GRU(units=units))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mse')
        return model

    model_name = f'gruPredictDepart_{LOOK_BACK}.keras'

    if os.path.exists(model_name):
        model_gru = load_model(model_name)
        # model_gru.fit(X_train, y_train, epochs=250,
        #               validation_split=0.2, batch_size=16, shuffle=False)
        # model_gru.save(model_name)
    else:
        model_gru = create_gru(64)
        model_gru.fit(X_train, y_train, epochs=250,
                      validation_split=0.2, batch_size=16, shuffle=False)
        model_gru.save(model_name)

    y_test_inv = scaler.inverse_transform(y_test)
    prediction_gru = scaler.inverse_transform(model_gru.predict(X_test))

    st.line_chart(pd.DataFrame(
        {'Actual': y_test_inv.flatten(), 'Prediction': prediction_gru.flatten()}))

    errors = prediction_gru - y_test_inv
    mae = np.abs(errors).mean()
    rmse = np.sqrt(np.square(errors).mean())
    mape = mean_absolute_percentage_error(y_test_inv, prediction_gru)

    def forecast_next_days(model, last_sequence, steps, scaler):
        forecast = []
        current_input = last_sequence.copy()
        for _ in range(steps):
            pred = model.predict(current_input.reshape(1, LOOK_BACK, 1))
            forecast.append(pred[0, 0])
            current_input = np.append(current_input[1:], pred, axis=0)
        forecast = np.array(forecast).reshape(-1, 1)
        return scaler.inverse_transform(forecast)

    last_sequence = scaler.transform(
        df_daily['Total Pax'].values.reshape(-1, 1))[-LOOK_BACK:]
    # Forecast dengan langkah sama seperti LOOK_BACK
    future_pred = forecast_next_days(
        model_gru, last_sequence, steps=LOOK_BACK, scaler=scaler)

    # Generate tanggal untuk forecast dengan jumlah sama seperti LOOK_BACK
    future_dates = pd.date_range(
        start=df_daily['Segments/Departure Date'].max(), periods=LOOK_BACK + 1, freq='D')[1:]

    st.subheader(
        f"GRU Forecast - Next {lookback_option} Day{'s' if lookback_option > 1 else ''}")

    st.line_chart(pd.DataFrame(
        {f'Forecasted Pax ({LOOK_BACK} Days Time Span)': future_pred.flatten()}, index=future_dates))

    r2 = r2_score(y_test_inv, prediction_gru)

    st.markdown(f"""
    ### GRU Evaluation (Time Window {LOOK_BACK} days)

    - **MAE:** {mae:.2f}  
    - **RMSE:** {rmse:.2f}  
    - **MAPE:** {mape * 100:.2f}%  
    - **R² (R-squared):** {r2:.4f}  
    """)

    st.markdown("""
    ---
    ### Glossary of Metrics
    
    - **GRU (Gated Recurrent Unit):** A special kind of neural network model that learns patterns over time. It’s good at understanding sequences like daily data and is used for tasks like forecasting.

    - **Time Window:** The number of previous time steps (this case: days) used as input features to predict the next value in the series. For example, a lookback of 7 means the model uses the past 7 days to forecast the next day.

    - **MAE (Mean Absolute Error):** The average of the absolute differences between predicted and actual values; Lower values indicate better accuracy.

    - **RMSE (Root Mean Squared Error):** The square root of the average squared differences between predicted and actual values; Lower values indicate better accuracy.

    - **MAPE (Mean Absolute Percentage Error):** The average absolute percentage error between predicted and actual values; Higher values indicate better accuracy.

    - **R² (R-squared):** The proportion of variance in the data explained by the model. Maximum value is 1 (perfect fit); Negative values mean the model performs worse than simply predicting the mean.
    """)


if model_option == 'SARIMAX':
    def sarimax_grid_search(endog_train, exog_train,
                            p_values, d_values, q_values,
                            P_values, D_values, Q_values, s_values,
                            scoring='mse'):
        best_score = np.inf
        best_cfg = None
        best_model_fit = None

        seasonal_params = list(itertools.product(P_values, D_values, Q_values, s_values))
        params = list(itertools.product(p_values, d_values, q_values))

        for order in params:
            for seasonal_order in seasonal_params:
                try:
                    model = SARIMAX(endog_train,
                                    order=order,
                                    seasonal_order=seasonal_order,
                                    exog=exog_train,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
                    model_fit = model.fit(disp=False)
                    pred = model_fit.predict(start=0, end=len(endog_train)-1, exog=exog_train)
                    if scoring == 'mse':
                        score = mean_squared_error(endog_train, pred)
                    elif scoring == 'aic':
                        score = model_fit.aic
                    else:
                        raise ValueError("Unsupported scoring method")

                    if score < best_score:
                        best_score = score
                        best_cfg = (order, seasonal_order)
                        best_model_fit = model_fit
                except Exception:
                    continue

        return best_cfg, best_model_fit, best_score

    st.title("SARIMAX Grid Search Forecasting")

    # Load data hari libur
    df_holiday = pd.read_excel('files/holidays_ID.xlsx')
    df_holiday['Segments/Departure Date'] = pd.to_datetime(df_holiday['Date'])
    df_holiday['holiday'] = df_holiday['Holiday Name'].notna().astype(int)
    df_holiday = df_holiday[['Segments/Departure Date', 'holiday']]

    # Persiapan data harian
    df['Segments/Departure Date'] = pd.to_datetime(df['Segments/Departure Date']).dt.normalize()
    df_daily = df.groupby('Segments/Departure Date')['Total Pax'].sum().reset_index()
    df_daily = df_daily.merge(df_holiday, on='Segments/Departure Date', how='left')
    df_daily['holiday'] = df_daily['holiday'].fillna(0).astype(int)
    df_daily = df_daily[df_daily['Total Pax'] != 0]

    # Filter tahun 2022–2024
    df_daily = df_daily[
        (df_daily["Segments/Departure Date"].dt.year >= 2022) &
        (df_daily["Segments/Departure Date"].dt.year <= 2024)
    ]

    # Index ulang untuk resampling
    df_daily.set_index('Segments/Departure Date', inplace=True)

    # Data mingguan (resample per minggu, minggu berakhir Minggu)
    df_7daily = df_daily.resample('W-SUN').agg({
        'Total Pax': 'sum',
        'holiday': 'sum'
    }).reset_index()
    df_7daily['Segments/Departure Date'] = df_7daily['Segments/Departure Date'] - pd.to_timedelta(6, unit='d')
    df_7daily = df_7daily[df_7daily['Total Pax'] != 0]

    # Filter 2022–2024
    df_7daily = df_7daily[
        (df_7daily["Segments/Departure Date"].dt.year >= 2022) &
        (df_7daily["Segments/Departure Date"].dt.year <= 2024)
    ]

    # Data bulanan (resample dari awal bulan)
    df_30daily = df_daily.resample('MS').agg({
        'Total Pax': 'sum',
        'holiday': 'sum'
    }).reset_index()
    df_30daily = df_30daily[df_30daily['Total Pax'] != 0]

    # Filter 2022–2024
    df_30daily = df_30daily[
        (df_30daily["Segments/Departure Date"].dt.year >= 2022) &
        (df_30daily["Segments/Departure Date"].dt.year <= 2024)
    ]

    # st.write(df_daily)
    # st.write(df_7daily)
    # st.write(df_30daily)
    
    train_ratio = 1

    # Harian
    n_train_daily = int(len(df_daily) * train_ratio)
    endog_train_daily = df_daily['Total Pax'].iloc[:n_train_daily]
    endog_test_daily = df_daily['Total Pax'].iloc[n_train_daily:]
    exog_train_daily = df_daily[['holiday']].iloc[:n_train_daily]
    exog_test_daily = df_daily[['holiday']].iloc[n_train_daily:]
    
    # 7 Harian
    n_train_7d = int(len(df_7daily) * train_ratio)
    endog_train_7d = df_7daily['Total Pax'].iloc[:n_train_7d]
    endog_test_7d = df_7daily['Total Pax'].iloc[n_train_7d:]
    exog_train_7d = df_7daily[['holiday']].iloc[:n_train_7d]
    exog_test_7d = df_7daily[['holiday']].iloc[n_train_7d:]

    # 30 Harian
    n_train_30d = int(len(df_30daily) * train_ratio)
    endog_train_30d = df_30daily['Total Pax'].iloc[:n_train_30d]
    endog_test_30d = df_30daily['Total Pax'].iloc[n_train_30d:]
    exog_train_30d = df_30daily[['holiday']].iloc[:n_train_30d]
    exog_test_30d = df_30daily[['holiday']].iloc[n_train_30d:]
    
    # Plot ACF dan PACF
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import adfuller

    # st.subheader("ACF and PACF Plots")

    # # daily
    # fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
    # plot_acf(endog_train_daily, ax=ax_acf, lags=30)
    # st.pyplot(fig_acf)

    # fig_pacf, ax_pacf = plt.subplots(figsize=(10, 4))
    # plot_pacf(endog_train_daily, ax=ax_pacf, lags=30, method='ywm')
    # st.pyplot(fig_pacf)

    # result = adfuller(endog_train_daily)    
    # st.write('ADF Statistic:', result[0])
    # st.write('p-value:', result[1])
    
    # # week
    # fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
    # plot_acf(endog_train_7d, ax=ax_acf, lags=12)
    # st.pyplot(fig_acf)

    # fig_pacf, ax_pacf = plt.subplots(figsize=(10, 4))
    # plot_pacf(endog_train_7d, ax=ax_pacf, lags=12, method='ywm')
    # st.pyplot(fig_pacf)

    # result = adfuller(endog_train_7d)    
    # st.write('ADF Statistic:', result[0])
    # st.write('p-value:', result[1])
    
    # # month
    # fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
    # plot_acf(endog_train_30d, ax=ax_acf, lags=12)
    # st.pyplot(fig_acf)

    # fig_pacf, ax_pacf = plt.subplots(figsize=(10, 4))
    # plot_pacf(endog_train_30d, ax=ax_pacf, lags=12, method='ywm')
    # st.pyplot(fig_pacf)

    # from statsmodels.tsa.stattools import adfuller
    # result = adfuller(endog_train_30d)    
    # st.write('ADF Statistic:', result[0])
    # st.write('p-value:', result[1])
    
    # Daily
    # # Grid params
    p = P = range(0, 4)
    d = D = range(0, 1)
    q = Q = range(0, 4)
    s = [3, 7, 14]

    best_cfg, best_model, best_score = sarimax_grid_search(
        endog_train_daily, exog_train_daily,
        p, d, q,
        P, D, Q, s,
        scoring='mse'
    )

    st.write(f"Best SARIMAX order: {best_cfg[0]}, seasonal_order: {best_cfg[1]}, MSE: {best_score:.2f}")

    params_df = pd.DataFrame({
        'order_p': [best_cfg[0][0]],
        'order_d': [best_cfg[0][1]],
        'order_q': [best_cfg[0][2]],
        'seasonal_order_P': [best_cfg[1][0]],
        'seasonal_order_D': [best_cfg[1][1]],
        'seasonal_order_Q': [best_cfg[1][2]],
        'seasonal_order_s': [best_cfg[1][3]],
        'MSE': [best_score]
    })

    predictions = best_model.predict(start=0, end=len(df_daily)-1, exog=exog_train_daily)

    df_plot = pd.DataFrame({'Actual': endog_train_daily, 'Predicted': predictions})
    st.line_chart(df_plot)

    mae = mean_absolute_error(endog_train_daily, predictions)
    mse = mean_squared_error(endog_train_daily, predictions)
    r2 = r2_score(endog_train_daily, predictions)
    mape = (abs((endog_train_daily - predictions) / endog_train_daily)).mean() * 100

    st.write(f"MAE: {mae:.2f}")
    st.write(f"MSE: {mse:.2f}")
    st.write(f"R2 Score: {r2:.2f}")
    st.write(f"MAPE: {mape:.2f}%")
    
    # # Weekly
    # # Grid params
    # p = P = range(0, 2)
    # d = D = range(0, 1)
    # q = Q = range(0, 2)
    # s = [4, 13, 26, 52]

    # best_cfg, best_model, best_score = sarimax_grid_search(
    #     endog_train_7d, exog_train_7d,
    #     p, d, q,
    #     P, D, Q, s,
    #     scoring='mse'
    # )

    # st.write(f"Best SARIMAX order: {best_cfg[0]}, seasonal_order: {best_cfg[1]}, MSE: {best_score:.2f}")

    # params_df = pd.DataFrame({
    #     'order_p': [best_cfg[0][0]],
    #     'order_d': [best_cfg[0][1]],
    #     'order_q': [best_cfg[0][2]],
    #     'seasonal_order_P': [best_cfg[1][0]],
    #     'seasonal_order_D': [best_cfg[1][1]],
    #     'seasonal_order_Q': [best_cfg[1][2]],
    #     'seasonal_order_s': [best_cfg[1][3]],
    #     'MSE': [best_score]
    # })

    # predictions = best_model.predict(start=0, end=len(df_7daily)-1, exog=exog_train_7d)

    # df_plot = pd.DataFrame({'Actual': endog_train_7d, 'Predicted': predictions})
    # st.line_chart(df_plot)

    # mae = mean_absolute_error(endog_train_7d, predictions)
    # mse = mean_squared_error(endog_train_7d, predictions)
    # r2 = r2_score(endog_train_7d, predictions)
    # mape = (abs((endog_train_7d - predictions) / endog_train_7d)).mean() * 100

    # st.write(f"MAE: {mae:.2f}")
    # st.write(f"MSE: {mse:.2f}")
    # st.write(f"R2 Score: {r2:.2f}")
    # st.write(f"MAPE: {mape:.2f}%")

    # # MONTHLY
    # # # Grid params
    # p = P = range(0, 4)
    # d = D = range(0, 1)
    # q = Q = range(0, 2)
    # s = [1, 3, 6, 12]

    # best_cfg, best_model, best_score = sarimax_grid_search(
    #     endog_train_30d, exog_train_30d,
    #     p, d, q,
    #     P, D, Q, s,
    #     scoring='mse'
    # )

    # st.write(f"Best SARIMAX order: {best_cfg[0]}, seasonal_order: {best_cfg[1]}, MSE: {best_score:.2f}")

    # params_df = pd.DataFrame({
    #     'order_p': [best_cfg[0][0]],
    #     'order_d': [best_cfg[0][1]],
    #     'order_q': [best_cfg[0][2]],
    #     'seasonal_order_P': [best_cfg[1][0]],
    #     'seasonal_order_D': [best_cfg[1][1]],
    #     'seasonal_order_Q': [best_cfg[1][2]],
    #     'seasonal_order_s': [best_cfg[1][3]],
    #     'MSE': [best_score]
    # })

    # predictions = best_model.predict(start=0, end=len(df_30daily)-1, exog=exog_train_30d)

    # df_plot = pd.DataFrame({'Actual': endog_train_30d, 'Predicted': predictions})
    # st.line_chart(df_plot)

    # mae = mean_absolute_error(endog_train_30d, predictions)
    # mse = mean_squared_error(endog_train_30d, predictions)
    # r2 = r2_score(endog_train_30d, predictions)
    # mape = (abs((endog_train_30d - predictions) / endog_train_30d)).mean() * 100

    # st.write(f"MAE: {mae:.2f}")
    # st.write(f"MSE: {mse:.2f}")
    # st.write(f"R2 Score: {r2:.2f}")
    # st.write(f"MAPE: {mape:.2f}%")