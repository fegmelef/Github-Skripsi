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
from components.linearregression import render_linear_regression

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
    ('GRU', 'SARIMAX', 'ARIMA', 'ARIMAX', 'Linear Regression')
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

if model_option == 'Linear Regression':
    render_linear_regression(df, start_year, end_year)

if model_option == 'GRU':
    from itertools import product

    lookback_option = st.sidebar.selectbox(
        'Choose Time Window (in Days):',
        (3, 7, 30)
    )

    st.subheader(
        f"GRU Forecast - Actual Vs Predicted ({lookback_option} Day{'s)' if lookback_option > 1 else ')'}")

    LOOK_BACK = lookback_option
    TRAIN_RATIO = 0.8

    features = df_daily[['Total Pax', 'holiday']].copy().values
    n = int(len(features) * TRAIN_RATIO)

    scaler = MinMaxScaler().fit(features[:n])
    features_scaled = scaler.transform(features)

    train_scaled = features_scaled[:n]
    test_scaled = features_scaled[n:]

    def create_dataset(X, look_back):
        Xs, ys = [], []
        for i in range(len(X) - look_back):
            Xs.append(X[i:i+look_back])
            ys.append(X[i+look_back][0])
        return np.array(Xs), np.array(ys)

    X_train, y_train = create_dataset(train_scaled, LOOK_BACK)
    X_test, y_test = create_dataset(test_scaled, LOOK_BACK)

    def create_gru(units, dropout_rate, optimizer):
        model = Sequential()
        model.add(GRU(units=units, return_sequences=True, input_shape=(LOOK_BACK, X_train.shape[2])))
        model.add(Dropout(dropout_rate))
        model.add(GRU(units=units))
        model.add(Dropout(dropout_rate))
        model.add(Dense(units=1))
        model.compile(optimizer=optimizer, loss='mse')
        return model

    st.subheader(f"GRU Grid Search - Time Window: {LOOK_BACK} days")

    param_grid = {
        'units': [32, 64],
        'epochs': [200],
        'batch_size': [16, 32],
        'dropout_rate': [0.2, 0.5],
        'optimizer': ['nadam', 'adam'],
    }
    grid_params = list(product(*param_grid.values()))
    best_rmse = float('inf')
    best_params = None
    best_model_path = ''

    with st.spinner('Running GRU Grid Search...'):
        for units, epochs, batch_size, dropout_rate, optimizer in grid_params:
            model_filename = f"gru_grid_{LOOK_BACK}d_{units}u_{epochs}e_{batch_size}b_{int(dropout_rate*100)}d_{optimizer}.keras"
            if os.path.exists(model_filename):
                model = load_model(model_filename)
            else:
                model = create_gru(units, dropout_rate, optimizer)
                model.fit(X_train, y_train, epochs=epochs,
                        validation_split=0.2, batch_size=batch_size, shuffle=False, verbose=0)
                model.save(model_filename)

            y_pred_scaled = model.predict(X_test, verbose=0)
            y_pred = scaler.inverse_transform(np.hstack([y_pred_scaled, X_test[:, -1, 1][:, None]]))[:, 0]
            y_true = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), X_test[:, -1, 1][:, None]]))[:, 0]
            rmse = np.sqrt(np.mean((y_pred - y_true)**2))

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = (units, epochs, batch_size, dropout_rate, optimizer)
                best_model_path = model_filename

    if best_params:
        # st.success(f"Best GRU: units={best_params[0]}, epochs={best_params[1]}, batch={best_params[2]}, dropout={best_params[3]}, opt={best_params[4]}, RMSE={best_rmse:.2f}")
        best_model = load_model(best_model_path)

        y_pred_scaled = best_model.predict(X_test, verbose=0)
        y_pred = scaler.inverse_transform(np.hstack([y_pred_scaled, X_test[:, -1, 1][:, None]]))[:, 0]
        y_true = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), X_test[:, -1, 1][:, None]]))[:, 0]

        st.line_chart(pd.DataFrame({'Actual': y_true.flatten(), 'Prediction': y_pred.flatten()}))

        errors = y_pred - y_true
        mae = np.abs(errors).mean()
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        def forecast_next_days(model, last_sequence, future_holiday, steps, scaler):
            predictions = []
            seq = last_sequence.copy()

            for i in range(steps):
                input_seq = seq[-LOOK_BACK:].reshape(1, LOOK_BACK, -1)
                pred_scaled = model.predict(input_seq, verbose=0)[0, 0]
                new_input = np.array([pred_scaled, future_holiday[i]])
                seq = np.vstack([seq, new_input])
                predictions.append(new_input)

            predictions = np.array(predictions)
            return scaler.inverse_transform(predictions)[:, 0]

        last_sequence = features_scaled[-LOOK_BACK:]
        future_steps = LOOK_BACK
        future_holiday = [0] * future_steps
        future_pred = forecast_next_days(best_model, last_sequence, future_holiday, future_steps, scaler)

        future_dates = pd.date_range(
            start=df_daily['Segments/Departure Date'].max(), periods=future_steps + 1, freq='D')[1:]

        st.subheader(f"GRU Forecast - Next {LOOK_BACK} Day{'s' if LOOK_BACK > 1 else ''}")
        st.line_chart(pd.DataFrame(
            {f'Forecasted Pax ({LOOK_BACK} Days Time Span)': future_pred.flatten()}, index=future_dates))

        st.markdown(f"""
        ### GRU Grid Search - Evaluation

        - **Best Params:** units={best_params[0]}, epochs={best_params[1]}, batch={best_params[2]}, dropout={best_params[3]}, optimizer={best_params[4]}
        - **MAE:** {mae:.2f}  
        - **RMSE:** {best_rmse:.2f}  
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

# Persiapan data harian
df['Segments/Departure Date'] = pd.to_datetime(
    df['Segments/Departure Date']).dt.normalize()
df_daily = df.groupby(
    'Segments/Departure Date')['Total Pax'].sum().reset_index()
df_daily = df_daily.merge(
    df_holiday, on='Segments/Departure Date', how='left')
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
df_7daily['Segments/Departure Date'] = df_7daily['Segments/Departure Date'] - \
    pd.to_timedelta(6, unit='d')
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

train_ratio = 1

# Harian
n_train_daily = int(len(df_daily) * train_ratio)
endog_train_daily = df_daily['Total Pax'].iloc[:n_train_daily]
exog_train_daily = df_daily[['holiday']].iloc[:n_train_daily]

# 7 Harian
n_train_7d = int(len(df_7daily) * train_ratio)
endog_train_7d = df_7daily['Total Pax'].iloc[:n_train_7d]
exog_train_7d = df_7daily[['holiday']].iloc[:n_train_7d]

# 30 Harian
n_train_30d = int(len(df_30daily) * train_ratio)
endog_train_30d = df_30daily['Total Pax'].iloc[:n_train_30d]
exog_train_30d = df_30daily[['holiday']].iloc[:n_train_30d]

if model_option == 'SARIMAX':
    def sarimax_grid_search(endog_train, exog_train,
                            p_values, d_values, q_values,
                            P_values, D_values, Q_values, s_values,
                            scoring='mse'):
        best_score = np.inf
        best_cfg = None
        best_model_fit = None

        seasonal_params = list(itertools.product(
            P_values, D_values, Q_values, s_values))
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
                    pred = model_fit.predict(start=0, end=len(
                        endog_train)-1, exog=exog_train)
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
    
    def fit_and_plot_sarimax(endog, exog, order, seasonal_order, df_plot_base, title):
        model = SARIMAX(endog,
                        exog=exog,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        predictions = model_fit.predict(start=0, end=len(endog)-1, exog=exog)

        df_plot = pd.DataFrame({'Actual': endog, 'Predicted': predictions})
        st.subheader(title)
        st.line_chart(df_plot)

        mae = mean_absolute_error(endog, predictions)
        mse = mean_squared_error(endog, predictions)
        r2 = r2_score(endog, predictions)
        mape = (abs((endog - predictions) / endog)).mean() * 100

        st.write(f"Order: {order}, Seasonal Order: {seasonal_order}")
        st.write(f"MAE: {mae:.2f}")
        st.write(f"MSE: {mse:.2f}")
        st.write(f"R2 Score: {r2:.2f}")
        st.write(f"MAPE: {mape:.2f}%")


    st.title("SARIMAX Grid Search Forecasting")

    # Load data hari libur
    df_holiday = pd.read_excel('files/holidays_ID.xlsx')
    df_holiday['Segments/Departure Date'] = pd.to_datetime(df_holiday['Date'])
    df_holiday['holiday'] = df_holiday['Holiday Name'].notna().astype(int)
    df_holiday = df_holiday[['Segments/Departure Date', 'holiday']]

    sarimaxOpt = st.sidebar.selectbox(
        'Choose Time Window (in Days):',
        ('Daily', 'Weekly', 'Monthly')
    )

    if sarimaxOpt == 'Daily':
        fit_and_plot_sarimax(
            endog=endog_train_daily,
            exog=exog_train_daily,
            order=(0, 0, 3),
            seasonal_order=(1, 0, 2, 7),
            df_plot_base=df_daily,
            title="SARIMAX - Daily"
        )

    elif sarimaxOpt == 'Weekly':
        fit_and_plot_sarimax(
            endog=endog_train_7d,
            exog=exog_train_7d,
            order=(1, 0, 1),
            seasonal_order=(0, 0, 1, 26),
            df_plot_base=df_7daily,
            title="SARIMAX - Weekly"
        )

    else:  # Monthly
        fit_and_plot_sarimax(
            endog=endog_train_30d,
            exog=exog_train_30d,
            order=(1, 0, 1),
            seasonal_order=(0, 0, 1, 3),
            df_plot_base=df_30daily,
            title="SARIMAX - Monthly"
        )



def evaluate_model(actual, predicted, model_name, order=None, seasonal_order=None):
    df_plot = pd.DataFrame({'Actual': actual, 'Predicted': predicted})
    st.line_chart(df_plot)
    
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    mape = (abs((actual - predicted) / actual)).mean() * 100
    
    if order:
        st.write(f"Order: {order}")
    if seasonal_order:
        st.write(f"Seasonal Order: {seasonal_order}")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"MSE: {mse:.2f}")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"R2 Score: {r2:.4f}")
    st.write(f"MAPE: {mape:.2f}%")


from statsmodels.tsa.arima.model import ARIMA

# ARIMA Implementation
def run_arima_gridsearch(endog, name="Daily"):
    st.subheader(f"ARIMA Grid Search - {name}")

    p_range = range(0, 4)
    d_range = range(0, 2)
    q_range = range(0, 4)
    pdq = list(itertools.product(p_range, d_range, q_range))

    results = []
    for order in pdq:
        try:
            model = ARIMA(endog, order=order)
            model_fit = model.fit()
            pred = model_fit.predict(start=0, end=len(endog)-1)
            rmse = np.sqrt(mean_squared_error(endog, pred))
            results.append((order, rmse))
        except:
            continue

    if results:
        best_order, best_rmse = sorted(results, key=lambda x: x[1])[0]
        model = ARIMA(endog, order=best_order)
        model_fit = model.fit()
        forecast = model_fit.predict(start=0, end=len(endog)-1)
        
        st.write(f"**Best ARIMA Order ({name})**: {best_order} with RMSE: {best_rmse:.2f}")
        evaluate_model(endog.values, forecast.values, model_name=f"ARIMA ({name})", order=best_order)
    else:
        st.error(f"No valid ARIMA models for {name} data.")

# --- Run Grid Search ---
if model_option == 'ARIMA':
    st.title("ARIMA Grid Search Forecasting")
    resolution_option = st.sidebar.selectbox("Select time resolution:", ["Daily", "Weekly", "Monthly"])

    if resolution_option == "Daily":
        run_arima_gridsearch(endog_train_daily, name="Daily")
    elif resolution_option == "Weekly":
        run_arima_gridsearch(endog_train_7d, name="Weekly")
    elif resolution_option == "Monthly":
        run_arima_gridsearch(endog_train_30d, name="Monthly")


# ARIMAX Implementation
def run_arimax_gridsearch(endog, exog, name=""):
    st.subheader(f"ARIMAX Grid Search - {name}")

    p_range = range(0, 4)
    d_range = range(0, 2)
    q_range = range(0, 4)
    pdq = list(itertools.product(p_range, d_range, q_range))

    results = []
    for order in pdq:
        try:
            model = ARIMA(endog, exog=exog, order=order)
            model_fit = model.fit()
            forecast = model_fit.predict(start=0, end=len(endog)-1, exog=exog)
            rmse = np.sqrt(mean_squared_error(endog, forecast))
            results.append((order, rmse))
        except:
            continue

    if results:
        best_order, best_rmse = sorted(results, key=lambda x: x[1])[0]
        model = ARIMA(endog, exog=exog, order=best_order)
        model_fit = model.fit()
        forecast = model_fit.predict(start=0, end=len(endog)-1, exog=exog)

        st.write(f"**Best ARIMAX Order ({name})**: {best_order} with RMSE: {best_rmse:.2f}")
        evaluate_model(endog.values, forecast.values, model_name=f"ARIMAX ({name})", order=best_order)
    else:
        st.error(f"No valid ARIMAX models for {name} data.")

# Main execution
if model_option == 'ARIMAX':
    st.title("ARIMAX Grid Search Forecasting")
    resolution_option = st.sidebar.selectbox("Select time resolution:", ["Daily", "Weekly", "Monthly"])

    if resolution_option == "Daily":
        run_arimax_gridsearch(endog_train_daily, exog_train_daily, name="Daily")
    elif resolution_option == "Weekly":
        run_arimax_gridsearch(endog_train_7d, exog_train_7d, name="Weekly")
    elif resolution_option == "Monthly":
        run_arimax_gridsearch(endog_train_30d, exog_train_30d, name="Monthly")















    # #### ini buat grid search SARIMAX
    # if sarimaxOpt == 'Daily':
    #     # Daily
    #     # # Grid params
    #     p = P = range(0, 4)
    #     d = D = range(0, 1)
    #     q = Q = range(0, 4)
    #     s = [3, 7, 14]

    #     best_cfg, best_model, best_score = sarimax_grid_search(
    #         endog_train_daily, exog_train_daily,
    #         p, d, q,
    #         P, D, Q, s,
    #         scoring='mse'
    #     )

    #     st.write(
    #         f"Best SARIMAX order: {best_cfg[0]}, seasonal_order: {best_cfg[1]}, MSE: {best_score:.2f}")

    #     params_df = pd.DataFrame({
    #         'order_p': [best_cfg[0][0]],
    #         'order_d': [best_cfg[0][1]],
    #         'order_q': [best_cfg[0][2]],
    #         'seasonal_order_P': [best_cfg[1][0]],
    #         'seasonal_order_D': [best_cfg[1][1]],
    #         'seasonal_order_Q': [best_cfg[1][2]],
    #         'seasonal_order_s': [best_cfg[1][3]],
    #         'MSE': [best_score]
    #     })

    #     predictions = best_model.predict(
    #         start=0, end=len(df_daily)-1, exog=exog_train_daily)

    #     df_plot = pd.DataFrame(
    #         {'Actual': endog_train_daily, 'Predicted': predictions})
    #     st.line_chart(df_plot)

    #     mae = mean_absolute_error(endog_train_daily, predictions)
    #     mse = mean_squared_error(endog_train_daily, predictions)
    #     r2 = r2_score(endog_train_daily, predictions)
    #     mape = (abs((endog_train_daily - predictions) /
    #             endog_train_daily)).mean() * 100

    #     st.write(f"MAE: {mae:.2f}")
    #     st.write(f"MSE: {mse:.2f}")
    #     st.write(f"R2 Score: {r2:.2f}")
    #     st.write(f"MAPE: {mape:.2f}%")

    # elif sarimaxOpt == 'Weekly':
    #     # Weekly
    #     # Grid params
    #     p = P = range(0, 2)
    #     d = D = range(0, 1)
    #     q = Q = range(0, 2)
    #     s = [4, 13, 26, 52]

    #     best_cfg, best_model, best_score = sarimax_grid_search(
    #         endog_train_7d, exog_train_7d,
    #         p, d, q,
    #         P, D, Q, s,
    #         scoring='mse'
    #     )

    #     st.write(
    #         f"Best SARIMAX order: {best_cfg[0]}, seasonal_order: {best_cfg[1]}, MSE: {best_score:.2f}")

    #     params_df = pd.DataFrame({
    #         'order_p': [best_cfg[0][0]],
    #         'order_d': [best_cfg[0][1]],
    #         'order_q': [best_cfg[0][2]],
    #         'seasonal_order_P': [best_cfg[1][0]],
    #         'seasonal_order_D': [best_cfg[1][1]],
    #         'seasonal_order_Q': [best_cfg[1][2]],
    #         'seasonal_order_s': [best_cfg[1][3]],
    #         'MSE': [best_score]
    #     })

    #     predictions = best_model.predict(
    #         start=0, end=len(df_7daily)-1, exog=exog_train_7d)

    #     df_plot = pd.DataFrame(
    #         {'Actual': endog_train_7d, 'Predicted': predictions})
    #     st.line_chart(df_plot)

    #     mae = mean_absolute_error(endog_train_7d, predictions)
    #     mse = mean_squared_error(endog_train_7d, predictions)
    #     r2 = r2_score(endog_train_7d, predictions)
    #     mape = (abs((endog_train_7d - predictions) / endog_train_7d)).mean() * 100

    #     st.write(f"MAE: {mae:.2f}")
    #     st.write(f"MSE: {mse:.2f}")
    #     st.write(f"R2 Score: {r2:.2f}")
    #     st.write(f"MAPE: {mape:.2f}%")

    # else:
    #     # MONTHLY
    #     # # Grid params
    #     p = P = range(0, 4)
    #     d = D = range(0, 1)
    #     q = Q = range(0, 2)
    #     s = [1, 3, 6, 12]

    #     best_cfg, best_model, best_score = sarimax_grid_search(
    #         endog_train_30d, exog_train_30d,
    #         p, d, q,
    #         P, D, Q, s,
    #         scoring='mse'
    #     )

    #     st.write(
    #         f"Best SARIMAX order: {best_cfg[0]}, seasonal_order: {best_cfg[1]}, MSE: {best_score:.2f}")

    #     params_df = pd.DataFrame({
    #         'order_p': [best_cfg[0][0]],
    #         'order_d': [best_cfg[0][1]],
    #         'order_q': [best_cfg[0][2]],
    #         'seasonal_order_P': [best_cfg[1][0]],
    #         'seasonal_order_D': [best_cfg[1][1]],
    #         'seasonal_order_Q': [best_cfg[1][2]],
    #         'seasonal_order_s': [best_cfg[1][3]],
    #         'MSE': [best_score]
    #     })

    #     predictions = best_model.predict(
    #         start=0, end=len(df_30daily)-1, exog=exog_train_30d)

    #     df_plot = pd.DataFrame(
    #         {'Actual': endog_train_30d, 'Predicted': predictions})
    #     st.line_chart(df_plot)

    #     mae = mean_absolute_error(endog_train_30d, predictions)
    #     mse = mean_squared_error(endog_train_30d, predictions)
    #     r2 = r2_score(endog_train_30d, predictions)
    #     mape = (abs((endog_train_30d - predictions) / endog_train_30d)).mean() * 100

    #     st.write(f"MAE: {mae:.2f}")
    #     st.write(f"MSE: {mse:.2f}")
    #     st.write(f"R2 Score: {r2:.2f}")
    #     st.write(f"MAPE: {mape:.2f}%")
    
    
    
    
    

    # # Plot ACF dan PACF
    # from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    # from statsmodels.tsa.stattools import adfuller

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