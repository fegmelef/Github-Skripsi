import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
from datetime import datetime
import calendar
from utils import load_data, filter_year_month, filter_year_month_depart
from streamlit_card import card
import altair as alt
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


def render_linear_regression(df, start_year, end_year):
    col1, col2 = st.columns(2)
    with col1:
        buffer_days = st.number_input(
            "Enter the number of buffer days (0–7):", min_value=0, max_value=7, value=1)
    with col2:
        buffer_option = st.radio(
            "Buffer Day Type:",
            options=["Before", "After", "Before and After"],
            index=2,
            horizontal=True
        )

    # Proses awal data libur
    df_holiday = pd.read_excel('files/holidays_ID.xlsx')
    df_holiday['Date'] = pd.to_datetime(df_holiday['Date'])
    df_holiday = df_holiday[['Date', 'Libur']].rename(
        columns={'Date': 'Segments/Departure Date', 'Libur': 'holiday'})

    df['Segments/Departure Date'] = pd.to_datetime(
        df['Segments/Departure Date'])
    df['Booking Date'] = pd.to_datetime(df['Booking Date'])

    dftest = df.dropna(subset=['Segments/Departure Date'])
    # dftest = dftest.fillna(method='ffill')

    df_daily = dftest.groupby(
        dftest['Segments/Departure Date'].dt.date)['Total Pax'].sum().reset_index()
    df_daily['Segments/Departure Date'] = pd.to_datetime(
        df_daily['Segments/Departure Date'])

    # Tambahkan semua tanggal
    full_dates = pd.DataFrame({
        'Segments/Departure Date': pd.date_range(start=df_daily['Segments/Departure Date'].min(),
                                                 end=df_daily['Segments/Departure Date'].max())
    })
    df_daily = full_dates.merge(
        df_daily, on='Segments/Departure Date', how='left')
    df_daily['Total Pax'] = df_daily['Total Pax'].fillna(0).astype(int)

    # Merge hari libur awal
    calendar = pd.DataFrame({'Segments/Departure Date': pd.date_range(start=df_daily['Segments/Departure Date'].min(),
                                                                      end=df_daily['Segments/Departure Date'].max())})
    calendar = calendar.merge(
        df_holiday, on='Segments/Departure Date', how='left')
    calendar['holiday'] = calendar['holiday'].fillna(0).astype(int)

    # Deteksi libur panjang
    calendar['grp'] = (calendar['holiday'] !=
                       calendar['holiday'].shift()).cumsum()
    calendar['seq'] = calendar.groupby('grp').cumcount()
    holiday_groups = calendar.groupby('grp').agg({
        'holiday': 'sum',
        'Segments/Departure Date': ['first', 'last']
    })
    holiday_groups.columns = ['count', 'start', 'end']
    long_holidays = holiday_groups[holiday_groups['count'] >= 3]

    # Tambah buffer date
    buffer_dates = set()
    for _, row in long_holidays.iterrows():
        if buffer_option in ["Before", "Before and After"]:
            for i in range(1, buffer_days + 1):
                buffer_dates.add(row['start'] - pd.Timedelta(days=i))
        if buffer_option in ["After", "Before and After"]:
            for i in range(1, buffer_days + 1):
                buffer_dates.add(row['end'] + pd.Timedelta(days=i))

    # Tandai ulang hari libur
    calendar['holiday'] = calendar['Segments/Departure Date'].isin(
        calendar[calendar['holiday'] ==
                 1]['Segments/Departure Date'].tolist() + list(buffer_dates)
    ).astype(int)

    # Gabung ke df_daily
    df_daily = df_daily.drop(columns='holiday', errors='ignore')
    df_daily = df_daily.merge(
        calendar[['Segments/Departure Date', 'holiday']],
        on='Segments/Departure Date',
        how='left'
    )

    # Buat Feature tambahan
    df_daily['weekday'] = df_daily['Segments/Departure Date'].dt.weekday
    df_daily['is_weekend'] = df_daily['weekday'].isin([5, 6]).astype(int)
    df_daily['month'] = df_daily['Segments/Departure Date'].dt.month
    df_daily['day_of_month'] = df_daily['Segments/Departure Date'].dt.day
    df_daily['lag_1'] = df_daily['Total Pax'].shift(1)
    df_daily['lag_7'] = df_daily['Total Pax'].shift(7)
    df_daily['rolling_mean_3'] = df_daily['Total Pax'].rolling(3).mean()
    df_daily['rolling_mean_7'] = df_daily['Total Pax'].rolling(7).mean()
    df_daily.dropna(inplace=True)

    # Feature yang digunakan
    Feature = ['holiday', 'weekday', 'is_weekend', 'month', 'day_of_month',
             'lag_1', 'lag_7', 'rolling_mean_3', 'rolling_mean_7']

    X = df_daily[Feature]
    y = df_daily['Total Pax']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Standardisasi
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model regresi
    reg = LinearRegression()
    reg.fit(X_train_scaled, y_train)

    # Prediksi dan evaluasi
    y_pred = reg.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)

    # Dataframe Coefficient
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': reg.coef_
    })

    # Tandai Coefficient besar/kecil
    coef_df['Category'] = coef_df['Coefficient'].apply(
        lambda x: 'Significant' if abs(x) >= 0.5 else 'Not Significant'
    )

    # Interpretation Coefficient
    def interpret_coefficient(row):
        feature = row['Feature']
        coef = row['Coefficient']
        if feature == 'holiday':
            return f"If the day is a holiday (1), the number of passengers tends to {'increase' if coef > 0 else 'decrease'} by ~{abs(coef):.2f} pax."
        elif feature == 'weekday':
            return f"For each increase in weekday (e.g., Monday to Tuesday), passengers {'increase' if coef > 0 else 'decrease'} by ~{abs(coef):.2f}."
        elif feature == 'is_weekend':
            return f"If it is the weekend, passengers {'increase' if coef > 0 else 'decrease'} by ~{abs(coef):.2f} compared to weekdays."
        elif feature == 'month':
            return f"For each increase in month, passengers {'increase' if coef > 0 else 'decrease'} slightly (~{abs(coef):.3f})."
        elif feature == 'day_of_month':
            return f"Later in the month, passengers tend to {'increase' if coef > 0 else 'decrease'} by ~{abs(coef):.3f}."
        elif feature == 'lag_1':
            return f"If yesterday's number was high, today's tends to {'increase' if coef > 0 else 'decrease'} by ~{abs(coef):.2f}."
        elif feature == 'lag_7':
            return f"If last week's value was high, today tends to {'increase' if coef > 0 else 'decrease'} slightly (~{abs(coef):.3f})."
        elif feature == 'rolling_mean_3':
            return f"If the 3-day average was high, today's passengers tend to {'increase' if coef > 0 else 'decrease'} by ~{abs(coef):.2f}."
        elif feature == 'rolling_mean_7':
            return f"If the 7-day average was high, today's passengers tend to {'increase' if coef > 0 else 'decrease'} by ~{abs(coef):.2f}."
        else:
            return f"General effect on Total Pax: {'positive' if coef > 0 else 'negative'} ~{abs(coef):.2f}"

    coef_df['Interpretation'] = coef_df.apply(interpret_coefficient, axis=1)

    # Urutkan berdasarkan magnitude
    coef_df = coef_df.reindex(
        coef_df['Coefficient'].abs().sort_values(ascending=False).index)

    # Highlight Coefficient terbesar & terkecil
    def highlight_extremes(s):
        if s.name == 'Coefficient':
            threshold = 0.5
            return [
                'background-color: lightgreen' if v > threshold
                else 'background-color: lightcoral' if v < -threshold
                else ''
                for v in s
            ]
        return ['' for _ in s]

    st.subheader('Coefficient dan Interpretation')
    st.dataframe(coef_df.style.apply(highlight_extremes, axis=0))

    # Buat DataFrame hasil prediksi
    result_df = pd.DataFrame({
        'Tanggal': X_test.index,
        'Actual': y_test,
        'Predicted': y_pred
    }).sort_values('Tanggal')

    result_df = result_df.set_index('Tanggal')[['Actual', 'Predicted']]

    st.subheader('Comparison of Actual vs Predicted')
    st.line_chart(result_df)

    st.write(f"R-squared: {r2:.4f}")
    st.write(f"---------")
    st.write("Glossary:")

    st.markdown("""
    #### Linear Regression

    A statistical method used to model the relationship between a target variable (in this case, the number of passengers) and one or more predictor variables (features).

    #### Features

    Features are the variables used by the model to predict the target.

    #### Coefficients

    Coefficients indicate how much influence each feature has on the target variable.  
    - A positive coefficient means the feature increases the target value.  
    - A negative coefficient means the feature decreases the target value.  
    - The magnitude of the coefficient indicates the strength of the feature's influence.

    #### R-squared (Coefficient of Determination)

    R-squared is a measure of how well the model explains the variability of the target data.  
    - R-squared values range from 0 to 1.  
    - The closer the value is to 1, the better the model predicts the data.  
    """)
