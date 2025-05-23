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

    # Buat fitur tambahan
    df_daily['weekday'] = df_daily['Segments/Departure Date'].dt.weekday
    df_daily['is_weekend'] = df_daily['weekday'].isin([5, 6]).astype(int)
    df_daily['month'] = df_daily['Segments/Departure Date'].dt.month
    df_daily['day_of_month'] = df_daily['Segments/Departure Date'].dt.day
    df_daily['lag_1'] = df_daily['Total Pax'].shift(1)
    df_daily['lag_7'] = df_daily['Total Pax'].shift(7)
    df_daily['rolling_mean_3'] = df_daily['Total Pax'].rolling(3).mean()
    df_daily['rolling_mean_7'] = df_daily['Total Pax'].rolling(7).mean()
    df_daily.dropna(inplace=True)

    # Fitur yang digunakan
    fitur = ['holiday', 'weekday', 'is_weekend', 'month', 'day_of_month',
             'lag_1', 'lag_7', 'rolling_mean_3', 'rolling_mean_7']

    X = df_daily[fitur]
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

    # Dataframe koefisien
    coef_df = pd.DataFrame({
        'Fitur': X.columns,
        'Koefisien': reg.coef_
    })

    # Tandai koefisien besar/kecil
    coef_df['Kategori'] = coef_df['Koefisien'].apply(
        lambda x: 'Besar' if abs(x) >= 0.5 else 'Kecil'
    )

    # Interpretasi koefisien
    def interpretasi_koefisien(row):
        fitur = row['Fitur']
        coef = row['Koefisien']
        if fitur == 'holiday':
            return f"Jika hari tersebut libur (1), jumlah penumpang cenderung {'bertambah' if coef > 0 else 'berkurang'} ~{abs(coef):.2f} pax."
        elif fitur == 'weekday':
            return f"Tiap kenaikan weekday (misal Senin ke Selasa), pax {'naik' if coef > 0 else 'turun'} ~{abs(coef):.2f}."
        elif fitur == 'is_weekend':
            return f"Jika akhir pekan, pax {'bertambah' if coef > 0 else 'berkurang'} ~{abs(coef):.2f} dibanding hari biasa."
        elif fitur == 'month':
            return f"Tiap kenaikan bulan, pax {'naik' if coef > 0 else 'turun'} sedikit (~{abs(coef):.3f})."
        elif fitur == 'day_of_month':
            return f"Semakin akhir bulan, pax {'naik' if coef > 0 else 'turun'} ~{abs(coef):.3f}."
        elif fitur == 'lag_1':
            return f"Jika kemarin tinggi, hari ini cenderung {'naik' if coef > 0 else 'turun'} ~{abs(coef):.2f}."
        elif fitur == 'lag_7':
            return f"Jika minggu lalu tinggi, hari ini sedikit {'naik' if coef > 0 else 'turun'} (~{abs(coef):.3f})."
        elif fitur == 'rolling_mean_3':
            return f"Jika rata-rata 3 hari terakhir tinggi, pax hari ini cenderung {'naik' if coef > 0 else 'turun'} ~{abs(coef):.2f}."
        elif fitur == 'rolling_mean_7':
            return f"Jika rata-rata 7 hari terakhir tinggi, pax hari ini cenderung {'naik' if coef > 0 else 'turun'} ~{abs(coef):.2f}."
        else:
            return f"Pengaruh umum terhadap Total Pax: {'positif' if coef > 0 else 'negatif'} ~{abs(coef):.2f}"

    coef_df['Interpretasi'] = coef_df.apply(interpretasi_koefisien, axis=1)

    # Urutkan berdasarkan magnitude
    coef_df = coef_df.reindex(
        coef_df['Koefisien'].abs().sort_values(ascending=False).index)

    # Highlight koefisien terbesar & terkecil
    def highlight_extremes(s):
        if s.name == 'Koefisien':
            threshold = 0.5
            return [
                'background-color: lightgreen' if v > threshold
                else 'background-color: lightcoral' if v < -threshold
                else ''
                for v in s
            ]
        return ['' for _ in s]

    st.subheader('Koefisien dan Interpretasi')
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

    Metode statistik yang digunakan untuk memodelkan hubungan antara satu variabel target (dalam hal ini jumlah penumpang) dengan satu atau lebih variabel prediktor (fitur).

    #### Fitur

    Fitur adalah variabel yang digunakan model untuk memprediksi target.
    
    #### Koefisien

    Koefisien menunjukkan seberapa besar pengaruh setiap fitur terhadap variabel target.  
    - Koefisien positif berarti fitur tersebut meningkatkan nilai target.  
    - Koefisien negatif berarti fitur tersebut menurunkan nilai target.  
    - Besarnya koefisien menunjukkan kekuatan pengaruh fitur.

    #### R-squared (Koefisien Determinasi)

    R-squared adalah ukuran seberapa baik model menjelaskan variabilitas data target.  
    - Nilai R-squared berkisar antara 0 sampai 1.  
    - Semakin mendekati 1, model semakin baik dalam memprediksi data.  

    """)
