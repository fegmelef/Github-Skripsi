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


def render_ttest(df, start_year, end_year):    
    df_holiday = pd.read_excel('files/holidays_ID.xlsx')
    df_holiday['Date'] = pd.to_datetime(df_holiday['Date'])
    df_holiday = df_holiday[['Date', 'Libur']].rename(columns={'Date': 'Segments/Departure Date', 'Libur': 'holiday'})

    df['Segments/Departure Date'] = pd.to_datetime(df['Segments/Departure Date'])
    df['Segments/Departure Date'] = pd.to_datetime(df['Segments/Departure Date'])
    df['Booking Date'] = pd.to_datetime(df['Booking Date'])

    start_year = df['Segments/Departure Date'].min().year
    end_year = df['Segments/Departure Date'].max().year

    df_daily = df.groupby(df['Segments/Departure Date'].dt.date)['Total Pax'].sum().reset_index()
    df_daily['Segments/Departure Date'] = pd.to_datetime(df_daily['Segments/Departure Date'])

    full_dates = pd.DataFrame({
        'Segments/Departure Date': pd.date_range(start=df_daily['Segments/Departure Date'].min(), end=df_daily['Segments/Departure Date'].max())
    })

    df_daily = full_dates.merge(df_daily, on='Segments/Departure Date', how='left')
    df_daily['Total Pax'] = df_daily['Total Pax'].fillna(0).astype(int)

    df_daily = df_daily.merge(df_holiday, on='Segments/Departure Date', how='left')
    df_daily['holiday'] = df_daily['holiday'].fillna(0).astype(int)

    st.write(df_daily)
    
    from scipy.stats import ttest_ind

    # Pisahkan data berdasarkan hari libur dan non-libur
    pax_holiday = df_daily[df_daily['holiday'] == 1]['Total Pax']
    pax_non_holiday = df_daily[df_daily['holiday'] == 0]['Total Pax']

    # Tampilkan ringkasan jumlah dan rata-rata
    total_holiday = pax_holiday.sum()
    total_non_holiday = pax_non_holiday.sum()
    mean_holiday = pax_holiday.mean()
    mean_non_holiday = pax_non_holiday.mean()

    st.write(f"Jumlah Total Pax saat hari libur: {total_holiday} (rata-rata per hari: {mean_holiday:.2f})")
    st.write(f"Jumlah Total Pax saat non-libur: {total_non_holiday} (rata-rata per hari: {mean_non_holiday:.2f})")

    # Lakukan uji t
    t_stat, p_value = ttest_ind(pax_holiday, pax_non_holiday, equal_var=False)

    st.write(f"T-statistic: {t_stat}")
    st.write(f"P-value: {p_value}")

    if p_value < 0.05:
        st.write("Hasil: Terdapat perbedaan signifikan Total Pax antara hari libur dan non-libur.")
    else:
        st.write("Hasil: Tidak terdapat perbedaan signifikan Total Pax antara hari libur dan non-libur.")

    import statsmodels.api as sm

    # Siapkan data
    X = df_daily[['holiday']]  # variabel independen
    X = sm.add_constant(X)     # tambahkan konstanta (intercept)
    y = df_daily['Total Pax']  # variabel dependen

    # Fit model regresi linear
    model = sm.OLS(y, X).fit()

    # Tampilkan ringkasan hasil regresi
    st.write(model.summary())
    
    st.write(f"Intercept: {model.params['const']}")
    st.write(f"Koefisien holiday: {model.params['holiday']}")
    st.write(f"P-value holiday: {model.pvalues['holiday']}")
    st.write(f"R-squared: {model.rsquared}")
