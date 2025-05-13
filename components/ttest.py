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
from scipy.stats import ttest_ind


def render_ttest(df, start_year, end_year):
    df_holiday = pd.read_excel('files/holidays_ID.xlsx')
    df_holiday['Date'] = pd.to_datetime(df_holiday['Date'])
    df_holiday = df_holiday[['Date', 'Libur']].rename(
        columns={'Date': 'Segments/Departure Date', 'Libur': 'holiday'})

    df['Segments/Departure Date'] = pd.to_datetime(
        df['Segments/Departure Date'])
    df['Booking Date'] = pd.to_datetime(df['Booking Date'])

    df_daily = df.groupby(
        df['Segments/Departure Date'].dt.date)['Total Pax'].sum().reset_index()
    df_daily['Segments/Departure Date'] = pd.to_datetime(
        df_daily['Segments/Departure Date'])

    # Buat range tanggal penuh
    full_dates = pd.DataFrame({'Segments/Departure Date': pd.date_range(start=df_daily['Segments/Departure Date'].min(),
                                                                        end=df_daily['Segments/Departure Date'].max())})
    df_daily = full_dates.merge(
        df_daily, on='Segments/Departure Date', how='left')
    df_daily['Total Pax'] = df_daily['Total Pax'].fillna(0).astype(int)

    col1, col2 = st.columns([1, 2])

    with col1:
        buffer_days = st.number_input(
            "Enter the number of buffer days (0-7)", min_value=0, max_value=7, value=1)

    with col2:
        buffer_option = st.radio(
            "Buffer Day Type:",
            options=["Before", "After", "Before and After"],
            index=2,
            horizontal=True
        )

    calendar = pd.DataFrame({'Segments/Departure Date': pd.date_range(start=df_daily['Segments/Departure Date'].min(),
                                                                      end=df_daily['Segments/Departure Date'].max())})
    calendar = calendar.merge(
        df_holiday, on='Segments/Departure Date', how='left')
    calendar['holiday'] = calendar['holiday'].fillna(0).astype(int)

    calendar['grp'] = (calendar['holiday'] !=
                       calendar['holiday'].shift()).cumsum()
    calendar['seq'] = calendar.groupby('grp').cumcount()
    holiday_groups = calendar.groupby('grp').agg({
        'holiday': 'sum',
        'Segments/Departure Date': ['first', 'last']
    })
    holiday_groups.columns = ['count', 'start', 'end']
    long_holidays = holiday_groups[holiday_groups['count'] >= 3]

    buffer_dates = set()
    for _, row in long_holidays.iterrows():
        if buffer_option in ["Before", "Before and After"]:
            for i in range(1, buffer_days + 1):
                buffer_dates.add(row['start'] - pd.Timedelta(days=i))
        if buffer_option in ["After", "Before and After"]:
            for i in range(1, buffer_days + 1):
                buffer_dates.add(row['end'] + pd.Timedelta(days=i))

    calendar['holiday'] = calendar['Segments/Departure Date'].isin(
        calendar[calendar['holiday'] ==
                 1]['Segments/Departure Date'].tolist() + list(buffer_dates)
    ).astype(int)

    # Gabungkan kembali dengan df_daily
    df_daily = df_daily.drop(columns='holiday', errors='ignore')
    df_daily = df_daily.merge(
        calendar[['Segments/Departure Date', 'holiday']], on='Segments/Departure Date', how='left')

    # Pisahkan data
    pax_holiday = df_daily[df_daily['holiday'] == 1]['Total Pax']
    pax_non_holiday = df_daily[df_daily['holiday'] == 0]['Total Pax']

    total_holiday = pax_holiday.sum()
    total_non_holiday = pax_non_holiday.sum()
    mean_holiday = pax_holiday.mean()
    mean_non_holiday = pax_non_holiday.mean()

    summary_df = pd.DataFrame({
        'Kategori': ['Holiday', 'Non-Holiday'],
        'Jumlah Total Pax': [total_holiday, total_non_holiday],
        'Rata-rata Per Hari': [mean_holiday, mean_non_holiday]
    }).set_index('Kategori')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Total Number of Pax")
        st.bar_chart(summary_df[['Jumlah Total Pax']])

    with col2:
        st.subheader("Average Pax Per Day")
        st.bar_chart(summary_df[['Rata-rata Per Hari']])

    # Lakukan uji t
    t_stat, p_value = ttest_ind(pax_holiday, pax_non_holiday, equal_var=False)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        card(
            title=f"Pax Mean (Holiday)",
            text=f"{mean_holiday:.2f}",
            styles={
                "card": {
                    "width": "100%",
                    "height": "100%",
                    "border-radius": "20px",
                    "box-shadow": "0 4px 8px rgba(0,0,0,0.2)",
                    "background-color": "#ffffff",
                    "padding": "2%",
                    "margin": "2%",
                },
                "filter": {
                    "background-color": "rgba(0, 0, 0, 0)"
                },
                "title": {
                    "font-size": "20px",
                    "color": "#333333"
                },
                "text": {
                    "font-size": "32px",
                    "font-weight": "bold",
                    "color": "#2E8B57"
                }
            }
        )

    with col2:
        card(
            title=f"Pax Mean (Non-Holiday)",
            text=f"{mean_non_holiday:.2f}",
            styles={
                "card": {
                    "width": "100%",
                    "height": "100%",
                    "border-radius": "20px",
                    "box-shadow": "0 4px 8px rgba(0,0,0,0.2)",
                    "background-color": "#ffffff",
                    "padding": "2%",
                    "margin": "2%",
                },
                "filter": {
                    "background-color": "rgba(0, 0, 0, 0)"
                },
                "title": {
                    "font-size": "20px",
                    "color": "#333333"
                },
                "text": {
                    "font-size": "32px",
                    "font-weight": "bold",
                    "color": "#2E8B57"
                }
            }
        )

    with col3:
        card(
            title=f"P-Value",
            text=f"{p_value:.1e}",
            styles={
                "card": {
                    "width": "100%",
                    "height": "100%",
                    "border-radius": "20px",
                    "box-shadow": "0 4px 8px rgba(0,0,0,0.2)",
                    "background-color": "#ffffff",
                    "padding": "2%",
                    "margin": "2%",
                },
                "filter": {
                    "background-color": "rgba(0, 0, 0, 0)"
                },
                "title": {
                    "font-size": "20px",
                    "color": "#333333"
                },
                "text": {
                    "font-size": "32px",
                    "font-weight": "bold",
                    "color": "#2E8B57"
                }
            }
        )

    with col4:
        card(
            title=f"T-Statistic",
            text=f"{t_stat:.5}",
            styles={
                "card": {
                    "width": "100%",
                    "height": "100%",
                    "border-radius": "20px",
                    "box-shadow": "0 4px 8px rgba(0,0,0,0.2)",
                    "background-color": "#ffffff",
                    "padding": "2%",
                    "margin": "2%",
                },
                "filter": {
                    "background-color": "rgba(0, 0, 0, 0)"
                },
                "title": {
                    "font-size": "20px",
                    "color": "#333333"
                },
                "text": {
                    "font-size": "32px",
                    "font-weight": "bold",
                    "color": "#2E8B57"
                }
            }
        )

    if p_value < 0.05:
        card(
            title='Conclusion:',
            text="There is a significant difference in Total Pax between holidays and non-holidays.",
            styles={
                "card": {
                    "width": "100%",
                    "height": "100%",
                    "border-radius": "20px",
                    "box-shadow": "0 4px 8px rgba(0,0,0,0.2)",
                    "background-color": "#ffffff",
                    "padding": "2%",
                    "margin": "2%",
                },
                "filter": {
                    "background-color": "rgba(0, 0, 0, 0)"
                },
                "title": {
                    "font-size": "20px",
                    "color": "#333333"
                },
                "text": {
                    "font-size": "32px",
                    "font-weight": "bold",
                    "color": "#2E8B57"
                }
            }
        )
    else:
        card(
            title='Conclusion:',
            text="There is NO a significant difference in Total Pax between holidays and non-holidays.",
            styles={
                "card": {
                    "width": "100%",
                    "height": "100%",
                    "border-radius": "20px",
                    "box-shadow": "0 4px 8px rgba(0,0,0,0.2)",
                    "background-color": "#ffffff",
                    "padding": "2%",
                    "margin": "2%",
                },
                "filter": {
                    "background-color": "rgba(0, 0, 0, 0)"
                },
                "title": {
                    "font-size": "20px",
                    "color": "#333333"
                },
                "text": {
                    "font-size": "32px",
                    "font-weight": "bold",
                    "color": "#2E8B57"
                }
            }
        )

    st.write('*Notes:*')
    st.markdown("### 📌 T-statistic")
    st.markdown("""
    - **T-statistic** measures how large the difference between two means is relative to the variation in the data.  
    - A larger value indicates a more significant difference.
        + If the T-statistic is **negative**, the mean of the first group is lower.
        + If the T-statistic is **positive**, the mean of the second group is lower.
    """)

    st.markdown("### 📌 P-value")
    st.markdown("""
    - **P-value** indicates the strength of evidence against the null hypothesis.  
        + If p-value < 0.05, the difference is considered **statistically significant**.  
        + If p-value ≥ 0.05, the difference is considered **not significant**.  
        + If the p-value is very small, the difference is most likely not due to chance.
    """)

    st.markdown("### 📌 Buffer Day")
    st.markdown("""
    - **Buffer Day** is an additional day added before and/or after a long holiday period (consecutive holidays ≥ 3 days).  
    - It is used to capture fluctuations in passenger volume around long holiday periods.
    """)
