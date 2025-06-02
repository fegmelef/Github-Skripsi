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
    def filterYearOnly(prefix: str, df: pd.DataFrame):
        if f"{prefix}_reset" not in st.session_state:
            st.session_state.clear()
            st.session_state[f"{prefix}_reset"] = True

        df_depart = df[df['Segments/Departure Date'].notna()].copy()
        df_depart['Depart Year'] = df_depart['Segments/Departure Date'].dt.year
        available_years = sorted(df_depart['Depart Year'].unique())

        all_years_key = f"{prefix}_all_years"
        selected_years_key = f"{prefix}_selected_years"

        if all_years_key not in st.session_state:
            st.session_state[all_years_key] = True
            st.session_state[selected_years_key] = available_years

        def year_check_change():
            if st.session_state[all_years_key]:
                st.session_state[selected_years_key] = available_years
            else:
                st.session_state[selected_years_key] = []

        def year_multi_change():
            st.session_state[all_years_key] = (
                len(st.session_state[selected_years_key]) == len(
                    available_years)
            )

        # col1, _ = st.columns([1, 3])

        # with col1:
        st.checkbox("All Years", key=all_years_key,
                    on_change=year_check_change)
        selected_years = st.multiselect(
            "Select Years (Departure Date)",
            options=available_years,
            key=selected_years_key,
            on_change=year_multi_change
        )

        if not st.session_state[all_years_key] and len(selected_years) == 0:
            st.error("Please select at least one year.")

        if len(selected_years) > 0:
            filtered_df = df_depart[df_depart['Depart Year'].isin(
                selected_years)]
        else:
            filtered_df = df

        return filtered_df

    df = filter_year_month_depart("ttest", df)

    df_holiday = pd.read_excel('files/holidays_ID.xlsx')

    df_holiday['Date'] = pd.to_datetime(df_holiday['Date'])
    df_holiday = df_holiday.rename(columns={
        'Date': 'Segments/Departure Date',
        'Libur': 'holiday',
        'Holiday Name': 'holiday_name'
    })

    df['Segments/Departure Date'] = pd.to_datetime(
        df['Segments/Departure Date'])
    df['Booking Date'] = pd.to_datetime(df['Booking Date'])

    dftest = df.dropna(subset=['Segments/Departure Date'])
    # dftest = dftest.fillna(method='ffill')
    
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    with col2:
        buffer_days = st.number_input(
            "Enter the number of buffer days (0-7)", min_value=0, max_value=7, value=1)
    with col3:
        buffer_option = st.radio(
            "Buffer Day Type:",
            options=["Before", "After", "Before and After"],
            index=2,
            horizontal=True
        )
    with col1:
        holiday_filter = st.selectbox(
            "Select Holiday Comparison Type:",
            options=[
                "All Holidays (including weekend)",
                "Long Holiday (more than 3 days)",
                "Weekend only",
                "Holiday only"
            ],
            index=0
        )
    with col4:
        customer_type = st.selectbox('Customer Type', ['All', 'Individual', 'Corporate'])

        if customer_type == 'Individual':
            filtered_df = dftest[dftest['Customer/Display Name'].str.upper() == 'RODEX DARMO FPO']
        elif customer_type == 'Corporate':
            filtered_df = dftest[dftest['Customer/Display Name'].str.upper() != 'RODEX DARMO FPO']
        else:
            filtered_df = dftest.copy()

        df_daily = filtered_df.groupby(
            filtered_df['Segments/Departure Date'].dt.date)['Total Pax'].sum().reset_index()
        df_daily['Segments/Departure Date'] = pd.to_datetime(df_daily['Segments/Departure Date'])

    full_dates = pd.DataFrame({'Segments/Departure Date': pd.date_range(
        start=df_daily['Segments/Departure Date'].min(),
        end=df_daily['Segments/Departure Date'].max()
    )})
    df_daily = full_dates.merge(
        df_daily, on='Segments/Departure Date', how='left')
    df_daily['Total Pax'] = df_daily['Total Pax'].fillna(0).astype(int)
    df_daily = df_daily[df_daily['Total Pax'] > 0]        

    calendar = pd.DataFrame({'Segments/Departure Date': pd.date_range(
        start=df_daily['Segments/Departure Date'].min(),
        end=df_daily['Segments/Departure Date'].max()
    )})
    calendar = calendar.merge(df_holiday[['Segments/Departure Date', 'holiday', 'holiday_name']],
                              on='Segments/Departure Date', how='left')

    calendar['holiday'] = calendar['holiday'].fillna(0).astype(int)
    calendar['holiday_name'] = calendar['holiday_name'].fillna('')

    calendar['is_weekend'] = (calendar['holiday_name']
                              == 'Weekend').astype(int)
    calendar['is_holiday'] = ((calendar['holiday_name'] != '') & (
        calendar['holiday_name'] != 'Weekend')).astype(int)
    calendar['is_any_holiday'] = (calendar['holiday_name'] != '').astype(
        int)  # Semua holiday + weekend

    calendar['grp'] = (calendar['is_any_holiday'] !=
                       calendar['is_any_holiday'].shift()).cumsum()
    calendar['seq'] = calendar.groupby('grp').cumcount()

    holiday_groups = calendar.groupby('grp').agg({
        'is_any_holiday': 'sum',
        'Segments/Departure Date': ['first', 'last']
    })
    holiday_groups.columns = ['count', 'start', 'end']
    long_holidays = holiday_groups[holiday_groups['count'] >= 3]

    buffer_dates = set()
    for _, row in long_holidays.iterrows():
        for i in range(1, buffer_days + 1):
            if buffer_option in ["Before", "Before and After"]:
                buffer_dates.add(row['start'] - pd.Timedelta(days=i))
            if buffer_option in ["After", "Before and After"]:
                buffer_dates.add(row['end'] + pd.Timedelta(days=i))

    calendar['is_buffer'] = calendar['Segments/Departure Date'].isin(
        buffer_dates).astype(int)
    calendar['is_holiday_with_buffer'] = (
        (calendar['is_any_holiday'] == 1) | (calendar['is_buffer'] == 1)).astype(int)

    df_daily = df_daily.drop(
        columns=['is_weekend', 'is_holiday', 'is_holiday_with_buffer'], errors='ignore')
    df_daily = df_daily.merge(
        calendar[['Segments/Departure Date', 'holiday_name',
                  'is_weekend', 'is_holiday', 'is_holiday_with_buffer']],
        on='Segments/Departure Date', how='left'
    )

    long_holiday_dates = pd.concat([
        pd.Series(pd.date_range(start=row['start'], end=row['end'])) for _, row in long_holidays.iterrows()
    ])

    df_daily['is_long_holiday_with_buffer'] = df_daily['Segments/Departure Date'].isin(
        long_holiday_dates).astype(int)
    df_daily.loc[df_daily['Segments/Departure Date'].isin(
        buffer_dates), 'is_long_holiday_with_buffer'] = 1

    df_daily['is_holiday_only_with_buffer'] = (
        (df_daily['is_holiday'] == 1) & (df_daily['is_weekend'] == 0)).astype(int)
    df_daily.loc[df_daily['Segments/Departure Date'].isin(
        buffer_dates), 'is_holiday_only_with_buffer'] = 1

    if holiday_filter == "All Holidays (including weekend)":
        pax_holiday = df_daily[
            (df_daily['is_holiday_with_buffer'] == 1) | (
                df_daily['is_weekend'] == 1)
        ]['Total Pax']
        pax_non_holiday = df_daily[
            (df_daily['is_holiday_with_buffer'] == 0) & (
                df_daily['is_weekend'] == 0)
        ]['Total Pax']

    elif holiday_filter == "Long Holiday (more than 3 days)":
        pax_holiday = df_daily[df_daily['is_long_holiday_with_buffer']
                               == 1]['Total Pax']
        pax_non_holiday = df_daily[df_daily['is_long_holiday_with_buffer']
                                   == 0]['Total Pax']

    elif holiday_filter == "Weekend only":
        pax_holiday = df_daily[
            (df_daily['is_weekend'] == 1) & (df_daily['is_holiday'] == 0)
        ]['Total Pax']
        pax_non_holiday = df_daily[
            (df_daily['is_weekend'] == 0) & (df_daily['is_holiday'] == 0)
        ]['Total Pax']

    elif holiday_filter == "Holiday only":
        pax_holiday = df_daily[df_daily['is_holiday_only_with_buffer']
                               == 1]['Total Pax']
        pax_non_holiday = df_daily[df_daily['is_holiday_only_with_buffer']
                                   == 0]['Total Pax']

    else:
        pax_holiday = pd.Series(dtype='float64')
        pax_non_holiday = pd.Series(dtype='float64')

    total_holiday = pax_holiday.sum()
    total_non_holiday = pax_non_holiday.sum()
    mean_holiday = pax_holiday.mean()
    mean_non_holiday = pax_non_holiday.mean()

    label_map = {
        "All Holidays (including weekend)": ("Holiday & Weekend", "Non-Holiday & Non-Weekend"),
        "Long Holiday (more than 3 days)": ("Long Holiday", "Non-long Holiday"),
        "Weekend only": ("Weekend", "Non-Weekend"),
        "Holiday only": ("Holiday", "Non-Holiday")
    }

    holiday_label, non_holiday_label = label_map.get(
        holiday_filter, ("Holiday", "Non-Holiday"))

    summary_df = pd.DataFrame({
        'Category': [holiday_label, non_holiday_label],
        'Total Pax': [total_holiday, total_non_holiday],
        'Average per Day': [mean_holiday, mean_non_holiday]
    }).set_index('Category')

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Total Number of Pax")
        st.bar_chart(summary_df[['Total Pax']])
    with col2:
        st.subheader("Average Pax Per Day")
        st.bar_chart(summary_df[['Average per Day']])

    t_stat, p_value = ttest_ind(pax_holiday, pax_non_holiday, equal_var=False)

    col1, col2, col3, col4 = st.columns(4)

    label_map = {
        "All Holidays (including weekend)": ("Holiday & Weekend", "Non-Holiday & Non-Weekend"),
        "Long Holiday (more than 3 days)": ("Long Holiday", "Non-long Holiday"),
        "Weekend only": ("Weekend", "Non-Weekend"),
        "Holiday only": ("Holiday", "Non-Holiday")
    }

    holiday_label, non_holiday_label = label_map.get(
        holiday_filter, ("Holiday", "Non-Holiday"))

    with col1:
        card(
            title=f"Pax Mean ({holiday_label})",
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
            title=f"Pax Mean ({non_holiday_label})",
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

    st.write('*Glossary:*')
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
