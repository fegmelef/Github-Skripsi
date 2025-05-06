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


def render_sales_summary(df, start_year, end_year):
    
    # region title & df Sales
    col1, col2 = st.columns([9, 1])

    with col1:
        st.title(f"Sales Summary ({start_year} - {end_year})")

    with col2:
        st.button('Refresh')

    filtered_df = filter_year_month("sales", df)
    # endregion

    # region card sales
    col1, col2, col3 = st.columns(3)

    with col1:
        card(
            title=f"# of Pax",
            text=f"{int(filtered_df['Total Pax'].sum()):,} Pax",
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
            title=f"# of Order",
            text=f"{filtered_df[filtered_df['Issued Date'].notna()]['Issued Date'].count():,} Orders",
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

    # order_per_month = (
    #     filtered_df
    #     .groupby(['Issued Year', 'Issued Month'])
    #     .size()
    #     .reset_index(name='Order Count')
    # )

    # most_order_row = order_per_month.loc[order_per_month['Order Count'].idxmax(
    # )]
    # most_order_count = most_order_row['Order Count']
    # most_order_month = calendar.month_name[int(most_order_row['Issued Month'])]
    # most_order_year = int(most_order_row['Issued Year'])

    grand_total_sales = filtered_df['Grand Total'].sum()

    def format_number(number):
        if number >= 1_000_000_000:
            return f"{number/1_000_000_000:.1f}B"
        elif number >= 1_000_000:
            return f"{number/1_000_000:.1f}M"
        elif number >= 1_000:
            return f"{number/1_000:.1f}K"
        else:
            return str(int(number))

    with col3:
        card(
            title="Total Sales",
            text=f"IDR {format_number(grand_total_sales)}",
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

    # endregion card sales
    
    # region chart sales
    chart_option = st.radio(
        "Filter:",
        # options=["Sales", "Income", "YoY Growth Sales", "YoY Growth Income", "Sales with YoY Growth", "Income with YoY Growth"],
        options=["Sales", "Income", "YoY Growth Sales", "YoY Growth Income"],
        key="chart_option",
        horizontal=True
    )
    col1, col2 = st.columns(2)

    # Hitung Order Count per bulan
    monthly_orders = (
        filtered_df
        .groupby(['Issued Year', 'Issued Month'])
        .size()
        .reset_index(name='Order Count')
    )

    # Hitung Income (Grand Total) per bulan
    monthly_income = (
        filtered_df
        .groupby(['Issued Year', 'Issued Month'])['Grand Total']
        .sum()
        .reset_index(name='Total Income')
    )

    # Pivot untuk Order Count
    sales_pivot = monthly_orders.pivot(
        index='Issued Month',
        columns='Issued Year',
        values='Order Count'
    ).fillna(0).sort_index()

    # Pivot untuk Income
    income_pivot = monthly_income.pivot(
        index='Issued Month',
        columns='Issued Year',
        values='Total Income'
    ).fillna(0).sort_index()

    # Copy untuk hitung YoY Sales
    pivot_orders = sales_pivot.copy()

    # Hitung YoY Growth untuk Order
    if len(pivot_orders.columns) > 1:
        for year in pivot_orders.columns[1:]:
            previous_year = year - 1
            if previous_year in pivot_orders.columns:
                pivot_orders[f'YoY Growth Sales {year}'] = (
                    (pivot_orders[year] - pivot_orders[previous_year]
                     ) / pivot_orders[previous_year] * 100
                )

    # Copy untuk hitung YoY Income
    pivot_income = income_pivot.copy()

    # Hitung YoY Growth untuk Income
    if len(pivot_income.columns) > 1:
        for year in pivot_income.columns[1:]:
            previous_year = year - 1
            if previous_year in pivot_income.columns:
                pivot_income[f'YoY Growth Income {year}'] = (
                    (pivot_income[year] - pivot_income[previous_year]
                     ) / pivot_income[previous_year] * 100
                )

    with col1:
        if chart_option == "Sales":
            st.write("### Sales per Year")
            st.line_chart(sales_pivot)

        elif chart_option == "Income":
            st.write("### Total Income per Year")
            st.line_chart(income_pivot)

        elif chart_option == "YoY Growth Sales":
            yoy_growth_sales = pivot_orders.filter(regex='YoY Growth Sales')
            st.write("### Year on Year Growth Sales (in %)")
            st.line_chart(yoy_growth_sales)

        elif chart_option == "YoY Growth Income":
            yoy_growth_income = pivot_income.filter(regex='YoY Growth Income')
            st.write("### Year on Year Growth Income (in %)")
            st.line_chart(yoy_growth_income)

        elif chart_option == "Sales with YoY Growth":
            st.write("### Sales per Year with YoY Growth (in %)")
            st.line_chart(pivot_orders)

        else:  # Income with YoY Growth
            st.write("### Income per Year with YoY Growth (in %)")
            st.line_chart(pivot_income)

    with col2:
        st.markdown(f"### Order Trend")
        order_chart_line = (
            filtered_df
            .groupby(pd.Grouper(key='Issued Date', freq='M'))
            .size()
            .reset_index(name='Order Count')
        )

        order_chart_line['Date Ordinal'] = order_chart_line['Issued Date'].apply(
            lambda x: x.toordinal())

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            order_chart_line['Date Ordinal'], order_chart_line['Order Count'])

        order_chart_line['Trendline'] = slope * \
            order_chart_line['Date Ordinal'] + intercept

        st.line_chart(order_chart_line.set_index(
            'Issued Date')[['Order Count', 'Trendline']])

    # endregion chart sales
