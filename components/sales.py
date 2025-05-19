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
from plotly.subplots import make_subplots


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
        options=["Pax Volume", "Sales Volume", "Sales Value",
                 "YoY Growth (Pax Volume)", "YoY Growth (Sales Volume)", "YoY Growth (Sales Value)"],
        key="chart_option",
        horizontal=True
    )
    col1, col2 = st.columns(2)

    monthly_orders = (
        filtered_df
        .groupby(['Issued Year', 'Issued Month'])
        .size()
        .reset_index(name='Order Count')
    )

    monthly_income = (
        filtered_df
        .groupby(['Issued Year', 'Issued Month'])['Grand Total']
        .sum()
        .reset_index(name='Sales Value')
    )

    sales_pivot = monthly_orders.pivot(
        index='Issued Month',
        columns='Issued Year',
        values='Order Count'
    ).fillna(0).sort_index()

    income_pivot = monthly_income.pivot(
        index='Issued Month',
        columns='Issued Year',
        values='Sales Value'
    ).fillna(0).sort_index()

    pivot_orders = sales_pivot.copy()

    if len(pivot_orders.columns) > 1:
        for year in pivot_orders.columns[1:]:
            previous_year = year - 1
            if previous_year in pivot_orders.columns:
                pivot_orders[f'YoY Growth Sales {year}'] = (
                    (pivot_orders[year] - pivot_orders[previous_year]
                     ) / pivot_orders[previous_year] * 100
                )

    pivot_income = income_pivot.copy()

    if len(pivot_income.columns) > 1:
        for year in pivot_income.columns[1:]:
            previous_year = year - 1
            if previous_year in pivot_income.columns:
                pivot_income[f'YoY Growth Income {year}'] = (
                    (pivot_income[year] - pivot_income[previous_year]
                     ) / pivot_income[previous_year] * 100
                )

    monthly_pax = (
        filtered_df
        .groupby(['Issued Year', 'Issued Month'])['Total Pax']
        .sum()
        .reset_index(name='Total Pax')
    )

    pax_pivot = monthly_pax.pivot(
        index='Issued Month',
        columns='Issued Year',
        values='Total Pax'
    ).fillna(0).sort_index()

    pivot_pax = pax_pivot.copy()

    if len(pivot_pax.columns) > 1:
        for year in pivot_pax.columns[1:]:
            previous_year = year - 1
            if previous_year in pivot_pax.columns:
                pivot_pax[f'YoY Growth Pax {year}'] = (
                    (pivot_pax[year] - pivot_pax[previous_year]
                     ) / pivot_pax[previous_year] * 100
                )

    with col1:
        if chart_option == "Sales Volume":
            st.write("### Monthly Sales Volume by Year")
            sales_melted = sales_pivot.reset_index().melt(
                id_vars='Issued Month', var_name='Year', value_name='Orders')
            fig = px.line(sales_melted, x='Issued Month', y='Orders', color='Year',
                          labels={'Issued Month': 'Month', 'Orders': 'Order Count'})
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_option == "YoY Growth (Sales Volume)":
            st.write("### Year on Year Growth of Sales Volume (in %)")
            yoy_sales_melted = pivot_orders.filter(regex='YoY Growth Sales').reset_index().melt(
                id_vars='Issued Month', var_name='Year', value_name='Growth')
            fig = px.line(yoy_sales_melted, x='Issued Month', y='Growth', color='Year',
                          labels={'Issued Month': 'Month', 'Growth': 'Growth (%)'})
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_option == "Sales Value":
            st.write("### Monthly Sales Value by Year")
            income_melted = income_pivot.reset_index().melt(
                id_vars='Issued Month', var_name='Year', value_name='Income')
            fig = px.line(income_melted, x='Issued Month', y='Income', color='Year',
                          labels={'Issued Month': 'Month', 'Income': 'Sales Value'})
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_option == "YoY Growth (Sales Value)":
            st.write("### Year on Year Growth of Sales Value (in %)")
            yoy_income_melted = pivot_income.filter(regex='YoY Growth Income').reset_index().melt(
                id_vars='Issued Month', var_name='Year', value_name='Growth')
            fig = px.line(yoy_income_melted, x='Issued Month', y='Growth', color='Year',
                          labels={'Issued Month': 'Month', 'Growth': 'Growth (%)'})
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_option == "Pax Volume":
            st.write("### Monthly Pax Volume by Year")
            pax_melted = pax_pivot.reset_index().melt(
                id_vars='Issued Month', var_name='Year', value_name='Pax')
            fig = px.line(pax_melted, x='Issued Month', y='Pax', color='Year',
                          labels={'Issued Month': 'Month', 'Pax': 'Total Passengers'})
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_option == "YoY Growth (Pax Volume)":
            st.write("### Year on Year Growth of Pax Volume (in %)")
            yoy_pax_melted = pivot_pax.filter(regex='YoY Growth Pax').reset_index().melt(
                id_vars='Issued Month', var_name='Year', value_name='Growth')
            fig = px.line(yoy_pax_melted, x='Issued Month', y='Growth', color='Year',
                          labels={'Issued Month': 'Month', 'Growth': 'Growth (%)'})
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5
                )
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Common trend analysis function
        def plot_trend(data, date_col, value_col, yaxis_title):
            fig = go.Figure()

            # Add actual data
            fig.add_trace(
                go.Scatter(x=data[date_col], y=data[value_col],
                           name='Actual', mode='lines+markers')
            )

            # Add trendline
            z = np.polyfit(data[date_col].astype(np.int64) // 10**9,
                           data[value_col], 1)
            p = np.poly1d(z)
            fig.add_trace(
                go.Scatter(x=data[date_col], y=p(data[date_col].astype(np.int64) // 10**9),
                           name='Trendline', line=dict(dash='dash'))
            )

            # Calculate stats
            slope = z[0] * (60*60*24*365)  # Convert to yearly slope
            trend_dir = "↑ Increasing" if slope > 0 else "↓ Decreasing"
            r_squared = np.corrcoef(data[date_col].astype(np.int64) // 10**9,
                                    data[value_col])[0, 1]**2

            # Update layout
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title=yaxis_title,
                showlegend=True,
                legend=dict(
                            orientation="h",  # Horizontal orientation
                            yanchor="bottom",  # Anchor to bottom
                            y=-0.3,          # Position below chart
                            xanchor="center",  # Center horizontally
                            x=0.5
                )
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Trend: {trend_dir} | R-squared: {r_squared:.2f}")

        if chart_option in ["Sales Volume", "YoY Growth (Sales Volume)"]:
            st.write("### Monthly Sales Volume Trend")
            data = (
                filtered_df
                .groupby(pd.Grouper(key='Issued Date', freq='M'))
                .size()
                .reset_index(name='Order Count')
            )
            plot_trend(data, 'Issued Date', 'Order Count', 'Order Count')

        elif chart_option in ["Sales Value", "YoY Growth (Sales Value)"]:
            st.write("### Monthly Sales Value Trend")
            data = (
                filtered_df
                .groupby(pd.Grouper(key='Issued Date', freq='M'))
                ['Grand Total']
                .sum()
                .reset_index(name='Grand Total')
            )
            plot_trend(data, 'Issued Date', 'Grand Total', 'Sales Value')

        elif chart_option in ["Pax Volume", "YoY Growth (Pax Volume)"]:
            st.write("### Monthly Pax Volume Trend")
            data = (
                filtered_df
                .groupby(pd.Grouper(key='Issued Date', freq='M'))
                ['Total Pax']
                .sum()
                .reset_index(name='Pax Volume')
            )
            plot_trend(data, 'Issued Date', 'Pax Volume', 'Total Passengers')
    # endregion chart sales

    st.write('*Glossary:*')
    st.markdown("""
        - **Pax Volume**  
        Total number of passengers transported over a specific period.  

        - **Sales Volume**  
        Total number of Orders during over a specific period.  

        - **Sales Value**  
        Total monetary value of all sales in a specific timeframe.  

        - **YoY Growth (Pax Volume)**  
        Year-on-Year percentage change in total passengers.  

        - **YoY Growth (Sales Volume)**  
        Year-on-Year percentage change in units sold.  

        - **YoY Growth (Sales Value)**  
        Year-on-Year percentage change in total sales revenue.  
        """)


    st.markdown("""
        - **Monthly Pax Volume Trend**  
        Shows how the number of passengers changes month over month.  

        - **Monthly Sales Volume Trend**  
        Shows how the number of units sold changes month over month.  

        - **Monthly Sales Value Trend**  
        Shows how the total monetary value of sales changes month over month.  
    """)
