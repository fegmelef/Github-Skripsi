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


def render_passenger_summary(df, start_year, end_year):
    # region title & df Passenger
    col1, col2 = st.columns([9, 1])

    with col1:
        st.title(f"Passengers Summary ({start_year} - {end_year})")

    with col2:
        st.button('Refresh')

    filtered_df = filter_year_month_depart("passenger", df)
    
    option_customer = st.radio(
        "Filter:",
        # options=["Customer", "Passenger", "Customer Order History", "Passenger Order History"],
        options=["Customer", "Customer Order History", 'Age Category'],
        key="option_customer",
        horizontal=True
    )
    # endregion titlee passenger

    # region top customer
    top_corporate = (
        filtered_df
        .assign(**{'Customer/Display Name': filtered_df['Customer/Display Name'].str.upper()})
        .groupby('Customer/Display Name')['Total Pax']
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    
    if option_customer == 'Customer':
        st.dataframe(top_corporate)

    # elif option_customer == 'Passenger':
    #     st.dataframe(top_passenger)

    elif option_customer == 'Customer Order History':
        selected_customer = st.selectbox(
            "Pilih Customer:",
            top_corporate['Customer/Display Name']
        )

        customer_history = (
            filtered_df[filtered_df['Customer/Display Name'].str.upper()
                        == selected_customer]
            .sort_values(by='Booking Date', ascending=False)
            [[
                'Customer/Display Name',
                'Booking Date',
                'Segments/Departure Date',
                'Segments/Arrival Date',
                'Sector',
                'Segments/Origin/Code',
                'Segments/Destination/Code',
            ]]
        )

        st.dataframe(customer_history)
        
    # endregion top customer

    # region age (adult child, infant)
    elif option_customer == 'Age Category':

        def get_generation(birth_year):
            if 1901 <= birth_year <= 1924:
                return 'The Greatest Generation (1901–1924)'
            elif 1925 <= birth_year <= 1945:
                return 'The Silent Generation (1925–1945)'
            elif 1946 <= birth_year <= 1964:
                return 'The Baby Boomer Generation (1946–1964)'
            elif 1965 <= birth_year <= 1979:
                return 'Generation X (1965–1979)'
            elif 1980 <= birth_year <= 1994:
                return 'Millennials (1980–1994)'
            elif 1995 <= birth_year <= 2012:
                return 'Generation Z (1995–2012)'
            elif 2013 <= birth_year:
                return 'Gen Alpha (>= 2013)'
            else:
                return 'Unknown'

        option_customerAge = st.radio(
            "Filter:",
            # options=["Customer", "Passenger", "Customer Order History", "Passenger Order History"],
            options=["Pie Chart", "Line Chart"],
            key="option_customerAge",
            horizontal=True
        )

        df_bday = df.copy()

        df_bday['Segments/Departure Date'] = df_bday['Segments/Departure Date'].fillna(
            method='ffill')
        df_bday = df_bday[df_bday['Passenger/Birth Date'].notna() &
                          (df_bday['Passenger/Birth Date'] <= df_bday['Segments/Departure Date'])]

        df_bday['Passenger/Birth Date'] = pd.to_datetime(
            df_bday['Passenger/Birth Date'])
        df_bday['Segments/Departure Date'] = pd.to_datetime(
            df_bday['Segments/Departure Date'])

        selected_years = filtered_df['Segments/Departure Date'].dt.year.unique()
        selected_months = filtered_df['Segments/Departure Date'].dt.month.unique()
        df_bday = df_bday[
            df_bday['Segments/Departure Date'].dt.year.isin(selected_years) &
            df_bday['Segments/Departure Date'].dt.month.isin(selected_months)
        ]

        df_bday['Birth Year'] = df_bday['Passenger/Birth Date'].dt.year
        df_bday['Age'] = (df_bday['Segments/Departure Date'] -
                          df_bday['Passenger/Birth Date']).dt.days // 365

        df_bday['Generation'] = df_bday['Birth Year'].apply(
            get_generation)

        generation_counts = df_bday['Generation'].value_counts()

        # region pie age
        
        if option_customerAge == 'Pie Chart':
            col1, col2 = st.columns([3, 7])

            with col1:
                labels = ['Adult', 'Child', 'Infant']
                sizes = [filtered_df['Adult'].sum(), filtered_df['Child'].sum(),
                         filtered_df['Infant'].sum()]

                fig = go.Figure(data=[go.Pie(labels=labels, values=sizes, hole=0.3,
                                             textinfo='percent+label', marker=dict(colors=['#ff9999', '#66b3ff', '#99ff99']))])

                fig.update_layout(
                    width=400, height=500)

                st.plotly_chart(fig)

            with col2:
                fig = go.Figure(data=[go.Pie(labels=generation_counts.index, values=generation_counts.values, hole=0.3,
                                             textinfo='percent+label', marker=dict(colors=['#ff9999', '#66b3ff', '#99ff99',
                                                                                           '#ffcc99', '#c2c2f0', '#ffb3e6', '#c2f0c2']))])

                fig.update_layout(
                    width=400, height=500)

                st.plotly_chart(fig)
                
        # endregion pie age

        # region line age
        elif option_customerAge == 'Line Chart':

            col1, col2 = st.columns([3, 7])

            with col1:
                filtered_df['Segments/Departure Date'] = pd.to_datetime(
                    filtered_df['Segments/Departure Date'])
                filtered_df['Departure Period'] = filtered_df['Segments/Departure Date'].dt.to_period(
                    'M').astype(str)

                grouped = filtered_df.groupby('Departure Period')[
                    ['Adult', 'Child', 'Infant']].sum().fillna(0)
                grouped = grouped.reset_index()

                fig = go.Figure()
                for col in ['Adult', 'Child', 'Infant']:
                    fig.add_trace(go.Scatter(
                        x=grouped['Departure Period'], y=grouped[col], mode='lines+markers', name=col))

                fig.update_layout(
                    height=500,
                    legend=dict(
                        orientation='h',
                        y=-0.3,
                        x=0.5,
                        xanchor='center',
                        yanchor='top'
                    ),
                    margin=dict(b=100)
                )

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                df_bday['Departure Period'] = df_bday['Segments/Departure Date'].dt.to_period(
                    'M').astype(str)
                grouped = df_bday.groupby(
                    ['Departure Period', 'Generation']).size().reset_index(name='Count')

                fig = px.line(grouped, x='Departure Period',
                              y='Count', color='Generation', markers=True)

                fig.update_layout(
                    height=500,
                    legend=dict(
                        orientation='v',
                        y=0.5,
                        x=1.02,
                        xanchor='left',
                        yanchor='middle'
                    ),
                    margin=dict(r=120)
                )

                st.plotly_chart(fig, use_container_width=True)
                
        # endregion line age
                
    # endregion age (adult child, infant)
