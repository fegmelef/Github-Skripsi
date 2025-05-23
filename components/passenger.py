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
    # filtered_df = filtered_df.ffill()

    option_customer = st.radio(
        "Filter:",
        # options=["Customer", "Passenger", "Customer Order History", "Passenger Order History"],
        options=["Customer", 'Customer Over Time',
                 'Age Category'],
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
            "Choose Customer:",
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

    # region customer over time
    elif option_customer == 'Customer Over Time':
        filterOverTime = filtered_df.copy()
        customer_order_df = (
            filterOverTime
            .assign(**{'Customer/Display Name': filterOverTime['Customer/Display Name'].str.upper()})
            .groupby('Customer/Display Name')['Total Pax']
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        customer_options = [
            (f"{row['Customer/Display Name']} ({int(row['Total Pax']):,} Pax)",
             row['Customer/Display Name'])
            for _, row in customer_order_df.iterrows()
        ]

        selected_customer = st.selectbox(
            "Choose Customer",
            options=customer_options,
            format_func=lambda x: x[0]
        )[1]

        # selected_name = st.selectbox("Choose Customer", selected_customer)

        filterOverTime['Segments/Departure Date'] = pd.to_datetime(
            filterOverTime['Segments/Departure Date'])

        df_selected = filterOverTime[filterOverTime['Customer/Display Name'].str.upper()
                                     == selected_customer]

        df_selected['YearMonth'] = df_selected['Segments/Departure Date'].dt.to_period(
            'M').astype(str)

        monthly_pax = df_selected.groupby(
            'YearMonth')['Total Pax'].sum().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_pax['YearMonth'], y=monthly_pax['Total Pax'], mode='lines+markers'))

        fig.update_layout(title=f"Total Pax Over Time - {selected_customer}",
                          xaxis_title='Month',
                          yaxis_title='Total Pax',
                          xaxis_tickangle=-45)

        st.plotly_chart(fig)
    # endregion customer over time

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
                return 'Generation Alpha (>= 2013)'
            else:
                return 'Unknown'

        def get_age(ageCount):
            if ageCount <= 20:
                return 'Age 0-20'
            elif 21 <= ageCount <= 35:
                return 'Age 21-35'
            elif 36 <= ageCount <= 50:
                return 'Age 36-50'
            elif 51 <= ageCount:
                return 'Age 50+'
            else:
                return 'Unknown'

        option_customerAge = st.radio(
            "Filter:",
            options=["Pie Chart", "Line Chart"],
            key="option_customerAge",
            horizontal=True
        )

        df_bday = df.copy()

        # Preprocessing
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
        df_all = df_bday[
            df_bday['Segments/Departure Date'].dt.year.isin(selected_years) &
            df_bday['Segments/Departure Date'].dt.month.isin(selected_months)
        ]

        df_all['Birth Year'] = df_all['Passenger/Birth Date'].dt.year
        df_all['Age'] = (df_all['Segments/Departure Date'] -
                         df_all['Passenger/Birth Date']).dt.days // 365

        df_all['Generation'] = df_all['Birth Year'].apply(get_generation)
        df_all['AgeCategory'] = df_all['Age'].apply(get_age)

        # Get counts for pie chart
        gen_all = df_all['Generation'].value_counts()
        age_all = df_all['AgeCategory'].value_counts()

        if 'last_option' not in st.session_state:
            st.session_state.last_option = None

        # option_customerAge = st.selectbox("Select Option", ['Pie Chart', 'Other'])

        if option_customerAge != st.session_state.last_option:
            for key in ['selected_years', 'selected_chart']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.last_option = option_customerAge

        if option_customerAge == 'Pie Chart':
            available_years = sorted(
                df_bday['Segments/Departure Date'].dt.year.unique())

            col1, col2 = st.columns(2)

            with col1:
                selected_years = st.multiselect(
                    "Select Year(s) to show yearly chart",
                    available_years,
                    default=[available_years[-1]],
                    key='selected_years'
                )

            with col2:
                selected_chart = st.selectbox(
                    "Select chart to display",
                    ['Adult/Child/Infant Distribution',
                        'Generation Distribution', 'Age Distribution'],
                    key='selected_chart'
                )

            def chunks(lst, n):
                for i in range(0, len(lst), n):
                    yield lst[i:i + n]

            chunked_years = list(chunks(selected_years, 3)
                                 )  # max 3 tahun per baris

            for year_group in chunked_years:
                cols = st.columns(len(year_group))

                for i, year in enumerate(year_group):
                    df_filtered = df_bday[df_bday['Segments/Departure Date'].dt.year == year].copy()

                    df_filtered['Birth Year'] = df_filtered['Passenger/Birth Date'].dt.year
                    df_filtered['Age'] = (
                        df_filtered['Segments/Departure Date'] - df_filtered['Passenger/Birth Date']).dt.days // 365
                    df_filtered['Generation'] = df_filtered['Birth Year'].apply(
                        get_generation)
                    df_filtered['AgeCategory'] = df_filtered['Age'].apply(
                        get_age)

                    if selected_chart == 'Adult/Child/Infant Distribution':
                        labels = ['Adult', 'Child', 'Infant']
                        sizes = [df_filtered['Adult'].sum(
                        ), df_filtered['Child'].sum(), df_filtered['Infant'].sum()]
                        fig = go.Figure(data=[go.Pie(
                            labels=labels,
                            values=sizes,
                            hole=0.3,
                            textinfo='percent+label',
                            marker=dict(
                                colors=['#ff9999', '#66b3ff', '#99ff99'])
                        )])
                        fig.update_layout(
                            title=f"Adult/Child/Infant Distribution ({year})", width=350, height=400)
                        fig.update_traces(showlegend=False)

                    elif selected_chart == 'Generation Distribution':  # Generation Distribution
                        generation_counts = df_filtered['Generation'].value_counts(
                        ).sort_index()
                        fig = go.Figure(data=[go.Pie(
                            labels=generation_counts.index,
                            values=generation_counts.values,
                            hole=0.3,
                            textinfo='label+value',
                            marker=dict(colors=[
                                        '#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#c2f0c2'])
                        )])
                        fig.update_layout(
                            title=f"Generation Distribution ({year})", width=350, height=400)
                        fig.update_traces(showlegend=False)

                    elif selected_chart == 'Age Distribution':
                        age_counts = df_filtered['AgeCategory'].value_counts(
                        ).sort_index()
                        fig = go.Figure(data=[go.Pie(
                            labels=age_counts.index,
                            values=age_counts.values,
                            hole=0.3,
                            textinfo='label+value',
                            marker=dict(
                                colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
                        )])
                        fig.update_layout(
                            title=f"Age Distribution ({year})", width=350, height=400)
                        fig.update_traces(showlegend=False)

                    with cols[i]:
                        st.plotly_chart(fig)

            if selected_chart == 'Adult/Child/Infant Distribution':
                labels = ['Adult', 'Child', 'Infant']
                sizes = [filtered_df['Adult'].sum(), filtered_df['Child'].sum(),
                         filtered_df['Infant'].sum()]

                fig = go.Figure(data=[go.Pie(labels=labels, values=sizes, hole=0.3,
                                             textinfo='percent+label', marker=dict(colors=['#ff9999', '#66b3ff', '#99ff99']))])

                fig.update_layout(
                    title=f"Adult/Child/Infant Distribution (All Time)",
                    width=400, height=500)

                st.plotly_chart(fig)

            elif selected_chart == 'Generation Distribution':
                fig = go.Figure(data=[go.Pie(labels=gen_all.index, values=gen_all.values, hole=0.3,
                                             textinfo='percent+label', marker=dict(colors=['#ff9999', '#66b3ff', '#99ff99',
                                                                                           '#ffcc99', '#c2c2f0', '#ffb3e6', '#c2f0c2']))])

                fig.update_layout(
                    title=f"Generation Distribution (All Time)",
                    width=400, height=500)

                st.plotly_chart(fig)

            elif selected_chart == 'Age Distribution':
                fig = go.Figure(data=[go.Pie(labels=age_all.index, values=age_all.values, hole=0.3,
                                             textinfo='percent+label', marker=dict(colors=['#ff9999', '#66b3ff', '#99ff99',
                                                                                           '#ffcc99', '#c2c2f0', '#ffb3e6', '#c2f0c2']))])

                fig.update_layout(
                    title=f"Age Distribution (All Time)",
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
                df_all['Departure Period'] = df_all['Segments/Departure Date'].dt.to_period(
                    'M').astype(str)
                grouped = df_all.groupby(
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
                
            df_all['Departure Period'] = df_all['Segments/Departure Date'].dt.to_period(
                'M').astype(str)
            grouped = df_all.groupby(
                ['Departure Period', 'AgeCategory']).size().reset_index(name='Count')

            fig = px.line(grouped, x='Departure Period',
                            y='Count', color='AgeCategory', markers=True)

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

    st.write('*Glossary:*')
    st.markdown("""
    - **Customer**  
    Shows the list and total count of passengers (Pax) for each customer.

    - **Customer Over Time**  
    Shows total passengers (Pax) per month for the selected customer.  

    - **Age Category**  
    Groups customers based on their age range.  
    """)
