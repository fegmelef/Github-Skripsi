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


def render_flights_summary(df, start_year, end_year, df_code):

    # region title & df flights
    col1, col2 = st.columns([9, 1])

    with col1:
        st.title(f"Flights Summary ({start_year} - {end_year})")

    with col2:
        st.button('Refresh')

    filtered_df = filter_year_month_depart("flight", df)

    option_flights = st.radio(
        "Sub-Menu:",
        options=["Airlines and Provider", "Direction and Sector",
                 'Flight Periods', 'Flight Routes'],
        key="option_flights",
        horizontal=True
    )

    df_direction = df.copy()
    # st.write(df_direction)

    # region airlines and provider
    if option_flights == 'Airlines and Provider':

        df_direction = df_direction.dropna(how='all')

        df_direction['Segments/Departure Date'] = pd.to_datetime(
            df_direction['Segments/Departure Date'])

        # df_direction['Total Pax'] = df_direction['Total Pax'].fillna(
        #     method='ffill')

        selected_years = filtered_df['Segments/Departure Date'].dt.year.unique()
        selected_months = filtered_df['Segments/Departure Date'].dt.month.unique()
        df_direction = df_direction[
            df_direction['Segments/Departure Date'].dt.year.isin(selected_years) &
            df_direction['Segments/Departure Date'].dt.month.isin(
                selected_months)
        ]

        plane_pax = df_direction.groupby(
            'Segments/Plane/Name')['Total Pax'].sum().reset_index()
        plane_pax = plane_pax.sort_values('Total Pax', ascending=False)

        provider_pax = df_direction.groupby(
            'Segments/Provider/Display Name')['Total Pax'].sum().reset_index()
        provider_pax = provider_pax.sort_values('Total Pax', ascending=False)

        df_direction['Departure Period'] = df_direction['Segments/Departure Date'].dt.to_period(
            'M').astype(str)

        option_airlines = st.radio(
            "Filter:",
            options=["Top Airlines", 'Top Providers',
                     'Airlines Over Time', 'Providers Over Time'],
            key="option_airlines",
            horizontal=True
        )

        if 'last_option_airlines' not in st.session_state:
            st.session_state.last_option_airlines = st.session_state.option_airlines

        if st.session_state.option_airlines != st.session_state.last_option_airlines:
            for key in [
                'top_n_quantity_slider_airlines',
                'top_n_quantity_slider_provider',
                'airlines_over_time', 'metric_providers', 'providers_over_time', 'metric_airlines'
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.last_option_airlines = st.session_state.option_airlines

        if option_airlines == 'Top Airlines':
            top_n = st.slider("Select the quantity to display",
                              min_value=5, max_value=20, value=10,
                              key="top_n_quantity_slider_airlines")

            plane_pax = (
                df_direction.groupby('List of Carriers')['Total Pax']
                .sum()
                .reset_index()
                .sort_values('Total Pax', ascending=False)
                .head(top_n)
            )

            fig_plane = px.bar(
                plane_pax,
                x='Total Pax',
                y='List of Carriers',
                title=f'Top {top_n} Airlines by Total Pax',
                orientation='h'
            )

            fig_plane.update_traces(
                hovertemplate='<b>Airline:</b> %{y}<br><b>Total Pax:</b> %{x}<extra></extra>'
            )

            fig_plane.update_layout(
                yaxis=dict(categoryorder='total ascending')
            )

            st.plotly_chart(fig_plane)

        elif option_airlines == 'Top Providers':
            top_n = st.slider("Select the quantity to display",
                              min_value=5, max_value=20, value=10,
                              key="top_n_quantity_slider_provider")

            provider_pax = (
                df_direction.groupby(
                    'List of Provider')['Total Pax']
                .sum()
                .reset_index()
                .sort_values('Total Pax', ascending=False)
                .head(top_n)
            )

            fig_provider = px.bar(
                provider_pax,
                x='Total Pax',
                y='List of Provider',
                title=f'Top {top_n} Providers by Total Pax',
                orientation='h'
            )

            fig_provider.update_traces(
                hovertemplate='<b>Provider:</b> %{y}<br><b>Total Pax:</b> %{x}<extra></extra>'
            )

            fig_provider.update_layout(
                yaxis=dict(categoryorder='total ascending')
            )

            st.plotly_chart(fig_provider)

        elif option_airlines == 'Airlines Over Time':
            col1, col2 = st.columns([6, 4])

            with col1:
                available_airlines = df_direction['List of Carriers'].dropna().unique()
                selected_airlines = st.multiselect(
                    "Select Airlines",
                    options=available_airlines,
                    default=available_airlines[0] if len(available_airlines) > 0 else None,
                    key='airlines_over_time'
                )

            with col2:
                metric = st.selectbox(
                    "Filter",
                    options=['By Total Pax', 'By Revenue', 'By Total Order'],
                    key='metric_airlines'
                )

            if selected_airlines:
                df_airline = df_direction[df_direction['List of Carriers'].isin(selected_airlines)]

                if metric == 'By Total Pax':
                    value_col = 'Total Pax'
                    grouped = (
                        df_airline.groupby(['Departure Period', 'List of Carriers'])[value_col]
                        .sum()
                        .reset_index()
                    )
                elif metric == 'By Revenue':
                    value_col = 'Grand Total'
                    grouped = (
                        df_airline.groupby(['Departure Period', 'List of Carriers'])[value_col]
                        .sum()
                        .reset_index()
                    )
                elif metric == 'By Total Order':
                    df_airline['Order Count'] = df_airline['Issued Date'].notna().astype(int)
                    grouped = (
                        df_airline.groupby(['Departure Period', 'List of Carriers'])['Order Count']
                        .sum()
                        .reset_index()
                    )
                    value_col = 'Order Count'

                fig_airline = px.line(
                    grouped,
                    x='Departure Period',
                    y=value_col,
                    color='List of Carriers',
                    title=f'{value_col} per Airlines (Monthly)'
                )
                st.plotly_chart(fig_airline)

        elif option_airlines == 'Providers Over Time':
            col1, col2 = st.columns([6, 4])

            with col1:
                available_providers = df_direction['List of Provider'].dropna().unique()
                selected_providers = st.multiselect(
                    "Select Providers",
                    options=available_providers,
                    default=available_providers[0] if len(available_providers) > 0 else None,
                    key='providers_over_time'
                )

            with col2:
                metric = st.selectbox(
                    "Metric",
                    options=['By Total Pax', 'By Revenue', 'By Total Order'],
                    key='metric_providers'
                )

            if selected_providers:
                df_provider = df_direction[df_direction['List of Provider'].isin(selected_providers)]

                if metric == 'By Total Pax':
                    value_col = 'Total Pax'
                    grouped = (
                        df_provider.groupby(['Departure Period', 'List of Provider'])[value_col]
                        .sum()
                        .reset_index()
                    )
                elif metric == 'By Revenue':
                    value_col = 'Grand Total'
                    grouped = (
                        df_provider.groupby(['Departure Period', 'List of Provider'])[value_col]
                        .sum()
                        .reset_index()
                    )
                elif metric == 'By Total Order':
                    df_provider['Order Count'] = df_provider['Issued Date'].notna().astype(int)
                    grouped = (
                        df_provider.groupby(['Departure Period', 'List of Provider'])['Order Count']
                        .sum()
                        .reset_index()
                    )
                    value_col = 'Order Count'

                fig_provider = px.line(
                    grouped,
                    x='Departure Period',
                    y=value_col,
                    color='List of Provider',
                    title=f'{value_col} per Provider (Monthly)'
                )
                st.plotly_chart(fig_provider)


    # endregion airlines and provider

    if 'last_option_flights' not in st.session_state:
        st.session_state.last_option_flights = option_flights

    if option_flights != st.session_state.last_option_flights:
        for key in [
            'selected_years2', 'selected_chart_ds',
            'top_n_slider', 'selected_sector_box',
            'selected_layer_radio', 'show_iata_checkbox', 'show_routes_checkbox',
            'route_type_select', 'route_scope_select',
            'selected_iata_code', 'top_n_dest_slider',
            'top_n_quantity_slider_airlines', 'top_n_quantity_slider_provider',
            'show_iata_checkbox',
            'show_routes_checkbox',
            'selected_layer_radio',
            'opt_route',
            'top_n_slider',
            'selected_sector_box',
            'option_airlines',
            'option_schedule',
            'option_period',
            'airline_schedule',
            'airlines_over_time', 'metric_providers', 'providers_over_time', 'metric_airlines'
        ]:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.last_option_flights = option_flights

    # region direction sector
    if option_flights == 'Direction and Sector':
        df_direction = df_direction.dropna(how='all')
        df_direction['Segments/Departure Date'] = pd.to_datetime(
            df_direction['Segments/Departure Date'])
        # df_direction['Total Pax'] = df_direction['Total Pax'].fillna(
        #     method='ffill')
        df_direction = df_direction.dropna(subset=['Segments/Departure Date'])

        available_years = sorted(
            df_direction['Segments/Departure Date'].dt.year.unique())

        col1, col2 = st.columns(2)

        with col1:
            selected_years2 = st.multiselect(
                "Select Year(s) to show chart",
                available_years,
                default=[available_years[-1]],
                key='selected_years2'
            )

        with col2:
            selected_chart = st.selectbox(
                "Select chart to display",
                ['Direction', 'Sector'],
                key='selected_chart_ds'
            )

        df_direction = df_direction.merge(
            df_code[["iata_code", "iso_country"]].rename(columns={
                "iata_code": "Segments/Origin/Code",
                "iso_country": "origin_country"
            }),
            on="Segments/Origin/Code",
            how="left"
        )

        df_direction = df_direction.merge(
            df_code[["iata_code", "iso_country"]].rename(columns={
                "iata_code": "Segments/Destination/Code",
                "iso_country": "dest_country"
            }),
            on="Segments/Destination/Code",
            how="left"
        )

        def update_sector(row):
            return "International" if row["origin_country"] != row["dest_country"] else "Domestic"

        # df_direction["Sector"] = df_direction.apply(update_sector, axis=1)
        # df_direction = df_direction.fillna(method='ffill')

        from math import ceil

        all_time_label = 'All Time'
        all_years_plus_all_time = selected_years2 + [all_time_label]

        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        chunked_years = list(chunks(all_years_plus_all_time, 3))

        for idx, year_group in enumerate(chunked_years):
            cols = st.columns(len(year_group))
            for j, year in enumerate(year_group):
                if year == all_time_label:
                    chart_data_all = df_direction.groupby(
                        selected_chart)['Total Pax'].sum().reset_index()
                    fig = px.pie(
                        chart_data_all,
                        names=selected_chart,
                        values='Total Pax',
                        title=f'{selected_chart} (All Time)',
                        hole=0.4
                    )
                    fig.update_traces(showlegend=True)
                else:
                    df_selected = df_direction[df_direction['Segments/Departure Date'].dt.year == year]
                    chart_data = df_selected.groupby(selected_chart)[
                        'Total Pax'].sum().reset_index()
                    fig = px.pie(
                        chart_data,
                        names=selected_chart,
                        values='Total Pax',
                        title=f'{selected_chart} ({year})',
                        hole=0.4
                    )
                    fig.update_traces(showlegend=False)

                with cols[j]:
                    st.plotly_chart(fig)

    # endregion direction sector

    # region periode flight
    if option_flights == 'Flight Periods':
        option_period = st.radio(
            "Filter:",
            options=["Flights Count", 'Flight Schedules'],
            key="option_period",
            horizontal=True
        )
        
        if 'last_option_period' not in st.session_state:
            st.session_state.last_option_period = option_period

        if option_period != st.session_state.last_option_period:
            for key in [
                'airline_schedule', 'option_schedule'
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.last_option_period = option_period

        # df_direction['Total Pax'] = df_direction['Total Pax'].fillna(
        #     method='ffill')
        df_direction['Segments/Departure Date'] = pd.to_datetime(
            df_direction['Segments/Departure Date'])
        df_direction['Segments/Arrival Date'] = pd.to_datetime(
            df_direction['Segments/Arrival Date'])

        df_direction['Departure Year'] = df_direction['Segments/Departure Date'].dt.year
        df_direction['Departure Month'] = df_direction['Segments/Departure Date'].dt.month
        df_direction['Arrival Year'] = df_direction['Segments/Arrival Date'].dt.year
        df_direction['Arrival Month'] = df_direction['Segments/Arrival Date'].dt.month

        selected_years = filtered_df['Segments/Departure Date'].dt.year.unique()
        selected_months = filtered_df['Segments/Departure Date'].dt.month.unique()
        df_direction = df_direction[
            df_direction['Segments/Departure Date'].dt.year.isin(selected_years) &
            df_direction['Segments/Departure Date'].dt.month.isin(
                selected_months)
        ]

        monthly_pax = (
            df_direction.groupby(['Departure Year', 'Departure Month'])[
                'Total Pax']
            .sum()
            .reset_index()
        )

        if option_period == 'Flights Count':

            col1, col2 = st.columns([6, 4])

            with col1:
                monthly_pax['Month Name'] = monthly_pax['Departure Month'].apply(
                    lambda x: calendar.month_abbr[int(x)])

                monthly_pax['Month Number'] = monthly_pax['Departure Month']
                monthly_pax = monthly_pax.sort_values(
                    ['Departure Year', 'Month Number'])

                all_months = list(range(1, 13))

                monthly_pax_fixed = []

                for year in monthly_pax['Departure Year'].unique():
                    df_year = monthly_pax[monthly_pax['Departure Year'] == year].set_index(
                        'Departure Month')
                    df_year = df_year.reindex(
                        all_months, fill_value=0).reset_index()
                    df_year['Departure Year'] = year
                    df_year['Month Name'] = df_year['Departure Month'].apply(
                        lambda x: calendar.month_abbr[x])
                    monthly_pax_fixed.append(df_year)

                monthly_pax = pd.concat(monthly_pax_fixed)

                fig_monthly = px.line(
                    monthly_pax,
                    x='Month Name',
                    y='Total Pax',
                    color='Departure Year',
                    title='Total Departure Over Time',
                    markers=True
                )
                st.plotly_chart(fig_monthly)

            with col2:
                yearly_pax = (
                    monthly_pax.groupby('Departure Year')['Total Pax']
                    .sum()
                    .reset_index()
                )

                fig_pie = px.pie(
                    yearly_pax,
                    values='Total Pax',
                    names='Departure Year',
                    title='Proportion Per Year',
                    hole=0.4
                )
                st.plotly_chart(fig_pie)

        elif option_period == 'Flight Schedules':
            option_col, filter_col = st.columns([3,7])

            with option_col:
                option_schedule = st.radio(
                    "Schedules for:",
                    options=["Departure", 'Arrival'],
                    key="option_schedule",
                    horizontal=True
                )

            with filter_col:
                available_airlines = df_direction['Segments/Plane/Name'].dropna().unique()
                airline_options = ['All'] + list(available_airlines)

                selected_airlines = st.selectbox(
                    "Select Airline",
                    options=airline_options,
                    key='airline_schedule',
                    index=0
                )

            col1, col2 = st.columns([6, 4])

            def categorize_time(hour):
                if 5 <= hour <= 12:
                    return 'Morning (5AM-12PM)'
                elif 13 <= hour <= 17:
                    return 'Afternoon (1PM-5PM)'
                elif 18 <= hour <= 21:
                    return 'Evening (6PM-9PM)'
                else:
                    return 'Night (10PM-4AM)'

            if option_schedule == 'Arrival':
                filtered_df['Segments/Arrival Date'] = pd.to_datetime(
                    filtered_df['Segments/Arrival Date'], errors='coerce'
                )

                selected_years = filtered_df['Segments/Arrival Date'].dt.year.unique()
                selected_months = filtered_df['Segments/Arrival Date'].dt.month.unique()

                df_arrival = df_direction[
                    df_direction['Segments/Arrival Date'].dt.year.isin(selected_years) &
                    df_direction['Segments/Arrival Date'].dt.month.isin(selected_months)
                ]

                if selected_airlines != 'All':
                    df_arrival = df_arrival[df_arrival['Segments/Plane/Name'] == selected_airlines]

                with col1:
                    df_arrival['Arrival Hour'] = df_arrival['Segments/Arrival Date'].dt.hour

                    hourly_pax = (
                        df_arrival.groupby('Arrival Hour')['Total Pax']
                        .sum()
                        .reindex(range(24), fill_value=0)
                        .reset_index()
                    )

                    fig_heatmap = px.imshow(
                        [hourly_pax['Total Pax']],
                        labels=dict(x="Arrival Hours", color="Total Pax"),
                        x=hourly_pax['Arrival Hour'],
                        y=[""],
                        color_continuous_scale='YlGnBu'
                    )

                    fig_heatmap.update_layout(
                        title='Total Pax by Arrival Hours',
                        yaxis=dict(showticklabels=False),
                        height=300,
                        coloraxis_colorbar=dict(
                            title="Total Pax",
                            tickvals=[hourly_pax['Total Pax'].min(), hourly_pax['Total Pax'].max()],
                            ticktext=[
                                f"{hourly_pax['Total Pax'].min():,.0f}", f"{hourly_pax['Total Pax'].max():,.0f}"
                            ],
                        )
                    )

                    st.plotly_chart(fig_heatmap)

                with col2:
                    df_arrival['Time Category'] = df_arrival['Arrival Hour'].apply(categorize_time)

                    time_category_pax = (
                        df_arrival.groupby('Time Category')['Total Pax']
                        .sum()
                        .reset_index()
                    )

                    fig_time_category = px.pie(
                        time_category_pax,
                        names='Time Category',
                        values='Total Pax',
                        title='Total Pax Distribution by Arrival Hours'
                    )
                    st.plotly_chart(fig_time_category)

            elif option_schedule == 'Departure':
                filtered_df['Segments/Departure Date'] = pd.to_datetime(
                    filtered_df['Segments/Departure Date'], errors='coerce'
                )

                selected_years = filtered_df['Segments/Departure Date'].dt.year.unique()
                selected_months = filtered_df['Segments/Departure Date'].dt.month.unique()

                df_departure = df_direction[
                    df_direction['Segments/Departure Date'].dt.year.isin(selected_years) &
                    df_direction['Segments/Departure Date'].dt.month.isin(selected_months)
                ]

                if selected_airlines != 'All':
                    df_departure = df_departure[df_departure['Segments/Plane/Name'] == selected_airlines]

                with col1:
                    df_departure['Departure Hour'] = df_departure['Segments/Departure Date'].dt.hour

                    hourly_pax = (
                        df_departure.groupby('Departure Hour')['Total Pax']
                        .sum()
                        .reindex(range(24), fill_value=0)
                        .reset_index()
                    )

                    fig_heatmap = px.imshow(
                        [hourly_pax['Total Pax']],
                        labels=dict(x="Departure Hours", color="Total Pax"),
                        x=hourly_pax['Departure Hour'],
                        y=[""],
                        color_continuous_scale='YlGnBu'
                    )

                    fig_heatmap.update_layout(
                        title='Total Pax by Departure Hours',
                        yaxis=dict(showticklabels=False),
                        height=300,
                        coloraxis_colorbar=dict(
                            title="Total Pax",
                            tickvals=[hourly_pax['Total Pax'].min(), hourly_pax['Total Pax'].max()],
                            ticktext=[
                                f"{hourly_pax['Total Pax'].min():,.0f}", f"{hourly_pax['Total Pax'].max():,.0f}"
                            ],
                        )
                    )

                    st.plotly_chart(fig_heatmap)

                with col2:
                    df_departure['Time Category'] = df_departure['Departure Hour'].apply(categorize_time)

                    time_category_pax = (
                        df_departure.groupby('Time Category')['Total Pax']
                        .sum()
                        .reset_index()
                    )

                    fig_time_category = px.pie(
                        time_category_pax,
                        names='Time Category',
                        values='Total Pax',
                        title='Total Pax Distribution by Departure Hours'
                    )
                    st.plotly_chart(fig_time_category)


    # endregion periode flight

    if option_flights == 'Flight Routes':
        opt_route = st.radio(
            "Sub-Menu:",
            options=["Top Routes", 'Map'],
            key="opt_route",
            horizontal=True
        )

        if 'last_opt_route' not in st.session_state:
            st.session_state.last_opt_route = st.session_state.opt_route

        if st.session_state.opt_route != st.session_state.last_opt_route:
            for key in [
                'route_type_select',
                'route_scope_select',
                'selected_iata_code',
                'top_n_dest_slider',
                'show_iata_checkbox',
                'show_routes_checkbox',
                'selected_layer_radio',
            ]:
                st.session_state.pop(key, None)
            st.session_state.last_opt_route = st.session_state.opt_route

        col1, col2 = st.columns([3, 7])

        if opt_route == 'Top Routes':
            df_routetop = df[
                df['Segments/Origin/Code'].notna() &
                df['Segments/Destination/Code'].notna()
            ]

            df_routetop = df_routetop[
                (df_routetop['Segments/Origin/Code'].astype(str) != 'False') &
                (df_routetop['Segments/Destination/Code'].astype(str) != 'False')
            ]

            # df_routetop['Total Pax'] = df_routetop['Total Pax'].fillna(
            #     method='ffill')

            df_route = df_routetop.groupby(
                ['Segments/Origin/Code', 'Segments/Destination/Code']
            )['Total Pax'].sum().reset_index()

            df_route['Route'] = df_route.apply(
                lambda row: f"{row['Segments/Origin/Code']} - {row['Segments/Destination/Code']}",
                axis=1
            )

            df_routetop['Sector'] = df_routetop['Sector'].fillna(
                method='ffill')

            sectors = df_routetop['Sector'].unique()

            with col2:
                top_n = st.slider(
                    "Select the Number of Top Routes",
                    min_value=5,
                    max_value=20,
                    value=10,
                    key='top_n_slider'
                )

            with col1:
                selected_sector = st.selectbox(
                    "Select Sector",
                    options=sectors,
                    key='selected_sector_box'
                )

            df_filtered_sector = df_routetop[df_routetop['Sector']
                                             == selected_sector]

            df_route = df_filtered_sector.groupby(
                ['Segments/Origin/Code', 'Segments/Destination/Code']
            )['Total Pax'].sum().reset_index()

            df_route['Route'] = df_route.apply(
                lambda row: f"{row['Segments/Origin/Code']} - {row['Segments/Destination/Code']}",
                axis=1
            )

            df_route = df_route.sort_values(
                'Total Pax', ascending=False).head(top_n)

            fig_route = px.bar(
                df_route,
                x='Total Pax',
                y='Route',
                title=f'Top {top_n} {selected_sector} Routes',
                orientation='h'
            )

            fig_route.update_layout(
                yaxis=dict(categoryorder='total ascending')
            )
            st.plotly_chart(fig_route)

        elif opt_route == 'Map':
            filtered_df['Segments/Departure Date'] = pd.to_datetime(
                filtered_df['Segments/Departure Date'], errors='coerce'
            )

            selected_years = filtered_df['Segments/Departure Date'].dt.year.unique()
            selected_months = filtered_df['Segments/Departure Date'].dt.month.unique()
            df_pax = df_direction[
                df_direction['Segments/Departure Date'].dt.year.isin(selected_years) &
                df_direction['Segments/Departure Date'].dt.month.isin(
                    selected_months)
            ]

            # df_pax["Total Pax"] = pd.to_numeric(
            #     df["Total Pax"], errors="coerce").fillna(method="ffill")

            df_pax_origin = df_pax.groupby(
                "Segments/Origin/Code")["Total Pax"].sum().reset_index()
            df_pax_origin.columns = ["iata_code", "Total Pax Sum"]

            df_pax_dest = df.groupby("Segments/Destination/Code")[
                "Total Pax"].sum().reset_index()
            df_pax_dest.columns = ["iata_code", "Total Pax Sum"]

            df_asal = df_pax_origin.merge(df_code, on="iata_code", how="left")
            df_tujuan = df_pax_dest.merge(df_code, on="iata_code", how="left")

            df_asal[['latitude', 'longitude']
                    ] = df_asal['coordinates'].str.split(',', expand=True)
            df_asal['latitude'] = pd.to_numeric(
                df_asal['latitude'], errors='coerce')
            df_asal['longitude'] = pd.to_numeric(
                df_asal['longitude'], errors='coerce')

            df_tujuan[['latitude', 'longitude']
                      ] = df_tujuan['coordinates'].str.split(',', expand=True)
            df_tujuan['latitude'] = pd.to_numeric(
                df_tujuan['latitude'], errors='coerce')
            df_tujuan['longitude'] = pd.to_numeric(
                df_tujuan['longitude'], errors='coerce')

            # Filter negara
            unique_countries = df_tujuan["iso_country"].dropna().unique()
            selected_country = st.selectbox("Select Country (ISO)", options=[
                "All"] + sorted(unique_countries))

            # Filter negara
            if selected_country != "All":
                df_asal_filtered = df_asal[df_asal["iso_country"]
                                           == selected_country]
                df_tujuan_filtered = df_tujuan[df_tujuan["iso_country"]
                                               == selected_country]
            else:
                df_tujuan_filtered = df_tujuan.copy()
                df_asal_filtered = df_asal.copy()

            # Hapus baris tanpa koordinat
            df_tujuan_filtered = df_tujuan_filtered.dropna(
                subset=["latitude", "longitude"])
            df_asal_filtered = df_asal_filtered.dropna(
                subset=["latitude", "longitude"])

            def get_blue_gradient(value):
                r = int(200 * (1 - value))
                g = int(200 * (1 - value))
                b = 255
                return [r, g, b, 160]

            min_val = df_asal_filtered["Total Pax Sum"].min()
            max_val = df_asal_filtered["Total Pax Sum"].max()

            # Tetapkan nilai skala default untuk radiusScale
            RADIUS_SCALE = 10

            # Radius proporsional tetapi cukup besar agar tetap terlihat di zoom out
            df_asal_filtered["scaled_radius"] = (
                (df_asal_filtered["Total Pax Sum"] -
                 min_val) / (max_val - min_val)
            ) * 20 + 5  # Gunakan skala kecil karena radiusScale akan membesarkan

            # Warna tetap
            df_asal_filtered["color_scale"] = (
                df_asal_filtered["scaled_radius"] -
                df_asal_filtered["scaled_radius"].min()
            ) / (df_asal_filtered["scaled_radius"].max() - df_asal_filtered["scaled_radius"].min())

            # Warna berdasarkan Total Pax
            df_asal_filtered["normalized"] = (
                df_asal_filtered["Total Pax Sum"] - min_val
            ) / (max_val - min_val)

            def get_gradient_color(value):
                if value < 500:
                    norm_val = value / 999  # normalisasi ke 0–1
                    r = 255
                    g = 255
                    b = int(255 * (1 - norm_val))  # dari 255 ke 0
                    return [r, g, b, 200]
                else:
                    norm_val = min((value - 1000) / (max_val - 1000), 1.0)
                    r = 255
                    g = int(255 * (1 - norm_val))  # dari 255 ke 0
                    b = 0
                    return [r, g, b, 220]

            df_asal_filtered["fill_color"] = df_asal_filtered["Total Pax Sum"].apply(
                get_gradient_color)
            df_tujuan_filtered["fill_color"] = df_tujuan_filtered["Total Pax Sum"].apply(
                get_gradient_color)

            ALL_LAYERS = {
                "Origin": pdk.Layer(
                    "ScatterplotLayer",
                    data=df_asal_filtered,
                    get_position=["longitude", "latitude"],
                    get_radius=35000,  # Ukuran tetap
                    get_fill_color="fill_color",
                    pickable=True,
                    auto_highlight=True,
                ),
                "Destination": pdk.Layer(
                    "ScatterplotLayer",
                    data=df_tujuan_filtered,
                    get_position=["longitude", "latitude"],
                    get_radius=35000,  # Ukuran tetap
                    get_fill_color="fill_color",
                    pickable=True,
                    auto_highlight=True,
                ),
                "IATA Code Airport": pdk.Layer(
                    "TextLayer",
                    data=df_asal_filtered,
                    get_position=["longitude", "latitude"],
                    get_text="iata_code",
                    get_color=[128, 0, 128, 230],
                    get_size=12,
                    get_alignment_baseline="'bottom'",
                ),
            }

            st.sidebar.markdown("### Map Filter")

            # Ambil nilai radio terbaru
            selected_layer = st.sidebar.radio(
                "Filter:",
                ("Origin", "Destination"),
                key="selected_layer_radio"
            )
            
            # Simpan nilai sebelumnya
            if 'last_selected_layer' not in st.session_state:
                st.session_state.last_selected_layer = st.session_state.selected_layer_radio

            # Jika terjadi perubahan, reset semua dan rerun
            if selected_layer != st.session_state.last_selected_layer:
                for key in [
                    'route_type_select',
                    'route_scope_select',
                    'selected_iata_code',
                    'top_n_dest_slider',
                    'show_iata_checkbox',
                    'show_routes_checkbox',
                ]:
                    st.session_state.pop(key, None)
                st.session_state.last_selected_layer = selected_layer
                st.rerun()

            show_airport_names = st.sidebar.checkbox(
                "Show IATA Code", value=True, key=f"show_iata_checkbox_{selected_layer}"
            )

            # Track previous checkbox state
            if 'last_show_routes' not in st.session_state:
                st.session_state.last_show_routes = False

            # Checkbox UI
            show_routes = st.sidebar.checkbox("Show Top Routes", value=False, key=f"show_routes_checkbox_{selected_layer}")

            # Deteksi transisi dari False → True
            show_routes_just_enabled = show_routes and not st.session_state.last_show_routes

            # Simpan state sekarang
            st.session_state.last_show_routes = show_routes

            
            selected_layers = []

            if selected_layer == "Origin":
                selected_layers.append(ALL_LAYERS["Origin"])
            elif selected_layer == "Destination":
                selected_layers.append(ALL_LAYERS["Destination"])

            if show_routes:
                available_iata_codes = sorted(set(df_asal_filtered["iata_code"]) | set(df_tujuan_filtered["iata_code"]))

                if show_routes_just_enabled:
                    st.session_state.route_type_select = "From"
                    st.session_state.route_scope_select = "All"
                    st.session_state.selected_iata_code = available_iata_codes[0]
                    st.session_state.top_n_dest_slider = 5
                    st.session_state.last_route_type = "From"

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    route_type = st.selectbox("Route Type", ["From", "To"], key="route_type_select")

                current_route_type = st.session_state.route_type_select

                if current_route_type != st.session_state.last_route_type:
                    st.session_state.route_scope_select = "All"
                    st.session_state.selected_iata_code = available_iata_codes[0]
                    st.session_state.top_n_dest_slider = 5
                    st.session_state.last_route_type = current_route_type

                with col2:
                    route_scope = st.selectbox("Route Scope", ["All", "Domestic", "International"], key="route_scope_select")

                with col3:
                    selected_iata = st.selectbox(f"Select IATA Code ({route_type})", available_iata_codes, key="selected_iata_code")

                with col4:
                    top_n = st.slider("Top N Destinations", min_value=5, max_value=10, key="top_n_dest_slider")

                # Merge lookup origin country
                df_pax = df_pax.merge(
                    df_code[["iata_code", "iso_country"]].rename(columns={
                        "iata_code": "Segments/Origin/Code",
                        "iso_country": "origin_country"
                    }),
                    on="Segments/Origin/Code",
                    how="left"
                )

                # Merge lookup destination country
                df_pax = df_pax.merge(
                    df_code[["iata_code", "iso_country"]].rename(columns={
                        "iata_code": "Segments/Destination/Code",
                        "iso_country": "dest_country"
                    }),
                    on="Segments/Destination/Code",
                    how="left"
                )

                # Isi Sector yang kosong berdasarkan perbandingan negara
                def fill_sector(row):
                    if pd.isna(row["Sector"]) or row["Sector"] == "":
                        if row["origin_country"] != row["dest_country"]:
                            return "International"
                        else:
                            return "Domestic"
                    else:
                        return row["Sector"]

                def update_sector(row):
                    if row["origin_country"] != row["dest_country"]:
                        return "International"
                    else:
                        return "Domestic"

                df_pax["Sector"] = df_pax.apply(update_sector, axis=1)

                # Optional: hapus kolom bantu
                df_pax = df_pax.drop(
                    columns=["origin_country", "dest_country"])

                if route_type == "From":
                    df_routes = df_pax[df_pax["Segments/Origin/Code"]
                                       == selected_iata]

                    if route_scope != "All":
                        df_routes = df_routes[df_routes["Sector"]
                                              == route_scope]

                    df_routes_grouped = (
                        df_routes.groupby(
                            "Segments/Destination/Code")["Total Pax"]
                        .sum()
                        .nlargest(top_n)
                        .reset_index()
                    )

                    if df_routes_grouped.empty:
                        st.warning(
                            "No data available for the selected filters.")
                    else:
                        df_routes_grouped.columns = [
                            "iata_code", "Total Pax Sum"]
                        df_routes_grouped = df_routes_grouped.merge(
                            df_code, on="iata_code", how="left")
                        df_origin_info = df_code[df_code["iata_code"]
                                                 == selected_iata].iloc[0]

                        df_routes_grouped["origin_lat"] = df_origin_info["coordinates"].split(",")[
                            0]
                        df_routes_grouped["origin_lon"] = df_origin_info["coordinates"].split(",")[
                            1]
                        df_routes_grouped[["latitude", "longitude"]] = df_routes_grouped["coordinates"].str.split(
                            ",", expand=True)

                        df_routes_grouped[["latitude", "longitude", "origin_lat", "origin_lon"]] = df_routes_grouped[
                            ["latitude", "longitude", "origin_lat", "origin_lon"]
                        ].apply(pd.to_numeric, errors="coerce")

                        df_routes_grouped = df_routes_grouped.dropna(
                            subset=["latitude", "longitude", "origin_lat", "origin_lon"])

                        if df_routes_grouped.empty:
                            st.warning(
                                "No valid coordinate data to display on the map.")
                        else:
                            route_layer = pdk.Layer(
                                "ArcLayer",
                                data=df_routes_grouped,
                                get_source_position=[
                                    "origin_lon", "origin_lat"],
                                get_target_position=["longitude", "latitude"],
                                get_source_color=[0, 128, 255, 160],
                                get_target_color=[255, 0, 0, 160],
                                auto_highlight=True,
                                pickable=True,
                                get_width="Total Pax Sum",
                                width_scale=3,
                            )

                            selected_layers.append(route_layer)

                else:  # route_type == "To"
                    df_routes = df_pax[df_pax["Segments/Destination/Code"]
                                       == selected_iata]

                    if route_scope != "All":
                        df_routes = df_routes[df_routes["Sector"]
                                              == route_scope]

                    df_routes_grouped = (
                        df_routes.groupby("Segments/Origin/Code")["Total Pax"]
                        .sum()
                        .nlargest(top_n)
                        .reset_index()
                    )

                    if df_routes_grouped.empty:
                        st.warning(
                            "No data available for the selected filters.")
                    else:
                        df_routes_grouped.columns = [
                            "iata_code", "Total Pax Sum"]
                        df_routes_grouped = df_routes_grouped.merge(
                            df_code, on="iata_code", how="left")
                        df_dest_info = df_code[df_code["iata_code"]
                                               == selected_iata].iloc[0]

                        df_routes_grouped["latitude"] = df_dest_info["coordinates"].split(",")[
                            0]
                        df_routes_grouped["longitude"] = df_dest_info["coordinates"].split(",")[
                            1]
                        df_routes_grouped[["origin_lat", "origin_lon"]] = df_routes_grouped["coordinates"].str.split(
                            ",", expand=True)

                        df_routes_grouped[["latitude", "longitude", "origin_lat", "origin_lon"]] = df_routes_grouped[
                            ["latitude", "longitude", "origin_lat", "origin_lon"]
                        ].apply(pd.to_numeric, errors="coerce")

                        df_routes_grouped = df_routes_grouped.dropna(
                            subset=["latitude", "longitude", "origin_lat", "origin_lon"])

                        if df_routes_grouped.empty:
                            st.warning(
                                "No valid coordinate data to display on the map.")
                        else:
                            route_layer = pdk.Layer(
                                "ArcLayer",
                                data=df_routes_grouped,
                                get_source_position=[
                                    "origin_lon", "origin_lat"],
                                get_target_position=["longitude", "latitude"],
                                get_source_color=[0, 128, 255, 160],
                                get_target_color=[255, 0, 0, 160],
                                auto_highlight=True,
                                pickable=True,
                                get_width="Total Pax Sum",
                                width_scale=3,
                            )

                            selected_layers.append(route_layer)

            if show_airport_names:
                selected_layers.append(ALL_LAYERS["IATA Code Airport"])

            if selected_layers:
                if show_routes and not df_routes_grouped.empty:
                    center_lat = df_routes_grouped[["latitude", "origin_lat"]].astype(
                        float).mean().mean()
                    center_lon = df_routes_grouped[["longitude", "origin_lon"]].astype(
                        float).mean().mean()
                else:
                    center_lat = df_tujuan_filtered["latitude"].mean()
                    center_lon = df_tujuan_filtered["longitude"].mean()

                st.pydeck_chart(
                    pdk.Deck(
                        map_style='mapbox://styles/mapbox/satellite-streets-v12',
                        initial_view_state=pdk.ViewState(
                            latitude=center_lat,
                            longitude=center_lon,
                            zoom=3,
                            pitch=40,
                        ),
                        layers=selected_layers,
                        tooltip={
                            "text": "Airport: {name}\nCode: {iata_code}\nCountry: {iso_country}\nTotal Pax: {Total Pax Sum}"
                        }
                    )
                )

    st.write('*Glossary:*')
    st.markdown("""
    - **Airlines and Provider**  
    Displays the airline companies and their associated service providers.
        + **Top Airlines**  
        Shows the airlines with the most passenger.

        + **Top Providers**  
        Shows the service providers with the most passenger.

        + **Airlines Over Time**  
        Displays total passengers or sales for selected airline by month.

        + **Providers Over Time**  
        Displays total passengers or sales for selected provider by month.

    - **Direction and Sector**  
    Shows the travel direction (e.g., one-way/return/multi city) and regional (e.g., domestic/international) sector classification.

    - **Flight Periods**  
    Represents the different periods during which flights are scheduled.

        + **Flights Count**  
        Shows the total number of passengers scheduled to fly within the selected period.

        + **Flight Schedules**  
        Displays the specific scheduled times for flights within each period (e.g., morning, afternoon, evening).

    - **Flight Routes**  
    Displays the origin and destination pairs for flights.
        + **Top Routes**  
        Shows the most popular flight routes.

        + **Map**  
        Displays a map visualization of the flight routes.
    """)
