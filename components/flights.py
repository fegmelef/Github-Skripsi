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

    # region airlines and provider
    if option_flights == 'Airlines and Provider':

        df_direction = df_direction.dropna(how='all')

        df_direction['Segments/Departure Date'] = pd.to_datetime(
            df_direction['Segments/Departure Date'])

        df_direction['Total Pax'] = df_direction['Total Pax'].fillna(
            method='ffill')

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

        if option_airlines == 'Top Airlines':
            top_n = st.slider("Pilih jumlah yang ingin ditampilkan",
                              min_value=5, max_value=20, value=10)

            plane_pax = (
                df_direction.groupby('Segments/Plane/Name')['Total Pax']
                .sum()
                .reset_index()
                .sort_values('Total Pax', ascending=False)
                .head(top_n)
            )

            fig_plane = px.bar(
                plane_pax,
                x='Total Pax',
                y='Segments/Plane/Name',
                title=f'Top {top_n} Airlines by Total Pax',
                orientation='h'
            )
            st.plotly_chart(fig_plane)

        elif option_airlines == 'Top Providers':
            top_n = st.slider("Pilih jumlah yang ingin ditampilkan",
                              min_value=5, max_value=20, value=10)

            provider_pax = (
                df_direction.groupby(
                    'Segments/Provider/Display Name')['Total Pax']
                .sum()
                .reset_index()
                .sort_values('Total Pax', ascending=False)
                .head(top_n)
            )

            fig_provider = px.bar(
                provider_pax,
                x='Total Pax',
                y='Segments/Provider/Display Name',
                title=f'Top {top_n} Providers by Total Pax',
                orientation='h'
            )
            st.plotly_chart(fig_provider)

        elif option_airlines == 'Airlines Over Time':
            available_airlines = df_direction['Segments/Plane/Name'].dropna().unique()

            selected_airlines = st.multiselect(
                "Pilih Airlines",
                options=available_airlines,
                default=available_airlines[0] if len(
                    available_airlines) > 0 else None
            )

            if selected_airlines:
                df_airline = df_direction[df_direction['Segments/Plane/Name'].isin(
                    selected_airlines)]
                airline_grouped = (
                    df_airline.groupby(
                        ['Departure Period', 'Segments/Plane/Name'])['Total Pax']
                    .sum()
                    .reset_index()
                )
                fig_airline = px.line(
                    airline_grouped,
                    x='Departure Period',
                    y='Total Pax',
                    color='Segments/Plane/Name',
                    title='Total Pax per Airlines (Monthly)'
                )
                st.plotly_chart(fig_airline)

        elif option_airlines == 'Providers Over Time':

            available_providers = df_direction['Segments/Provider/Display Name'].dropna(
            ).unique()
            selected_providers = st.multiselect(
                "Pilih Provider", options=available_providers,
                default=available_providers[0] if len(
                    available_providers) > 0 else None
            )

            if selected_providers:
                df_provider = df_direction[df_direction['Segments/Provider/Display Name'].isin(
                    selected_providers)]
                provider_grouped = (
                    df_provider.groupby(
                        ['Departure Period', 'Segments/Provider/Display Name'])['Total Pax']
                    .sum()
                    .reset_index()
                )
                fig_provider = px.line(
                    provider_grouped,
                    x='Departure Period',
                    y='Total Pax',
                    color='Segments/Provider/Display Name',
                    title='Total Pax per Provider (Monthly)'
                )
                st.plotly_chart(fig_provider)

    # endregion airlines and provider

    # region direction sector
    elif option_flights == 'Direction and Sector':
        df_direction = df_direction.dropna(how='all')

        df_direction['Segments/Departure Date'] = pd.to_datetime(
            df_direction['Segments/Departure Date'])

        df_direction['Total Pax'] = df_direction['Total Pax'].fillna(
            method='ffill')

        selected_years = filtered_df['Segments/Departure Date'].dt.year.unique()
        selected_months = filtered_df['Segments/Departure Date'].dt.month.unique()
        df_direction = df_direction[
            df_direction['Segments/Departure Date'].dt.year.isin(selected_years) &
            df_direction['Segments/Departure Date'].dt.month.isin(
                selected_months)
        ]

        col1, col2 = st.columns(2)

        with col1:
            direction_pax = (
                df_direction.groupby('Direction')['Total Pax']
                .sum()
                .reset_index()
            )

            fig_direction = px.pie(
                direction_pax,
                names='Direction',
                values='Total Pax',
                title='Total Pax per Direction',
                hole=0.4
            )
            st.plotly_chart(fig_direction)

        with col2:
            sector_pax = (
                df_direction.groupby('Sector')['Total Pax']
                .sum()
                .reset_index()
            )

            fig_sector = px.pie(
                sector_pax,
                names='Sector',
                values='Total Pax',
                title='Total Pax per Sector',
                hole=0.4
            )
            st.plotly_chart(fig_sector)

    # endregion direction sector

    # region periode flight
    elif option_flights == 'Flight Periods':
        option_period = st.radio(
            "Filter:",
            options=["Flights Count", 'Flight Schedules'],
            key="option_period",
            horizontal=True
        )
        df_direction['Total Pax'] = df_direction['Total Pax'].fillna(
            method='ffill')
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
            option_schedule = st.radio(
                "Schedules for:",
                options=["Departure", 'Arrival'],
                key="option_schedule",
                horizontal=True
            )

            col1, col2 = st.columns([6, 4])

            def categorize_time(hour):
                if 5 <= hour <= 12:
                    return 'Morning (5-12)'
                elif 12 <= hour <= 17:
                    return 'Afternoon (12-17)'
                elif 17 <= hour <= 21:
                    return 'Evening (17-21)'
                else:  # 21-23 and 0-4
                    return 'Night (21-5)'

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

                with col1:
                    df_arrival['Arrival Hour'] = df_arrival['Segments/Arrival Date'].dt.hour

                    # Buat pivot table jam vs jumlah pax (bisa 1 baris, atau kamu bisa extend ke hari/nama bandara jika ingin 2D)
                    hourly_pax = (
                        df_arrival.groupby('Arrival Hour')['Total Pax']
                        .sum()
                        .reindex(range(24), fill_value=0)  # pastikan semua jam ada
                        .reset_index()
                    )

                    fig_heatmap = px.imshow(
                        [hourly_pax['Total Pax']],  # List of list (1D heatmap)
                        labels=dict(x="Jam Kedatangan", color="Total Pax"),
                        x=hourly_pax['Arrival Hour'],
                        y=[""],  # satu baris
                        color_continuous_scale='YlGnBu'
                    )

                    fig_heatmap.update_layout(
                        title='Heatmap Total Pax Berdasarkan Jam Kedatangan',
                        yaxis=dict(showticklabels=False),
                        height=200,
                    )

                    st.plotly_chart(fig_heatmap)


                with col2:
                    df_arrival['Time Category'] = df_arrival['Arrival Hour'].apply(
                        categorize_time)

                    time_category_pax = (
                        df_arrival.groupby('Time Category')['Total Pax']
                        .sum()
                        .reset_index()
                    )

                    fig_time_category = px.pie(
                        time_category_pax,
                        names='Time Category',
                        values='Total Pax',
                        title='Distribusi Total Pax Berdasarkan Waktu Kedatangan'
                    )
                    st.plotly_chart(fig_time_category)

            elif option_schedule == 'Departure':
                filtered_df['Segments/Departure Date'] = pd.to_datetime(
                    filtered_df['Segments/Departure Date'], errors='coerce'
                )

                selected_years = filtered_df['Segments/Departure Date'].dt.year.unique()
                selected_months = filtered_df['Segments/Departure Date'].dt.month.unique()
                df_direction = df_direction[
                    df_direction['Segments/Departure Date'].dt.year.isin(selected_years) &
                    df_direction['Segments/Departure Date'].dt.month.isin(
                        selected_months)
                ]

                with col1:
                    df_direction['Departure Hour'] = df_direction['Segments/Departure Date'].dt.hour

                    hourly_pax = (
                        df_direction.groupby('Departure Hour')['Total Pax']
                        .sum()
                        .reindex(range(24), fill_value=0) 
                        .reset_index()
                    )

                    fig_heatmap = px.imshow(
                        [hourly_pax['Total Pax']],
                        labels=dict(x="Jam Keberangkatan", color="Total Pax"),
                        x=hourly_pax['Departure Hour'],
                        y=[""],  # hanya satu baris
                        color_continuous_scale='YlGnBu'
                    )

                    fig_heatmap.update_layout(
                        title='Heatmap Total Pax Berdasarkan Jam Keberangkatan',
                        yaxis=dict(showticklabels=False),
                        height=200,
                    )

                    st.plotly_chart(fig_heatmap)

                with col2:
                    df_direction['Time Category'] = df_direction['Departure Hour'].apply(
                        categorize_time)

                    time_category_pax = (
                        df_direction.groupby('Time Category')['Total Pax']
                        .sum()
                        .reset_index()
                    )

                    fig_time_category = px.pie(
                        time_category_pax,
                        names='Time Category',
                        values='Total Pax',
                        title='Distribusi Total Pax Berdasarkan Waktu Keberangkatan'
                    )
                    st.plotly_chart(fig_time_category)

    # endregion periode flight

    elif option_flights == 'Flight Routes':
        opt_route = st.radio(
            "Sub-Menu:",
            options=["Top Routes", 'Map'],
            key="opt_route",
            horizontal=True
        )

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

            df_routetop['Total Pax'] = df_routetop['Total Pax'].fillna(
                method='ffill')

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
                top_n = st.slider("Pilih Jumlah Top Route",
                                  min_value=5, max_value=20, value=10)

            with col1:
                selected_sector = st.selectbox("Pilih Sector", options=sectors)

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

            df_pax["Total Pax"] = pd.to_numeric(
                df["Total Pax"], errors="coerce").fillna(method="ffill")

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
            selected_country = st.selectbox("Pilih Negara (ISO)", options=[
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
                "Asal": pdk.Layer(
                    "ScatterplotLayer",
                    data=df_asal_filtered,
                    get_position=["longitude", "latitude"],
                    get_radius=35000,  # Ukuran tetap
                    get_fill_color="fill_color",
                    pickable=True,
                    auto_highlight=True,
                ),
                "Tujuan": pdk.Layer(
                    "ScatterplotLayer",
                    data=df_tujuan_filtered,
                    get_position=["longitude", "latitude"],
                    get_radius=35000,  # Ukuran tetap
                    get_fill_color="fill_color",
                    pickable=True,
                    auto_highlight=True,
                ),
                "Nama Bandara": pdk.Layer(
                    "TextLayer",
                    data=df_asal_filtered,
                    get_position=["longitude", "latitude"],
                    get_text="iata_code",
                    get_color = [128, 0, 128, 230],
                    get_size=12,
                    get_alignment_baseline="'bottom'",
                ),
            }

            st.sidebar.markdown("### Map Layers")
            selected_layer = st.sidebar.radio(
                "Filter:",
                ("Asal", "Tujuan")
            )

            show_airport_names = st.sidebar.checkbox("Tampilkan Nama Bandara", value=True)

            selected_layers = []

            if selected_layer == "Asal":
                selected_layers.append(ALL_LAYERS["Asal"])
            elif selected_layer == "Tujuan":
                selected_layers.append(ALL_LAYERS["Tujuan"])

            if show_airport_names:
                selected_layers.append(ALL_LAYERS["Nama Bandara"])

            if selected_layers:
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
                            "text": "Bandara: {name}\nKode: {iata_code}\nNegara: {iso_country}\nTotal Pax: {Total Pax Sum}"
                        },
                    )
                )
