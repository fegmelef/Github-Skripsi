import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
from datetime import datetime
import calendar
from utils import load_data, filter_year_month, filter_year_month_depart, load_iata
from streamlit_card import card
import altair as alt
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from components.sales import render_sales_summary
from components.passenger import render_passenger_summary
from components.flights import render_flights_summary

st.set_page_config(page_title="Dashboard", page_icon="🛫",
                   layout="wide")

# region Submenu (dalam halaman Dashboard.py)

sub_page = st.sidebar.radio(
    "Submenu Dashboard",
    ["Sales", "Passengers", "Flights"]
)

# Load data
if 'df' not in st.session_state:
    st.session_state.df = load_data()
    st.session_state.df_code = load_iata()
    df = st.session_state.df
    df_code = st.session_state.df_code
else:
    df = st.session_state.df
    df_code = st.session_state.df_code

df['Issued Date'] = pd.to_datetime(df['Issued Date'])
df['Segments/Departure Date'] = pd.to_datetime(df['Segments/Departure Date'])
start_year = df['Issued Date'].min().year
end_year = df['Issued Date'].max().year

# endregion submenu

# region subpage Sales
if sub_page == "Sales":

    render_sales_summary(df, start_year, end_year)
    
# endregion subpage Sales


# region subpage Passenger
elif sub_page == "Passengers":

    render_passenger_summary(df, start_year, end_year)

# endregion subpage Passenger


# region subpage Passenger
elif sub_page == 'Flights':
    
    render_flights_summary(df, start_year, end_year, df_code)

# endregion subpage Passenger


# # untuk name passenger
    # top_passenger = (
    #     df['Passenger/Display Name']
    #     .str.upper()
    #     .value_counts()
    #     .reset_index(name='Total Orders')
    #     .rename(columns={'index': 'Passenger/Display Name'})
    # )
    
    # elif option_customer == 'Passenger Order History':
    #     selected_passenger = st.selectbox(
    #         "Pilih Passenger:",
    #         top_passenger['Passenger/Display Name']
    #     )

    #     customer_history = (
    #         filtered_df[filtered_df['Passenger/Display Name'].str.upper() == selected_passenger]
    #         .sort_values(by='Booking Date', ascending=False)
    #         [[
    #             'Passenger/Display Name',
    #             'Booking Date',
    #             'Segments/Departure Date',
    #             'Segments/Arrival Date',
    #             'Sector',
    #             'Segments/Origin/Code',
    #             'Segments/Destination/Code',
    #         ]]
    #     )

    #     st.dataframe(customer_history)