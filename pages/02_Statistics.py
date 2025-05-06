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
from components.input_date import render_input_date
from components.ttest import render_ttest
from components.linearregression import render_linear_regression


st.set_page_config(page_title="Statistics", page_icon="📈", layout="wide")

sub_page = st.sidebar.radio(
    "Submenu Statistics",
    ["Input Date Data", "T-Test", "Linear Regression"]
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

# region subpage Sales
if sub_page == "Input Date Data":

    render_input_date(df, start_year, end_year)
    
# endregion subpage Sales


# region subpage Passenger
elif sub_page == "T-Test":

    render_ttest(df, start_year, end_year)

# endregion subpage Passenger


# region subpage Passenger
elif sub_page == 'Linear Regression':
    
    render_linear_regression(df, start_year, end_year)

# endregion subpage Passenger