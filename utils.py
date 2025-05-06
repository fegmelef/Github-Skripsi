import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_excel('files/history.xlsx')
    drop_columns = [
        'External ID',
        'Agent/ID',
        'Agent Type/ID',
        'Booker/ID',
        'Customer/ID',
        'Head Office/ID',
        'Issued by/ID',
        'PNR',
        'Updated by/ID',
        'Itinerary/Display Name',
        # 'Origin/Display Name',
        'Origin/Name',
        'Origin/City',
        'Destination/City',
        'Destination/Country/ID',
        'Segments/Destination/Carrier',
        'Segments/Destination/Display Name',
        'Segments/Origin/Carrier',
        'Segments/Plane/ID',
        'Segments/Plane/Display Name',
        'Segments/Origin/Display Name',
        'Segments/Carrier Type/Display Name',
        'Journeys/Departure Date',
        'Journeys/Arrival Date',
        'List of Carriers',
        'List of Provider',
        # 'Destination/Display Name',
        'Segments/Provider/Code',
        'Origin/Country/Country Name',
        'Segments/Carrier Type/Code',
    ]
    df = df.drop(columns=[col for col in drop_columns if col in df.columns])
    return df


def load_iata():
    df_code = pd.read_csv('files/airport-codes.csv')
    return df_code


def filter_year_month(prefix: str, df: pd.DataFrame):
    # Reset session state when entering a subpage
    if f"{prefix}_reset" not in st.session_state:
        st.session_state.clear()  # Clear session state to reset previous selections
        # Flag to ensure reset happens only once
        st.session_state[f"{prefix}_reset"] = True

    df = df[df['Issued Date'].notna()]
    df['Issued Year'] = df['Issued Date'].dt.year
    df['Issued Month'] = df['Issued Date'].dt.month

    available_years = sorted(df['Issued Year'].unique())
    available_months = sorted(df['Issued Month'].unique())

    all_years_key = f"{prefix}_all_years"
    selected_years_key = f"{prefix}_selected_years"
    all_months_key = f"{prefix}_all_months"
    selected_months_key = f"{prefix}_selected_months"

    if all_years_key not in st.session_state:
        st.session_state[all_years_key] = True
        st.session_state[selected_years_key] = available_years

    if all_months_key not in st.session_state:
        st.session_state[all_months_key] = True
        st.session_state[selected_months_key] = available_months

    def year_check_change():
        if st.session_state[all_years_key]:
            st.session_state[selected_years_key] = available_years
        else:
            st.session_state[selected_years_key] = []

    def year_multi_change():
        if len(st.session_state[selected_years_key]) == len(available_years):
            st.session_state[all_years_key] = True
        else:
            st.session_state[all_years_key] = False

    def month_check_change():
        if st.session_state[all_months_key]:
            st.session_state[selected_months_key] = available_months
        else:
            st.session_state[selected_months_key] = []

    def month_multi_change():
        if len(st.session_state[selected_months_key]) == len(available_months):
            st.session_state[all_months_key] = True
        else:
            st.session_state[all_months_key] = False

    col1, col2 = st.columns([3, 7])

    with col1:
        st.checkbox("All Years", key=all_years_key,
                    on_change=year_check_change)
        selected_years = st.multiselect(
            "Pilih Tahun (Issued Date)",
            options=available_years,
            key=selected_years_key,
            on_change=year_multi_change
        )
        if not st.session_state[all_years_key] and len(selected_years) == 0:
            st.error("Harap pilih minimal satu tahun.")

    with col2:
        st.checkbox("All Months", key=all_months_key,
                    on_change=month_check_change)
        selected_months = st.multiselect(
            "Pilih Bulan (Issued Date)",
            options=available_months,
            key=selected_months_key,
            on_change=month_multi_change
        )
        if not st.session_state[all_months_key] and len(selected_months) == 0:
            st.error("Harap pilih minimal satu bulan.")

    if len(selected_years) > 0 and len(selected_months) > 0:
        filtered_df = df[
            (df['Issued Year'].isin(selected_years)) &
            (df['Issued Month'].isin(selected_months))
        ]
    else:
        filtered_df = df

    return filtered_df


def filter_year_month_depart(prefix: str, df: pd.DataFrame):
    # Reset session state when entering a subpage
    if f"{prefix}_reset" not in st.session_state:
        st.session_state.clear()  # Clear session state to reset previous selections
        # Flag to ensure reset happens only once
        st.session_state[f"{prefix}_reset"] = True

    df_depart = df[df['Segments/Departure Date'].notna()]
    df_depart['Depart Year'] = df_depart['Segments/Departure Date'].dt.year
    df_depart['Depart Month'] = df_depart['Segments/Departure Date'].dt.month

    available_years = sorted(df_depart['Depart Year'].unique())
    available_months = sorted(df_depart['Depart Month'].unique())

    all_years_key = f"{prefix}_all_years"
    selected_years_key = f"{prefix}_selected_years"
    all_months_key = f"{prefix}_all_months"
    selected_months_key = f"{prefix}_selected_months"

    if all_years_key not in st.session_state:
        st.session_state[all_years_key] = True
        st.session_state[selected_years_key] = available_years

    if all_months_key not in st.session_state:
        st.session_state[all_months_key] = True
        st.session_state[selected_months_key] = available_months

    def year_check_change():
        if st.session_state[all_years_key]:
            st.session_state[selected_years_key] = available_years
        else:
            st.session_state[selected_years_key] = []

    def year_multi_change():
        if len(st.session_state[selected_years_key]) == len(available_years):
            st.session_state[all_years_key] = True
        else:
            st.session_state[all_years_key] = False

    def month_check_change():
        if st.session_state[all_months_key]:
            st.session_state[selected_months_key] = available_months
        else:
            st.session_state[selected_months_key] = []

    def month_multi_change():
        if len(st.session_state[selected_months_key]) == len(available_months):
            st.session_state[all_months_key] = True
        else:
            st.session_state[all_months_key] = False

    col1, col2 = st.columns([3, 7])

    with col1:
        st.checkbox("All Years", key=all_years_key,
                    on_change=year_check_change)
        selected_years = st.multiselect(
            "Pilih Tahun (Departure Date)",
            options=available_years,
            key=selected_years_key,
            on_change=year_multi_change
        )
        if not st.session_state[all_years_key] and len(selected_years) == 0:
            st.error("Harap pilih minimal satu tahun.")

    with col2:
        st.checkbox("All Months", key=all_months_key,
                    on_change=month_check_change)
        selected_months = st.multiselect(
            "Pilih Bulan (Departure Date)",
            options=available_months,
            key=selected_months_key,
            on_change=month_multi_change
        )
        if not st.session_state[all_months_key] and len(selected_months) == 0:
            st.error("Harap pilih minimal satu bulan.")

    if len(selected_years) > 0 and len(selected_months) > 0:
        filtered_df = df_depart[
            (df_depart['Depart Year'].isin(selected_years)) &
            (df_depart['Depart Month'].isin(selected_months))
        ]
    else:
        filtered_df = df

    return filtered_df

