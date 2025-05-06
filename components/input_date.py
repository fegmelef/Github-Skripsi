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
import os
import holidays


def render_input_date(df, start_year, end_year):
    holiday_folder = 'files'
    holiday_file = os.path.join(holiday_folder, 'holidays.xlsx')

    if not os.path.exists(holiday_folder):
        os.mkdir(holiday_folder)

    countries = [
        "ID", "AF", "AL", "DZ", "AS", "AD", "AO", "AG", "AR", "AM", "AW", "AU", "AT", "AZ", "BS", "BH", "BD", 
        "BB", "BY", "BE", "BZ", "BO", "BA", "BW", "BR", "BN", "BG", "BF", "BI", "KH", "CM", "CA", "TD", "CL", 
        "CN", "CO", "CG", "CR", "HR", "CU", "CW", "CY", "CZ", "DK", "DJ", "DM", "DO", "EC", "EG", "SV", "EE", 
        "SZ", "ET", "FJ", "FI", "FR", "TF", "GA", "GE", "DE", "GH", "GR", "GL", "GU", "GT", "GG", "GN", "HT", 
        "HN", "HK", "HU", "IS", "IN", "IR", "IE", "IM", "IL", "IT", "CI", "JM", "JP", "JE", "JO", "KZ", "KE", 
        "KW", "KG", "LA", "LV", "LS", "LI", "LT", "LU", "MO", "MG", "MW", "MY", "MV", "MT", "MH", "MR", "MX", 
        "MD", "MC", "ME", "MA", "MZ", "NA", "NL", "NZ", "NI"
    ]

    country_code = st.selectbox("Pilih Negara", countries, index=countries.index("ID"))

    holiday_file = os.path.join(holiday_folder, f'holidays_{country_code}.xlsx')

    if os.path.exists(holiday_file):
        df_holiday = pd.read_excel(holiday_file)
    else:
        current_year = datetime.now().year
        country_holidays = holidays.CountryHoliday(country_code, years=range(2022, current_year + 1))

        date_range = pd.date_range(start="2022-01-01", end=datetime.today())
        df_holiday = pd.DataFrame(date_range, columns=["Date"])

        df_holiday["Holiday Name"] = df_holiday["Date"].apply(
            lambda x: country_holidays.get(x.strftime('%Y-%m-%d'), "")
        )

        df_holiday["Holiday Name"] = df_holiday.apply(
            lambda row: "Weekend" if row["Holiday Name"] == "" and row["Date"].weekday() in [5, 6] else row["Holiday Name"],
            axis=1
        )

        df_holiday["Libur"] = df_holiday["Holiday Name"].apply(lambda x: 1 if x != "" else 0)

        df_holiday.to_excel(holiday_file, index=False)


    df_holiday["Holiday Name"] = df_holiday["Holiday Name"].fillna("")

    selected_date = st.date_input(
        "Pilih tanggal", 
        min_value=datetime(2022, 1, 1), 
        max_value=datetime.today(), 
        value=datetime.today()
    )

    selected_date_ts = pd.to_datetime(selected_date)
    df_holiday["Date"] = pd.to_datetime(df_holiday["Date"])

    existing_name = ""
    if selected_date_ts in df_holiday["Date"].values:
        existing_name = df_holiday.loc[df_holiday["Date"] == selected_date_ts, "Holiday Name"].values[0]
    else:
        existing_name = "" 

    holiday_name = st.text_input("Masukkan nama hari libur (kosongkan untuk menghapus):", value=existing_name)

    col1, col2 = st.columns([4,6])

    with col1:
        if st.button("Simpan Hari Libur"):
            selected_date_ts = pd.to_datetime(selected_date)
            df_holiday["Date"] = pd.to_datetime(df_holiday["Date"])

            if selected_date_ts in df_holiday["Date"].values:
                if holiday_name.strip() == "":
                    df_holiday.loc[df_holiday["Date"] == selected_date_ts, ["Holiday Name", "Libur"]] = ["", 0]
                    st.warning(f"Hari libur dihapus: {selected_date}")
                else:
                    df_holiday.loc[df_holiday["Date"] == selected_date_ts, "Holiday Name"] = holiday_name
                    df_holiday.loc[df_holiday["Date"] == selected_date_ts, "Libur"] = 1
                    st.success(f"Hari libur diubah: {selected_date} - {holiday_name}")
            else:
                if holiday_name.strip() != "":
                    new_row = pd.DataFrame({"Date": [selected_date_ts], "Holiday Name": [holiday_name], "Libur": [1]})
                    df_holiday = pd.concat([df_holiday, new_row], ignore_index=True)
                    st.success(f"Hari libur ditambahkan: {selected_date} - {holiday_name}")
                else:
                    st.info("Tidak ada perubahan karena nama libur kosong dan tanggal tidak ditemukan.")

            df_holiday.to_excel(holiday_file, index=False)

    with col2:
        if st.button("Update Hari Libur Hingga Hari Ini"):
            current_year = datetime.now().year
            country_holidays_new = holidays.CountryHoliday(country_code, years=range(2022, current_year + 1))

            date_range_new = pd.date_range(start="2022-01-01", end=datetime.today())
            df_holiday_new = pd.DataFrame(date_range_new, columns=["Date"])

            df_holiday_new["Holiday Name"] = df_holiday_new["Date"].apply(lambda x: country_holidays_new.get(x.strftime('%Y-%m-%d'), ""))

            df_holiday_new["Libur"] = df_holiday_new["Holiday Name"].apply(lambda x: 1 if x != "" else 0)

            df_holiday_combined = pd.concat([df_holiday, df_holiday_new]).drop_duplicates(subset=["Date"]).reset_index(drop=True)

            df_holiday_combined.to_excel(holiday_file, index=False)

            st.success("Data hari libur berhasil diperbarui hingga hari ini.")

    st.subheader("📅 Daftar Hari Libur")
    st.dataframe(df_holiday.sort_values("Date").reset_index(drop=True))
