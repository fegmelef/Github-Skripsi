import streamlit as st
import pandas as pd
from utils import load_data, load_iata
import os

st.set_page_config(page_title="Home", page_icon="🏠",
                   layout="wide")

df = load_data()
df_code = load_iata()

df = df.dropna(how='all')

st.session_state.df = df
st.session_state.df_code = df_code

st.title("Rodex Tours & Travel")
st.write("Data Loaded! Data yang ditampilkan merupakan data History Issued Ticket yang sudah diupload sebelumnya.")

st.button("🔄 Refresh Data")
st.dataframe(st.session_state.df, height=200, use_container_width=True)

upload_folder = 'files'

if not os.path.exists(upload_folder):
    os.mkdir(upload_folder)

uploaded_files = st.file_uploader(
    'Anda bisa upload file Issued Ticket maupun merge file (langsung upload lebih dari 1 file) yang ada di bawah ini.',
    type=['.xls', '.xlsx', '.csv'],
    accept_multiple_files=True,
    key="uploaded_files"
)

if uploaded_files:
    st.write(
        f"{len(uploaded_files)} file terupload. Tekan tombol di bawah untuk merge dan upload.")
    if st.button("Merge & Upload"):
        merged_dfs = []
        base_columns = None
        errors = []

        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name

            try:
                if file_name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
            except Exception as e:
                errors.append(f"{file_name} gagal dibaca: {str(e)}")
                continue

            if base_columns is None:
                base_columns = list(df.columns)
                merged_dfs.append(df)
            elif list(df.columns) == base_columns:
                merged_dfs.append(df)
            else:
                errors.append(
                    f"{file_name} memiliki struktur kolom yang berbeda dan gagal merge.")

        if errors:
            st.error("Penggabungan dibatalkan karena ada kesalahan pada file:")
            for e in errors:
                st.error(f"- {e}")

            merged_path = os.path.join(upload_folder, "history.xlsx")
            if os.path.exists(merged_path):
                df_history = pd.read_excel(merged_path)
                st.cache_data.clear()
                st.session_state.df = df_history
            else:
                st.error("File history.xlsx sebelumnya tidak ditemukan.")
        else:
            final_df = pd.concat(merged_dfs, ignore_index=True)
            merged_path = os.path.join(upload_folder, "history.xlsx")
            final_df.to_excel(merged_path, index=False)
            st.success(f"File berhasil dimerge dan disimpan sebagai: {merged_path}, silakan refresh data")
            st.cache_data.clear()
            st.session_state.df = final_df
