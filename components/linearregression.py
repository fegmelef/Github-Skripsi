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
# import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error


def render_linear_regression(df, start_year, end_year):
    col1, col2 = st.columns(2)
    with col1:
        buffer_days = st.number_input("Enter the number of buffer days (0–7):", min_value=0, max_value=7, value=1)
    with col2:
        buffer_option = st.radio(
            "Buffer Day Type:",
            options=["Before", "After", "Before and After"],
            index=2,
            horizontal=True
        )

    aggregation_level = st.selectbox("Aggregation Level", options=["Daily", "Weekly", "Monthly"])

    # df = pd.read_excel("files/data_dummy.xlsx")
    df_holiday = pd.read_excel("files/holidays_ID.xlsx")
    df_holiday['Date'] = pd.to_datetime(df_holiday['Date'])
    df_holiday = df_holiday[['Date', 'Libur']].rename(columns={'Date': 'Segments/Departure Date', 'Libur': 'holiday'})

    df['Segments/Departure Date'] = pd.to_datetime(df['Segments/Departure Date'])
    df['Booking Date'] = pd.to_datetime(df['Booking Date'])

    dftest = df.dropna(subset=['Segments/Departure Date'])

    df_daily = dftest.groupby(dftest['Segments/Departure Date'].dt.date)['Total Pax'].sum().reset_index()
    df_daily['Segments/Departure Date'] = pd.to_datetime(df_daily['Segments/Departure Date'])

    full_dates = pd.DataFrame({
        'Segments/Departure Date': pd.date_range(start=df_daily['Segments/Departure Date'].min(),
                                                end=df_daily['Segments/Departure Date'].max())
    })
    df_daily = full_dates.merge(df_daily, on='Segments/Departure Date', how='left')
    df_daily['Total Pax'] = df_daily['Total Pax'].fillna(0).astype(int)

    calendar = pd.DataFrame({'Segments/Departure Date': pd.date_range(start=df_daily['Segments/Departure Date'].min(),
                                                                    end=df_daily['Segments/Departure Date'].max())})
    calendar = calendar.merge(df_holiday, on='Segments/Departure Date', how='left')
    calendar['holiday'] = calendar['holiday'].fillna(0).astype(int)

    calendar['grp'] = (calendar['holiday'] != calendar['holiday'].shift()).cumsum()
    calendar['seq'] = calendar.groupby('grp').cumcount()
    holiday_groups = calendar.groupby('grp').agg({
        'holiday': 'sum',
        'Segments/Departure Date': ['first', 'last']
    })
    holiday_groups.columns = ['count', 'start', 'end']
    long_holidays = holiday_groups[holiday_groups['count'] >= 3]

    buffer_dates = set()
    for _, row in long_holidays.iterrows():
        if buffer_option in ["Before", "Before and After"]:
            for i in range(1, buffer_days + 1):
                buffer_dates.add(row['start'] - pd.Timedelta(days=i))
        if buffer_option in ["After", "Before and After"]:
            for i in range(1, buffer_days + 1):
                buffer_dates.add(row['end'] + pd.Timedelta(days=i))

    calendar['holiday'] = calendar['Segments/Departure Date'].isin(
        calendar[calendar['holiday'] == 1]['Segments/Departure Date'].tolist() + list(buffer_dates)
    ).astype(int)

    df_daily = df_daily.drop(columns='holiday', errors='ignore')
    df_daily = df_daily.merge(calendar[['Segments/Departure Date', 'holiday']], on='Segments/Departure Date', how='left')

    if aggregation_level == "Weekly":
        df_daily['Week'] = df_daily['Segments/Departure Date'].dt.to_period('W-MON').dt.start_time
        df_final = df_daily.groupby('Week').agg({
            'Total Pax': 'sum',
            'holiday': 'sum'
        }).reset_index()
        df_final['weekday'] = df_final['Week'].dt.weekday
        df_final['is_weekend'] = 0
        df_final['month'] = df_final['Week'].dt.month
        df_final['day_of_month'] = df_final['Week'].dt.day
        df_final = df_final.rename(columns={'Week': 'Date'})
    elif aggregation_level == "Monthly":
        df_daily['Month'] = df_daily['Segments/Departure Date'].dt.to_period('M').dt.to_timestamp()
        df_final = df_daily.groupby('Month').agg({
            'Total Pax': 'sum',
            'holiday': 'sum'
        }).reset_index()
        df_final['weekday'] = df_final['Month'].dt.weekday
        df_final['is_weekend'] = 0
        df_final['month'] = df_final['Month'].dt.month
        df_final['day_of_month'] = df_final['Month'].dt.day
        df_final = df_final.rename(columns={'Month': 'Date'})
    else:
        df_final = df_daily.copy()
        df_final = df_final.rename(columns={'Segments/Departure Date': 'Date'})
        df_final['weekday'] = df_final['Date'].dt.weekday
        df_final['is_weekend'] = df_final['weekday'].isin([5, 6]).astype(int)
        df_final['month'] = df_final['Date'].dt.month
        df_final['day_of_month'] = df_final['Date'].dt.day

    df_final['lag_1'] = df_final['Total Pax'].shift(1)
    df_final['lag_7'] = df_final['Total Pax'].shift(7)
    df_final['rolling_mean_3'] = df_final['Total Pax'].rolling(3).mean()
    df_final['rolling_mean_7'] = df_final['Total Pax'].rolling(7).mean()

    df_final.dropna(inplace=True)
    df_final = df_final[df_final['Total Pax'] != 0]
    df_final.set_index('Date', inplace=True)

    Feature = ['holiday', 'weekday', 'is_weekend', 'month', 'day_of_month',
            'lag_1', 'lag_7', 'rolling_mean_3', 'rolling_mean_7']

    X = df_final[Feature]
    y = df_final['Total Pax']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    reg = LinearRegression()
    reg.fit(X_train_scaled, y_train_scaled)

    y_pred_scaled = reg.predict(X_test_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    r2 = r2_score(y_test, y_pred)

    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': reg.coef_
    })
    coef_df['Category'] = coef_df['Coefficient'].apply(
        lambda x: 'Significant' if abs(x) >= 0.5 else 'Not Significant'
    )

    def interpret_coefficient(row):
        feature = row['Feature']
        coef = row['Coefficient']
        if feature == 'holiday':
            return f"If the day is a holiday (1), the number of passengers tends to {'increase' if coef > 0 else 'decrease'} by ~{abs(coef):.2f} pax."
        elif feature == 'weekday':
            return f"For each increase in weekday (e.g., Monday to Tuesday), passengers {'increase' if coef > 0 else 'decrease'} by ~{abs(coef):.2f}."
        elif feature == 'is_weekend':
            return f"If it is the weekend, passengers {'increase' if coef > 0 else 'decrease'} by ~{abs(coef):.2f} compared to weekdays."
        elif feature == 'month':
            return f"For each increase in month, passengers {'increase' if coef > 0 else 'decrease'} slightly (~{abs(coef):.3f})."
        elif feature == 'day_of_month':
            return f"Later in the month, passengers tend to {'increase' if coef > 0 else 'decrease'} by ~{abs(coef):.3f}."
        elif feature == 'lag_1':
            return f"If yesterday's number was high, today's tends to {'increase' if coef > 0 else 'decrease'} by ~{abs(coef):.2f}."
        elif feature == 'lag_7':
            return f"If last week's value was high, today tends to {'increase' if coef > 0 else 'decrease'} slightly (~{abs(coef):.3f})."
        elif feature == 'rolling_mean_3':
            return f"If the 3-day average was high, today's passengers tend to {'increase' if coef > 0 else 'decrease'} by ~{abs(coef):.2f}."
        elif feature == 'rolling_mean_7':
            return f"If the 7-day average was high, today's passengers tend to {'increase' if coef > 0 else 'decrease'} by ~{abs(coef):.2f}."
        else:
            return f"General effect on Total Pax: {'positive' if coef > 0 else 'negative'} ~{abs(coef):.2f}"

    coef_df['Interpretation'] = coef_df.apply(interpret_coefficient, axis=1)
    coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)

    def highlight_extremes(s):
        if s.name == 'Coefficient':
            threshold = 0.5
            return [
                'background-color: lightgreen' if v > threshold
                else 'background-color: lightcoral' if v < -threshold
                else ''
                for v in s
            ]
        return ['' for _ in s]

    st.subheader('Coefficient and Interpretation')
    st.dataframe(coef_df.style.apply(highlight_extremes, axis=0))

    index_name = df_final.index.name
    result_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    }, index=X_test.index).sort_index()


    result_df.index = result_df.index.strftime('%Y-%m-%d')
    st.subheader(f'Comparison of Actual vs Predicted ({aggregation_level})')
    st.line_chart(result_df)

    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**MSE:** {mse:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAPE:** {mape:.2f}%")
    st.write(f"**R² Score:** {r2:.2f}")
    st.write(f"---------")
    st.write("Glossary:")

    st.markdown("""
    #### Linear Regression

    A statistical method used to model the relationship between a target variable (in this case, the number of passengers) and one or more predictor variables (features).

    #### Features

    Features are the variables used by the model to predict the target.

    #### Coefficients

    Coefficients indicate how much influence each feature has on the target variable.  
    - A positive coefficient means the feature increases the target value.  
    - A negative coefficient means the feature decreases the target value.  
    - The magnitude of the coefficient indicates the strength of the feature's influence.

    #### R-squared (Coefficient of Determination)

    R-squared is a measure of how well the model explains the variability of the target data.  
    - R-squared values range from 0 to 1.  
    - The closer the value is to 1, the better the model predicts the data.  
    """)
