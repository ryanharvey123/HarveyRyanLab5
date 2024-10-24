import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Load your data from the CSV file
# Replace 'expenses.data.csv' with the correct path to your file if needed
data_6_years = pd.read_csv('extended_expenses_data.csv')

# Ensure 'Date' column is in datetime format and set as index
data_6_years['Date'] = pd.to_datetime(data_6_years['Date'])
data_6_years.set_index('Date', inplace=True)

# ARIMAX model function
def arimax_forecast(train_expenses, train_exog, test_exog, order=(1, 1, 1)):
    model = SARIMAX(train_expenses, exog=train_exog, order=order)
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=len(test_exog), exog=test_exog)
    return forecast

# Streamlit App layout
st.title("ARIMAX Model for Expense Forecasting - Ryan Harvey")
st.write("This app allows you to adjust GDP and CPI to see how they affect future expense forecasts.")

# Sidebar for user input
st.sidebar.header("Adjust External Variables")

# Let the user adjust GDP and CPI for future forecast
future_gdp = st.sidebar.slider('Future GDP', 20000, 25000, int(data_6_years['GDP'].iloc[-1]))
future_cpi = st.sidebar.slider('Future CPI', 250, 300, int(data_6_years['CPI'].iloc[-1]))

# Split the data into train and test
train_size = int(len(data_6_years) * 0.8)
train_expenses = data_6_years['Monthly_Expenses'][:train_size]
train_exog = data_6_years[['GDP', 'CPI']][:train_size]
test_exog = data_6_years[['GDP', 'CPI']][train_size:]

# Adjust last known exogenous values (GDP, CPI) for forecasting
test_exog.loc[:, 'GDP'] = future_gdp
test_exog.loc[:, 'CPI'] = future_cpi

# Forecast
forecast = arimax_forecast(train_expenses, train_exog, test_exog)

# Plot the results
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_expenses.index, train_expenses, label='Training Expenses')
ax.plot(test_exog.index, forecast, label='Forecasted Expenses', linestyle='--')
ax.set_title('ARIMAX Model: Expense Forecasting with Adjusted GDP and CPI')
ax.set_xlabel('Date')
ax.set_ylabel('Monthly Expenses')
ax.legend()

# Display the plot in the app
st.pyplot(fig)

# Display the forecast values
st.write("Forecasted Expenses for the next period based on adjusted GDP and CPI:")
st.dataframe(forecast)
