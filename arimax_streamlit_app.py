
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Load your data (replace with your dataset)
# For simplicity, we'll generate some random data similar to what you already have

np.random.seed(42)
months_6_years = pd.date_range(start="2018-01-01", periods=72, freq='M')

expenses_6_years = np.random.normal(loc=2000, scale=200, size=len(months_6_years)) + np.linspace(0, 750, len(months_6_years))
gdp_6_years = np.random.normal(loc=21000, scale=300, size=len(months_6_years))
cpi_6_years = np.random.normal(loc=260, scale=5, size=len(months_6_years)) + np.linspace(0, 30, len(months_6_years))

data_6_years = pd.DataFrame({
    'Date': months_6_years,
    'Monthly_Expenses': expenses_6_years,
    'GDP': gdp_6_years,
    'CPI': cpi_6_years
})

data_6_years.set_index('Date', inplace=True)

# ARIMAX model function
def arimax_forecast(train_expenses, train_exog, test_exog, order=(1, 1, 1)):
    model = SARIMAX(train_expenses, exog=train_exog, order=order)
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=len(test_exog), exog=test_exog)
    return forecast

# Streamlit App layout
st.title("ARIMAX Model for Expense Forecasting")
st.write("This app allows you to adjust GDP and CPI to see how they affect future expense forecasts.")

# Sidebar for user input
st.sidebar.header("Adjust External Variables")

# Let the user adjust GDP and CPI for future forecast
future_gdp = st.sidebar.slider('Future GDP', 20000, 25000, int(gdp_6_years[-1]))
future_cpi = st.sidebar.slider('Future CPI', 250, 300, int(cpi_6_years[-1]))

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
