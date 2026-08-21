# Day 10 - Energy Consumption Forecasting

# pip install statsmodels matplotlib pandas numpy prophet pmdarima
# pip install --force-reinstall numpy==2.0.0 #if you get an error with statsmodels
# ------------------ Synthetic Data ------------------
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate dates
dates = pd.date_range(start='1/1/2020', periods=100)

# Generate consumption data with a trend and seasonality
trend = np.linspace(100, 500, 100)  # Linear trend
seasonality = 50 * np.sin(np.linspace(0, 3 * np.pi, 100))  # Seasonal pattern
noise = np.random.normal(0, 20, 100)  # Random noise

# Combine trend, seasonality, and noise
consumption = trend + seasonality + noise
consumption = np.clip(consumption, 100, 500)  # Ensure values are within a realistic range

# Create DataFrame
data = pd.DataFrame({'Date': dates, 'Consumption': consumption})
print("Synthetic energy consumption data with real-world patterns created.")



# ------------------ Plot ------------------
import matplotlib.pyplot as plt

plt.plot(data['Date'], data['Consumption'])
plt.xlabel('Date')
plt.ylabel('Consumption')
plt.title('Energy Consumption')
plt.show()

# ------------------ Energy Consumption Forecasting ------------------

import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Fit the ARIMA model
model = ARIMA(data['Consumption'], order=(5, 1, 0))
model_fit = model.fit()

# Forecast the next 10 steps
forecast = model_fit.forecast(steps=10)


# ------------------ Plot ------------------
plt.plot(data['Date'], data['Consumption'], label='Consumption')
plt.plot(pd.date_range(start=data['Date'].iloc[-1], periods=10, freq='D'), forecast, label='Forecast')
plt.xlabel('Date')
plt.ylabel('Energy Consumption')
plt.title('Energy Consumption Forecast')
plt.legend()
plt.show()


# ------------------ Improved Model ------------------
# The Exponential Smoothing model is an improvement over the ARIMA model for this specific dataset
# because it can handle both trend and seasonality in the data.
# The ARIMA model used previously was configured with a specific order that may not capture the
# seasonal patterns in the energy consumption data.
# The Exponential Smoothing model with additive trend and seasonal components is more suitable
# for datasets with clear seasonal patterns, which is common in energy consumption data.

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Fit the Exponential Smoothing model
model = ExponentialSmoothing(data['Consumption'], trend='add', seasonal='add', seasonal_periods=12)
model_fit = model.fit()

# Forecast the next 10 steps
forecast = model_fit.forecast(steps=10)

# Plot the actual consumption and the forecasted values
plt.plot(data['Date'], data['Consumption'], label='Consumption')
plt.plot(pd.date_range(start=data['Date'].iloc[-1], periods=10, freq='D'), forecast, label='Forecast')
plt.xlabel('Date')
plt.ylabel('Energy Consumption')
plt.title('Energy Consumption Forecast')
plt.legend()
plt.show()



# ------------------ Forecasting with Alternative models ------------------
import numpy as np
np.float_ = np.float64
from prophet import Prophet

data = pd.DataFrame({'ds': dates, 'y': consumption})
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

def plot_forecast(true_values, forecast, title):
    plt.figure(figsize=(12, 6))
    plt.plot(true_values.index, true_values, label='Actual')
    plt.plot(forecast.index, forecast, label='Forecast')
    plt.title(title)
    plt.legend()
    plt.show()

# 3. Prophet
prophet_model = Prophet()
prophet_model.fit(train)
future = prophet_model.make_future_dataframe(periods=len(test))
prophet_forecast = prophet_model.predict(future)
plot_forecast(test['y'], prophet_forecast.iloc[-len(test):]['yhat'], 'Prophet Forecast')

prophet_model.plot_components(prophet_forecast)
plt.show()
