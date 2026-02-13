from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Here we created fake monthly sales data (36 months, upward trend + summer peaks + noise)
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=36, freq='ME')
sales = (10000 + 500 * np.arange(36) + 5000 * np.sin(2 * np.pi *
         np.arange(36) / 12) + np.random.normal(0, 1000, 36))

df = pd.DataFrame({'sales': sales}, index=dates)
print(df.head())

#  Line Plot (your #1: trends/cycles easy)

plt.figure(figsize=(10, 4))
plt.plot(df.index, df["sales"])
plt.title("Ice Cream Sales: Line Plot (Trend + Seasonality Visible)")
plt.ylabel("Sales $")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#  Decomposition Plot (your #6: trend/seasonal/resid)

decomp = seasonal_decompose(df['sales'], model="additive", period=12)
decomp.plot()
plt.suptitle("Decomposition: See All Components!", fontsize=14)
plt.tight_layout()
plt.show()

print("\nTrend sample:\n", decomp.trend.dropna().head())

# NEW: Plot ingredients separately!

trend_part = 10000 + 500 * np.arange(36)
seasonal_part = 5000 * np.sin(2 * np.pi * np.arange(36) / 12)

plt.figure(figsize=(10, 5))
plt.plot(trend_part, label="Trend (Base + Growth)", linewidth=3)
plt.plot(seasonal_part, label="seasonal Wave", linewidth=2)
plt.plot(np.zeros(36), 'k--', alpha=0.5, label="Zero Line")
plt.title("Recipe Ingredients: Trend vs Seasonal")
plt.ylabel("Sales $")
plt.legend()
plt.show()

model = ARIMA(df["sales"], order=(1, 1, 1))  # p,q,d = s/m/l memory

fitted = model.fit()

# Forecast for the next 12 Months
forecast = fitted.forecast(steps=12)
forecast_ci = fitted.get_forecast(steps=12).con_int()

plt.figure(figsize=(10, 5))
plt.plot(df.index, df["sales"], label="Real Past")
plt.plot(forecast.index, forecast, label="ARIMA Prediction", linewidth=2)
plt.fill_between(forecast_ci.index,
                 forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], alpha=0.3)
plt.title("Future Ice Cream Sales Prediction!")
plt.legend()
plt.show()
