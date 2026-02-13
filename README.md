# Time Series Forecasting: Decomposition & ARIMA (Ice Cream Sales Example)

This mini-project demonstrates end-to-end **time series analysis and forecasting** in Python using a simple, intuitive example: monthly ice cream sales.

It is designed for beginners and interview prep, but follows good practices that apply to real-world use cases like **financial transaction volumes, fraud detection, and demand forecasting**.

## What This Project Shows

1. **Synthetic Time Series Generation**
   - 36 months of monthly sales data
   - Components:
     - Long-term **trend** (steady growth over time)
     - **Seasonality** (higher sales in “summer” months)
     - Random **noise** (unexpected ups/downs)

2. **Visualization**
   - Line plot of sales over time to visually inspect:
     - Upward trend
     - Seasonal peaks
     - Random fluctuations

3. **Classical Time Series Decomposition**
   - Using `statsmodels.tsa.seasonal.seasonal_decompose` to split the series into:
     - Trend
     - Seasonal component
     - Residuals (noise)
   - Helps clearly **see** each component separately.

4. **ARIMA Forecasting**
   - Fit an `ARIMA(1,1,1)` model on the sales data.
   - Forecast the next 12 months.
   - Plot:
     - Historical sales
     - Forecasted values
     - Confidence intervals (uncertainty band)

## File Structure

- `timeseries_example.py`  
  Main script that:
  - Generates the synthetic time series
  - Plots the original data
  - Performs decomposition
  - Fits an ARIMA model and forecasts future values

## How to Run

```bash
# Clone the repo
git clone https://github.com/Syamchand-Chilaka/timeseries_forecasting.git
cd timeseries_forecasting

# (Optional) Create and activate a virtual environment

# Install dependencies
pip install pandas numpy matplotlib statsmodels

# Run the example
python timeseries_example.py
