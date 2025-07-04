# Stock Price Forecasting with ARIMA-LSTM Hybrid Model

This project implements a hybrid model combining ARIMA and LSTM for forecasting stock prices. It leverages time series analysis with ARIMA to capture linear dependencies and LSTM to model non-linear patterns in the stock price data.

## Features

-   **Data Fetching:** Fetches stock price data from Yahoo Finance using `yfinance`.
-   **Data Preprocessing:**
    -      Splits data into training and testing sets.
    -      Performs Augmented Dickey-Fuller (ADF) test for stationarity.
    -      Applies differencing to make the time series stationary.
    -      Scales data for LSTM using MinMaxScaler.
-   **ARIMA Modeling:**
    -      Fits an ARIMA model to the stationary time series.
    -      Extracts residuals from the ARIMA model for LSTM input.
-   **LSTM Modeling:**
    -      Trains an LSTM model on the residuals from the ARIMA model.
    -      Generates LSTM predictions.
-   **Hybrid Forecasting:**
    -      Combines ARIMA forecasts with LSTM predictions to produce final forecasts.
-   **Model Evaluation:**
    -      Calculates Mean Absolute Percentage Error (MAPE) and Root Mean Squared Error (RMSE) to evaluate model performance.
-   **Visualization:**
    -      Plots actual stock prices and hybrid forecasts.

## Technologies

-   **Python:** Core programming language.
-   **Pandas:** Data manipulation and analysis.
-   **NumPy:** Numerical computations.
-   **YFinance:** Stock data retrieval.
-   **Statsmodels:** Time series analysis (ARIMA, ADF).
-   **Scikit-learn:** Data preprocessing (MinMaxScaler).
-   **TensorFlow/Keras:** LSTM model implementation.
-   **Matplotlib/Seaborn:** Data visualization.

## Setup

1.  **Clone the Repository:**

    ```bash
    git clone [repository URL]
    cd [repository directory]
    ```

2.  **Install Dependencies:**

    ```bash
    pip install pandas numpy yfinance statsmodels scikit-learn tensorflow matplotlib seaborn
    ```

3.  **Run the Script:**

    ```bash
    python your_script_name.py
    ```

    Replace `your_script_name.py` with the actual name of your Python script.

## Usage

1.  **Modify Stock Ticker and Date Range:**
    -   In the `Task 5: Main Execution` section, change the `ticker`, `start_date`, and `end_date` variables to fetch data for your desired stock and time period.

    ```python
    ticker = 'TCS.NS'  # Replace with your stock ticker
    start_date = '2014-01-01'
    end_date = '2024-01-01'
    ```

2.  **Run the Script:**
    -   Execute the Python script.
3.  **View Results:**
    -   The script will output the MAPE and RMSE values and display a plot showing the actual stock prices and the hybrid forecasts.

## Code Explanation

-   **`fetch_stock_data(ticker, start, end)`:** Fetches stock data from Yahoo Finance.
-   **`train_test_split(series, split_ratio=0.8)`:** Splits data into training and testing sets.
-   **`adf_test(series)`:** Performs the ADF test for stationarity.
-   **`make_stationary(series)`:** Applies differencing to make the series stationary.
-   **`fit_arima(series)`:** Fits an ARIMA model to the data.
-   **`prepare_lstm_data(residuals, time_steps=10)`:** Prepares data for LSTM training.
-   **`build_lstm_model(input_shape)`:** Builds and compiles the LSTM model.
-   **`forecast_hybrid(series, test_size, lstm_epochs=10, time_steps=10)`:** Generates hybrid forecasts.
-   **`evaluate_model(actual, predicted)`:** Evaluates the model using MAPE and RMSE.

## Notes

-   The ARIMA model's `p`, `d`, and `q` parameters are determined through a grid search within the `fit_arima` function. You may adjust the ranges of these parameters.
-   The LSTM model's architecture and hyperparameters can be tuned for better performance.
-   The `time_steps` variable in `prepare_lstm_data` and `forecast_hybrid` determines the number of previous time steps used as input for the LSTM model.
-   The `lstm_epochs` variable in `forecast_hybrid` sets the number of training epochs for the LSTM model.

## Future Improvements

-   Implement hyperparameter tuning for the LSTM model.
-   Add support for other evaluation metrics.
-   Incorporate external factors (e.g., news sentiment) into the model.
-   Develop a web interface for easy interaction with the model.
