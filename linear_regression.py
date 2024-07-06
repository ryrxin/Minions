import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ctf metrics calciation
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mse, rmse, mae, mape

# 1. slr training function
def slr_train_model(X_train, y_train):
  regressor = LinearRegression()
  slr_model = regressor.fit(X_train, y_train)
  return slr_model

# 2. slr testing function
def slr_next_day_pred(model, X_test):
  slr_y_pred = model.predict(X_test)
  return slr_y_pred

num_lags=1 ## NUM OF LAGS VARIABLE
days_predict = 365 ## t-TH day predicted

# 3. slr execute function
def slr_cumulative_days_pctchange(df, tickers, num_lags, days_predict):
    master_df = (df.pct_change() + 1)[1:].dropna()
    # prep DICTIONARY // not tuple beware later
    models = {}
    predictions = {}
    metrics = {}

    for ticker in tickers:
        X = master_df[ticker].values
        y = master_df[ticker].values  # this is the price I want to predict

        # lag variables
        X_lagged = np.array([X[i - num_lags:i] for i in range(num_lags, len(X))])  # creating lagged features
        y_lagged = y[num_lags:]

        # training sets
        X_train = X_lagged
        y_train = y_lagged
        # testing set
        X_test = X[-num_lags:].reshape(1, -1)

        # array to store predictions
        predictions[ticker] = []

        for day in range(days_predict):
            model = slr_train_model(X_train, y_train)
            models[ticker] = model

            # predicting next day
            y_pred = slr_next_day_pred(model, X_test)

            # store predictions
            predictions[ticker].append(float(y_pred[0]))  # Convert y_pred to float for JSON serialization

            # shifting x test array for new day
            X_test = np.roll(X_test, -1)
            X_test[0, -1] = float(y_pred[0])  # set last value to predicted value

            X_train = np.vstack([X_train, X_test[0]])
            y_train = np.append(y_train, y_pred)

        # calculating ctf metrics
        mse, rmse, mae, mape = calculate_metrics(y[-1:], y_pred)
        metrics[ticker] = {
            'MSE': float(mse),  # Convert metrics to float for JSON serialization
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MAPE': float(mape)
        }

    return predictions, metrics
