from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# ctf metrics calculation
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100        
    return mse, rmse, mae, mape

# 1. xgb training function
def xgb_train_model(X_train, y_train):
    xgb_model = XGBRegressor(n_estimators=100)
    xgb_model.fit(X_train, y_train)
    return xgb_model

# 2. xgb prediction function
def xgb_next_day_pred(model, X_test):
    xgb_y_pred = model.predict(X_test)
    return xgb_y_pred

# 3. xgb execution function
def xgb_predict_days_predict(df, tickers, num_lags, days_predict):
    master_df = (df.pct_change() + 1)[1:].dropna()
    predictions = {}
    metrics = {}

    for ticker in tickers:
        X = master_df[ticker].values
        y = master_df[ticker].values

        # lagged variables
        X_lagged = np.array([X[i - num_lags:i] for i in range(num_lags, len(X))])
        y_lagged = y[num_lags:]

        X_train = X_lagged
        y_train = y_lagged

        model = xgb_train_model(X_train, y_train)

        predictions[ticker] = []

        for i in range(days_predict):

            X_pred = X[-num_lags:].reshape(1, -1)

            y_pred = xgb_next_day_pred(model, X_pred)
            predictions[ticker].append(y_pred[0])

            X = np.append(X, y_pred)

        # ctf for the entire forecast horizon
        y_true = master_df[ticker].values[-days_predict:]
        mse, rmse, mae, mape = calculate_metrics(y_true, predictions[ticker])
        metrics[ticker] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }

    return predictions, metrics
