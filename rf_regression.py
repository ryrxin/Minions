from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ctf metrics calciation
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100        
    return mse, rmse, mae, mape

# 1. random forest training function
def rf_train_model(X_train, y_train):
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(X_train, y_train)
    return rf_model

# 2. random forest prediction function
def rf_next_day_pred(model, X_test):
    rf_y_pred = model.predict(X_test)
    return rf_y_pred

# 3. random forest execute function
def rf_predict_days_predict(df, tickers, num_lags, days_predict):
    master_df = (df.pct_change() + 1)[1:].dropna()
    predictions = {}
    metrics = {}

    for ticker in tickers:
        X = master_df[ticker].values
        y = master_df[ticker].values

        # lagged variables
        X_lagged = np.array([X[i - num_lags:i] for i in range(num_lags, len(X))])
        y_lagged = y[num_lags:]

        # Split data into training and testing (for simplicity, assume all data is used for training)
        X_train = X_lagged
        y_train = y_lagged

        # Train the model
        model = rf_train_model(X_train, y_train)

        # Initialize array to store predictions
        predictions[ticker] = []

        # Perform multi-step forecasting
        for i in range(days_predict):
            # Prepare input for prediction (use last num_lags values for prediction)
            X_pred = X[-num_lags:].reshape(1, -1)

            # Predict next day
            y_pred = model.predict(X_pred)
            predictions[ticker].append(y_pred[0])

            # Update X with predicted value for next iteration
            X = np.append(X, y_pred)

        # Calculate metrics for the entire forecast horizon
        y_true = master_df[ticker].values[-days_predict:]
        mse, rmse, mae, mape = calculate_metrics(y_true, predictions[ticker])
        metrics[ticker] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }

    return predictions, metrics