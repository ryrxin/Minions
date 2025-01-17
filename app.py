from flask import Flask, request, render_template, send_file
import numpy as np
import io
import pandas as pd

from download_data import data_selection
from linear_regression import slr_cumulative_days_pctchange
from rf_regression import rf_predict_days_predict
from xgb_regression import xgb_predict_days_predict
from k_means_clustering import fetch_seasonal_data, compute_optimal_k, compute_markettrend_optimal_k, compute_seasonal_optimal_k, data_prep_returns, data_prep_seasonal, seasonal_trend_elbow, returns_volatility_elbow, plot_returns, plot_markettrend, plot_seasonaltrend
from association_pairing import analyze_pairs, analyze_and_plot_single_pair

app = Flask(__name__)

# index
@app.route('/')
def index():
    return render_template('index.html')

# predict
@app.route('/predict')
def predict():
    return render_template('predict.html')

# predict route
@app.route('/predict', methods=['POST'])
def prediction():
    index = request.form['index']
    start_date = request.form['start_date']
    num_lags = int(request.form['num_lags'])
    days_predict = int(request.form['days_predict'])

    tickers, df = data_selection(index, start_date)
    slr_predictions, slr_metrics = slr_cumulative_days_pctchange(df, tickers, num_lags, days_predict)
    rf_predictions, rf_metrics = rf_predict_days_predict(df, tickers, num_lags, days_predict)
    xgb_predictions, xgb_metrics = xgb_predict_days_predict(df, tickers, num_lags, days_predict)

       # Prepare the data for the CSV
    data_combined = []
    
    for stock in tickers:
        data_combined.append({
            'Stock': stock,
            'Model': 'MLR',
            'Prediction': slr_predictions.get(stock, [None])[0],
            'MSE': slr_metrics.get(stock, {}).get('MSE', None),
            'RMSE': slr_metrics.get(stock, {}).get('RMSE', None),
            'MAE': slr_metrics.get(stock, {}).get('MAE', None),
            'MAPE': slr_metrics.get(stock, {}).get('MAPE', None)
        })
        data_combined.append({
            'Stock': stock,
            'Model': 'RF',
            'Prediction': rf_predictions.get(stock, [None])[0],
            'MSE': rf_metrics.get(stock, {}).get('MSE', None),
            'RMSE': rf_metrics.get(stock, {}).get('RMSE', None),
            'MAE': rf_metrics.get(stock, {}).get('MAE', None),
            'MAPE': rf_metrics.get(stock, {}).get('MAPE', None)
        })
        data_combined.append({
            'Stock': stock,
            'Model': 'XGB',
            'Prediction': xgb_predictions.get(stock, [None])[0],
            'MSE': xgb_metrics.get(stock, {}).get('MSE', None),
            'RMSE': xgb_metrics.get(stock, {}).get('RMSE', None),
            'MAE': xgb_metrics.get(stock, {}).get('MAE', None),
            'MAPE': xgb_metrics.get(stock, {}).get('MAPE', None)
        })

    df_output = pd.DataFrame(data_combined)

    # Create a CSV file in memory
    output = io.StringIO()
    df_output.to_csv(output, index=False)
    output.seek(0)

    return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name='predictions.csv')

    # return render_template('predict.html', slr_result=slr_result, rf_result=rf_result, xgb_result=xgb_result)

# plot
@app.route('/clustering')
def clustering():
    return render_template('clustering.html')

# plot route
@app.route('/clustering', methods=['POST'])
def create_plot():
    plot_type = request.form['plot']
    index = request.form['index']
    start_date = request.form['start_date']

    tickers, df = data_selection(index, start_date)
    returns = data_prep_returns(tickers, df)
    optimal_k = returns_volatility_elbow(returns)

    match plot_type:
        case "Returns":
            cluster_labels = compute_optimal_k(returns, optimal_k)
            plot_div = plot_returns(cluster_labels, returns)
        
        case "Market":
            cluster_labels_markettrend = compute_markettrend_optimal_k(returns)
            plot_div = plot_markettrend(cluster_labels_markettrend, returns)

        case "Seasonal":
            seasonal_data = fetch_seasonal_data(tickers, df)
            X_seasonal = data_prep_seasonal(seasonal_data)
            optimal_k_seasonal = seasonal_trend_elbow(X_seasonal)
            cluster_labels_seasonal = compute_seasonal_optimal_k(X_seasonal, optimal_k_seasonal)
            plot_div = plot_seasonaltrend(X_seasonal, cluster_labels_seasonal, tickers)

    return render_template('clustering.html', plot_div=plot_div, plot_type=plot_type, index=index, start_date=start_date)

# association pairing
@app.route('/associationpairing')
def association_pairing():
    return render_template('associationpairing.html')

@app.route('/getstocks', methods=['POST'])
def get_stocks():
    index = request.form['index']
    start_date = request.form['start_date']

    tickers = data_selection(index, start_date)[0]

    return render_template('associationpairing.html', index=index, tickers=tickers, start_date=start_date)

@app.route('/associationpairing', methods=['POST'])
def analyse_pairs():
    index = request.form['index']
    start_date = request.form['start_date']
    stock1 = request.form['stock1']
    stock2 = request.form['stock2']

    tickers, df = data_selection(index, start_date)

    results, plot_priceseries, plot_spread, plot_z_score = analyze_and_plot_single_pair(df, stock1, stock2)

    return render_template('associationpairing.html', results=results, plot_priceseries=plot_priceseries, plot_spread=plot_spread, plot_z_score=plot_z_score)

# main
if __name__ == '__main__':
    app.run()