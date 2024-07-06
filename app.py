from flask import Flask, request, jsonify, render_template
import numpy as np

from download_data import data_selection
from linear_regression import slr_cumulative_days_pctchange
from k_means_clustering import fetch_seasonal_data, compute_optimal_k, compute_markettrend_optimal_k, compute_seasonal_optimal_k, data_prep_returns, data_prep_seasonal, seasonal_trend_elbow, returns_volatility_elbow, plot_returns, plot_markettrend, plot_seasonaltrend
from association_pairing import calculate_spread, analyze_pairs, analyze_and_plot_pairs

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
    ticker_select = request.form['tickers']
    start_date = request.form['start_date']
    num_lags = int(request.form['num_lags'])
    days_predict = int(request.form['days_predict'])

    tickers, df = data_selection(ticker_select, start_date)
    predictions, metrics = slr_cumulative_days_pctchange(df, tickers, num_lags, days_predict)

    result = {
        'predictions': predictions,
        'metrics': metrics
    }

    chart_div = result

    return render_template('index.html', chart_div=chart_div)

# plot
@app.route('/plot')
def plot():
    return render_template('plot.html')

# plot route
@app.route('/plot', methods=['POST'])
def create_plot():
    plot_type = request.form['plot']
    ticker_select = request.form['tickers']
    start_date = request.form['start_date']

    tickers, df = data_selection(ticker_select, start_date)
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

    return render_template('plot.html', plot_div=plot_div, plot_type=plot_type, ticker_select=ticker_select, start_date=start_date)

# association pairing
@app.route('/associationpairing')
def association_pairing():
    return render_template('associationpairing.html')

@app.route('/associationpairing', methods=['POST'])
def analyse_pairs():
    ticker_select = request.form['tickers']
    start_date = request.form['start_date']
    stock1 = request.form['stock1']
    stock2 = request.form['stock2']
    plotpairs = 'plotpairs' in request.form
    tickers, df = data_selection(ticker_select, start_date)

    calculate_spread(df, stock1, stock2)
    results = analyze_pairs(df, tickers)

    if plotpairs:
        chart_div = analyze_and_plot_pairs(df, tickers)
    else:
        chart_div = "<div>Main analysis done, no plot generated.</div>"

    return render_template('associationpairing.html', chart_div=chart_div)

# main
if __name__ == '__main__':
    app.run(debug=True)