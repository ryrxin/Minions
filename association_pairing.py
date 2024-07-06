import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from itertools import combinations 
import matplotlib.pyplot as plt
from math import ceil
import plotly.express as px

def calculate_spread(df, stock1, stock2):
    return df[stock1] - df[stock2]

def analyze_pairs(df, stock1, stock2):
    results = []

    # Drop rows with any NaN values
    df = df[[stock1, stock2]].dropna()

    # Calculate spread for the current pair
    spread = calculate_spread(df, stock1, stock2)
    spread_mean = spread.mean()
    spread_std = spread.std()
    zscore = (spread - spread_mean) / spread_std

    # Perform cointegration test and calculate correlation
    score, p_value, _ = coint(df[stock1], df[stock2])
    correlation = df.corr().loc[stock1, stock2]

    # Plotting price series
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[stock1], label=stock1)
    plt.plot(df.index, df[stock2], label=stock2)
    plt.title('Price Series')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    price_series_plot = plt.gcf()

    # Plotting spread
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, spread, label='Spread')
    plt.axhline(y=spread_mean, color='r', linestyle='--', label='Spread Mean')
    plt.title('Spread')
    plt.xlabel('Date')
    plt.ylabel('Spread Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    spread_plot = plt.gcf()

    # Plotting Z-score
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, zscore, label='Z-score')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.axhline(y=1.0, color='b', linestyle='--')
    plt.axhline(y=-1.0, color='b', linestyle='--')
    plt.title('Z-score')
    plt.xlabel('Date')
    plt.ylabel('Z-score Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    zscore_plot = plt.gcf()

    # Append results for the current pair to the results list
    results.append({
        'Stock 1': stock1,
        'Stock 2': stock2,
        'Correlation': correlation,
        'Cointegration p-value': p_value,
        'Spread Mean': spread_mean,
        'Spread Std': spread_std,
        'Latest Z-Score': zscore.iloc[-1] if len(zscore) > 0 else np.nan,
        'Price Series Plot': price_series_plot,
        'Spread Plot': spread_plot,
        'Z-score Plot': zscore_plot
    })

    results_df = pd.DataFrame(results)
    results_html = results_df.to_html(index=False)

    return results_html, results_df, spread

# Function to analyze and plot a single pair using Plotly Express
def analyze_and_plot_single_pair(df, stock1, stock2):
    results = []

    # Calculate spread
    spread = calculate_spread(df, stock1, stock2)
    spread_mean = spread.mean()
    spread_std = spread.std()
    zscore = (spread - spread_mean) / spread_std

    # Perform cointegration test
    score, p_value, _ = coint(df[stock1], df[stock2])
    correlation = df.corr().loc[stock1, stock2]

    # Append results
    results.append({
        'Stock 1': stock1,
        'Stock 2': stock2,
        'Correlation': correlation,
        'Cointegration p-value': p_value,
        'Spread Mean': spread_mean,
        'Spread Std': spread_std,
        'Latest Z-Score': zscore.iloc[-1]
    })

    # Plot price series using Plotly Express
    fig_price = px.line(df, x=df.index, y=[stock1, stock2], title=f'Price Series: {stock1} and {stock2}')
    fig_price.update_layout(xaxis_title='Date', yaxis_title='Price')
    plot_priceseries = fig_price.to_html(full_html=False)

    # Plot spread using Plotly Express
    fig_spread = px.line(df, x=df.index, y=spread, title=f'Spread between {stock1} and {stock2}')
    fig_spread.update_layout(xaxis_title='Date', yaxis_title='Spread')
    plot_spread = fig_spread.to_html(full_html=False)

    # Plot z-score using Plotly Express
    fig_zscore = px.line(df, x=df.index, y=zscore, title=f'Z-score of {stock1} and {stock2}')
    fig_zscore.add_hline(y=zscore.mean(), line_dash="dot", annotation_text="Mean", annotation_position="bottom right")
    fig_zscore.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Z-score 1.0", annotation_position="bottom right")
    fig_zscore.add_hline(y=-1.0, line_dash="dash", line_color="green", annotation_text="Z-score -1.0", annotation_position="bottom right")
    fig_zscore.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="Z-score 2.0", annotation_position="bottom right")
    fig_zscore.add_hline(y=-2.0, line_dash="dash", line_color="green", annotation_text="Z-score -2.0", annotation_position="bottom right")
    fig_zscore.update_layout(xaxis_title='Date', yaxis_title='Z-score')
    plot_z_score = fig_zscore.to_html(full_html=False)

    return results, plot_priceseries, plot_spread, plot_z_score