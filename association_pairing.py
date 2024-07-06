import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from itertools import combinations 
import matplotlib.pyplot as plt
from math import ceil

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

def analyze_and_plot_pairs(df, stock1, stock2):
    results = []

    # Calculate spread for the specified pair
    spread = calculate_spread(df, stock1, stock2)
    spread_mean = spread.mean()
    spread_std = spread.std()
    zscore = (spread - spread_mean) / spread_std

    # Perform cointegration test and calculate correlation
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
        'Latest Z-Score': zscore.iloc[-1] if len(zscore) > 0 else np.nan
    })

    # Plotting price series
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[stock1], label=f'{stock1} Price')
    plt.plot(df.index, df[stock2], label=f'{stock2} Price')
    plt.title(f'Price Series for {stock1} and {stock2}')
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
    plt.title(f'Spread between {stock1} and {stock2}')
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
    plt.title(f'Z-score of {stock1} and {stock2}')
    plt.xlabel('Date')
    plt.ylabel('Z-score Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    zscore_plot = plt.gcf()

    # Display plots
    plt.show()

    return pd.DataFrame(results), price_series_plot, spread_plot, zscore_plot