import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from itertools import combinations 
import matplotlib.pyplot as plt
from math import ceil

def calculate_spread(df, stock1, stock2):
    spread = df[stock1] - df[stock2]
    return spread

def analyze_pairs(df, tickers):
    results = []

    # Drop rows with any NaN values
    df = df.dropna(axis=1)
    tickers = df.columns.tolist()

    # Iterate through all possible combinations of stock pairs
    for stock1, stock2 in combinations(tickers, 2):

        # Calculate spread for the current pair
        spread = calculate_spread(df, stock1, stock2)
        spread_mean = spread.mean()
        spread_std = spread.std()
        zscore = (spread - spread_mean) / spread_std

        # Perform cointegration test and calculate correlation
        score, p_value, _ = coint(df[stock1], df[stock2])
        correlation = df.corr().loc[stock1, stock2]

        # Append results for the current pair to the results list
        results.append({
            'Stock 1': stock1,
            'Stock 2': stock2,
            'Correlation': correlation,
            'Cointegration p-value': p_value,
            'Spread Mean': spread_mean,
            'Spread Std': spread_std,
            'Latest Z-Score': zscore.iloc[-1] if len(zscore) > 0 else np.nan
        })

    # Display results in a table
    results_df = pd.DataFrame(results)
    print(results_df)

# Function to calculate spread
def calculate_spread(df, stock1, stock2):
    return df[stock1] - df[stock2]

# Function to analyze pairs and plot graphs
def analyze_and_plot_pairs(df, tickers):
    results = []
    df = df.dropna(axis=1)
    tickers = df.columns.tolist()

    # Calculate the number of subplots needed
    num_pairs = len(list(combinations(tickers, 2)))
    num_cols = 3  # Adjust the number of columns to accommodate more subplots
    num_rows = ceil(num_pairs / num_cols)

    fig_price, axes_price = plt.subplots(num_rows, num_cols, figsize=(20, 5*num_rows), sharex=True, sharey=True)
    fig_price.suptitle('Price Series for Stock Pairs', fontsize=16)
    axes_price = axes_price.flatten()

    fig_spread, axes_spread = plt.subplots(num_rows, num_cols, figsize=(20, 5*num_rows), sharex=True, sharey=True)
    fig_spread.suptitle('Spread for Stock Pairs', fontsize=16)
    axes_spread = axes_spread.flatten()

    fig_zscore, axes_zscore = plt.subplots(num_rows, num_cols, figsize=(20, 5*num_rows), sharex=True, sharey=True)
    fig_zscore.suptitle('Z-score for Stock Pairs', fontsize=16)
    axes_zscore = axes_zscore.flatten()

    for i, (stock1, stock2) in enumerate(combinations(tickers, 2)):
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

        # Plot price series
        axes_price[i].plot(df.index, df[stock1], label=f'{stock1} Price')
        axes_price[i].plot(df.index, df[stock2], label=f'{stock2} Price')
        axes_price[i].set_title(f'{stock1} and {stock2}')
        axes_price[i].legend()

        # Plot spread
        spread.plot(ax=axes_spread[i], title=f'Spread between {stock1} and {stock2}')

        # Plot z-score
        zscore.plot(ax=axes_zscore[i], title=f'Z-score of {stock1} and {stock2}')
        axes_zscore[i].axhline(zscore.mean(), color='black')
        axes_zscore[i].axhline(1.0, color='red', linestyle='--')
        axes_zscore[i].axhline(-1.0, color='green', linestyle='--')
        axes_zscore[i].axhline(2.0, color='red', linestyle='--', linewidth=1)
        axes_zscore[i].axhline(-2.0, color='green', linestyle='--', linewidth=1)

    # Remove unused subplots
    for j in range(i + 1, len(axes_price)):
        fig_price.delaxes(axes_price[j])
        fig_spread.delaxes(axes_spread[j])
        fig_zscore.delaxes(axes_zscore[j])

    # Adjust layout
    fig_price.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_spread.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_zscore.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()

    return pd.DataFrame(results)

# Analyze pairs and plot graphs