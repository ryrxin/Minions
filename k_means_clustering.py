import pandas as pd
import yfinance as yf
import numpy as np
import pandas_datareader as dr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from kneed import KneeLocator
from scipy.cluster.vq import kmeans, vq
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

def fetch_seasonal_data(tickers, df):
  df2 = df.copy()
  df2.reset_index(inplace=True)
  seasonal_data = pd.DataFrame()

  for ticker in tickers:
     # Extract relevant data from the main df
      ticker_data = df2[['Date', ticker]].rename(columns={ticker: 'Adj Close'})

      ticker_data['Month'] = ticker_data['Date'].dt.month
      winter_months = [12, 1, 2]
      spring_months = [3, 4, 5]
      summer_months = [6, 7, 8]
      fall_months = [9, 10, 11]

      ticker_data.loc[ticker_data['Month'].isin(winter_months), 'Season'] = 'Winter'
      ticker_data.loc[ticker_data['Month'].isin(spring_months), 'Season'] = 'Spring'
      ticker_data.loc[ticker_data['Month'].isin(summer_months), 'Season'] = 'Summer'
      ticker_data.loc[ticker_data['Month'].isin(fall_months), 'Season'] = 'Fall'

      # Calculate percentage change
      ticker_data['Pct_Change'] = ticker_data['Adj Close'].pct_change()

      # Aggregate mean percentage change by season
      seasonal_agg = ticker_data.groupby('Season')['Pct_Change'].mean().reset_index()
      seasonal_agg['Ticker'] = ticker

      # Concatenate seasonal data for all tickers
      seasonal_data = pd.concat([seasonal_data, seasonal_agg], ignore_index=True)

  return seasonal_data


def data_prep_returns(tickers, df):
  prices_list = []
  for ticker in tickers:
    try:
      prices = dr.DataReader(df)
      prices = pd.DataFrame(prices)
      prices.columns = [ticker]
      prices_list.append(prices)
    except:
      pass
  # creating empty dataframe for returns and volatility
  returns = pd.DataFrame()

  # calculate returns and volatility
  returns["Returns"] = df.pct_change().mean() * 252
  returns['Volatility'] = df.pct_change().std() * 252                             # might need to change this so that user can set their preferred type e.g daily/annual
  returns.dropna(inplace=True) # drops NaN

  # prepare data for k-means clustering
  print('this is returns: \n', returns)
  data = np.asarray([np.asarray(returns['Returns']),np.asarray(returns['Volatility'])]).T
  X = data
  return X

def data_prep_seasonal(seasonal_data):
  # pivot seasonal data to have seasons as columns
  seasonal_data_pivot = seasonal_data.pivot(index='Ticker', columns='Season', values='Pct_Change').reset_index()
  seasonal_data_pivot = seasonal_data_pivot.fillna(0)
  X_seasonal = seasonal_data_pivot[['Winter', 'Spring', 'Summer', 'Fall']].values
  return X_seasonal

def returns_volatility_elbow(X):
  distorsions= []
  max_clusters = min(20, len(X))

  # calculate distortions for different numbers of clusters
  for k in range(1, max_clusters):
    k_means = KMeans(n_clusters = k)
    k_means.fit(X)
    distorsions.append(k_means.inertia_)
  print(distorsions)

  kn = KneeLocator(range(1, max_clusters), distorsions, curve='convex', direction='decreasing')
  optimal_k = kn.knee
  return optimal_k

def seasonal_trend_elbow(X_seasonal):
  distorsions_seasonal = []
  max_clusters_seasonal = min(20, len(X_seasonal))
  # calculate distortions for different numbers of clusters
  for k in range(1, max_clusters_seasonal):
    k_means_seasonal = KMeans(n_clusters=k)
    k_means_seasonal.fit(X_seasonal)
    distorsions_seasonal.append(k_means_seasonal.inertia_)
  print(distorsions_seasonal)

  kn_seasonal = KneeLocator(range(1, max_clusters_seasonal), distorsions_seasonal, curve='convex', direction='decreasing')
  optimal_k_seasonal = kn_seasonal.knee
  print(f"The optimal number of clusters for seasonal trends is: {optimal_k_seasonal}")
  return optimal_k_seasonal

def compute_optimal_k(X, optimal_k):
  centroids,_ = kmeans(X, optimal_k)
  cluster_labels,_ = vq(X,centroids)
  return cluster_labels

def compute_seasonal_optimal_k(X_seasonal, optimal_k_seasonal):
  centroids_seasonal,_ = kmeans(X_seasonal, optimal_k_seasonal)
  cluster_labels_seasonal,_ = vq(X_seasonal, centroids_seasonal)
  return cluster_labels_seasonal

def compute_markettrend_optimal_k(X):
  centroids_markettrend,_ = kmeans(X, 3)
  cluster_labels_markettrend,_ = vq(X,centroids_markettrend)
  return cluster_labels_markettrend

def plot_returns(cluster_labels, returns):
  returns = pd.DataFrame(returns, columns=['Returns', 'Volatility'])
  # DataFrame with cluster details
  cluster_details = [(name, cluster) for name, cluster in zip(returns.index, cluster_labels)]
  cluster_details_df = pd.DataFrame(cluster_details, columns=['Ticker', 'Cluster'])

  # combining returns data with cluster labels
  clustered_returns = returns.reset_index()
  clustered_returns['Cluster'] = cluster_details_df['Cluster']

  # plotting the clusters
  clustered_returns.columns = ['Ticker', 'Returns', 'Volatility', 'Cluster']
  fig = px.scatter(clustered_returns, x="Returns", y="Volatility", color="Cluster", hover_data=["Ticker"])
  fig.update_layout(coloraxis_showscale=False)

  # output to html
  plot_div = fig.to_html(full_html=False)
  return plot_div

def plot_markettrend(cluster_labels_markettrend, returns):
  returns = pd.DataFrame(returns, columns=['Returns', 'Volatility'])
  # DataFrame with market trend cluster details
  cluster_details_markettrend = [(name, cluster) for name, cluster in zip(returns.index, cluster_labels_markettrend)]
  cluster_details_markettrend_df = pd.DataFrame(cluster_details_markettrend, columns=['Ticker', 'Cluster'])

  # combining market trend data with cluster labels
  clustered_markettrend = returns.reset_index()
  clustered_markettrend = pd.concat([clustered_markettrend, cluster_details_markettrend_df[['Ticker']]], axis=1)
  clustered_markettrend['Cluster'] = cluster_details_markettrend_df['Cluster']
  # market trend mapping
  trend_mapping = {
    0: 'Growth Stocks',
    1: 'Safe Stocks',
    2: 'Bounce Back Stocks'
  }
  clustered_markettrend['Market Trend'] = clustered_markettrend['Cluster'].map(trend_mapping)
  fig = px.scatter(clustered_markettrend, x="Returns", y="Volatility", color="Market Trend", hover_data=["Ticker"])
  fig.update_layout(coloraxis_showscale=False)

  plot_div = fig.to_html(full_html=False)
  return plot_div

def plot_seasonaltrend(X_seasonal, cluster_labels_seasonal, tickers):
    X_seasonal = pd.DataFrame(X_seasonal, columns=['Winter', 'Spring', 'Summer', 'Fall'])
    # Ensure 'Ticker' column is included
    X_seasonal['Ticker'] = tickers
    # DataFrame with seasonal trend cluster labels
    clustered_seasonal = X_seasonal.copy()
    clustered_seasonal['Seasonal_Cluster'] = cluster_labels_seasonal
    # seasonal trend mapping
    seasonal_trend_mapping = {
        0: 'Winter Stocks',
        1: 'Spring Stocks',
        2: 'Summer Stocks',
        3: 'Fall Stocks'
    }
    clustered_seasonal['Seasonal Trend'] = clustered_seasonal['Seasonal_Cluster'].map(seasonal_trend_mapping)
    fig_seasonal = px.scatter_matrix(clustered_seasonal,
                                     dimensions=['Winter', 'Spring', 'Summer', 'Fall'],
                                     color='Seasonal Trend',
                                     hover_data=['Ticker'],
                                     title="Seasonal Performance Mapping")
    fig_seasonal.update_layout(coloraxis_showscale=False)
    plot_div = fig_seasonal.to_html(full_html=False)
    return plot_div