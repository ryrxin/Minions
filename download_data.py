import yfinance as yf
import pandas as pd
from statsmodels.tsa.stattools import coint
from itertools import combinations
from math import sqrt, ceil
import matplotlib.pyplot as plt

def data_selection(index, start_date):
    match index:
        case "SPY":
            return fetch_spy_data(start_date)

        case "NDX":
            return fetch_ndx_data(start_date)

        case "DAX":
            return fetch_dax_data(start_date)

        case "FTSE":
            return fetch_ftse_data(start_date)

        case _ :
            return "Error"
        
# ===== SPY =====-
def fetch_spy_data(start_date):
    raw = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    tickers = raw['Symbol'].tolist()
    df = yf.download(tickers, start = start_date)['Adj Close']
    df = df.dropna(axis=1, how='all')
    tickers = df.columns.tolist()
    
    return tickers, df

# ===== NDX =====-
def fetch_ndx_data(start_date):
    raw = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")[4]
    tickers = raw['Ticker'].tolist()
    df = yf.download(tickers, start = start_date)['Adj Close']
    df = df.dropna(axis=1, how='all')
    tickers = df.columns.tolist()
    
    return tickers, df

# ===== DAX =====-
def fetch_dax_data(start_date):
    raw = pd.read_html("https://en.wikipedia.org/wiki/DAX")[4]
    tickers = raw['Ticker'].tolist()
    df = yf.download(tickers, start = start_date)['Adj Close']
    df = df.dropna(axis=1, how='all')
    tickers = df.columns.tolist()
    
    return tickers, df

# ===== FTSE =====-
def fetch_ftse_data(start_date):
    raw = pd.read_html("https://en.wikipedia.org/wiki/FTSE_100_Index")[4]
    tickers = raw['Ticker'].tolist()
    df = yf.download(tickers, start = start_date)['Adj Close']
    df = df.dropna(axis=1, how='all')
    tickers = df.columns.tolist()
    
    return tickers, df
