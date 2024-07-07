
# Minions Portfolio Builder
The minions portfolio builder initiates 3 different features for stock analyses and portfolio building. The model allows for selection of stocks from 4 big index funds: 'SPY', 'NDX', 'DAX', and 'FTSE' using the Yahoo Finance API.

This attempts to build an AutoML tool for track 7 of the Tik Tok Tech Jam 2024: Unleashing Potential in Machine Learning Infrastructure.

## Table of Contents:
1. Setup & Configuration
2. Clustering Feature
3. Association Paring Feature
4. Stock Price Prediction Feature

## 1. Setup & Configuration

1. **Install Git**
- Download and install the latest version of Git at https://git-scm.com/

2. **Clone the repository**
- In your terminal, run ```git clone https://github.com/ryrxin/Minions```

3. **Download the required packages**
- Change your directory to the folder (e.g. ```cd Minions```)
- In your terminal, run ```pip install -r requirements.txt```

4. **Run the program**
- In your terminal, run ```python app.py```


## 2. Clustering Feature 
**Data Preparation:**
Calculate returns and volatility for stocks in percentage change.
Prepares seasonal data for seasonal clustering.

**K-Means Clustering:**
Determine the optimal number of clusters using the elbow plot method.
Compute cluster labels for different analyses.

**Visualization:**
Generate interactive 3 plots to visualize the clusters:
- Clusters for returns and volatility
- Clusters for market trends (Growth, Safe, Bounce Back stocks
- Clusters for Seasonality (Winter, Spring, Summer, Fall)

## 3. Assocation Pairing Feature
**Data Preparation:** After pairs are selected, probability of cointegration, spread and other statistical measures are calculated.

**Visualization:** Generate 3 interactive plots to visualize price series of the pair, spread of the pair, and z-score for arbitrage.

## 4. Stock Price Prediction Feature
**Data Preparation:** To ensure predictions are calculated on the same scale, price percentage change is the primary measure.

**Usage:** 3 time series regression models will be used to work on a common prediction task. Measure of accuracy for each model is returned to compare robustness and reliability of each model. Select T-th day prediction and number of lags to tune model. Make your analysis based on MSE/RMSE/MAE/MAPE values from each model.

**Linear Regressor:** The model utilizes rolling-window splitting and serves as a baseline model.

**Random Forest Regressor:** This model utilizes a simple train-test split and initiaties a 100 tree estimation.

**XGBoost Regressor:** This model also utilizes a simple train-test split.
