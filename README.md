# Stock-Price-Predictor
Interactive ML-powered stock forecasting application using historical market data, technical indicators, and multiple regression models. Provides 1–30 day predictions, interactive charts, trading signals, model metrics, and feature importance analysis.
# 📈 Stock Price Predictor

A fast, interactive stock price forecasting web application built with **FastAPI**, **Scikit-Learn**, and a modern **Chart.js** dashboard.

---

## ✨ Features

- **Live Market Data**: Fetches historical Open, High, Low, Close, Volume data for any stock ticker (e.g. `AAPL`, `NVDA`, `TSLA`, `MSFT`, `GOOGL`, `AMZN`, `BTC-USD`).
- **Multiple Machine Learning Algorithms**:
  - **Linear Regression**: Fast, interpretable baseline with coefficients.
  - **Ridge Regression**: L2-regularized model for robust generalization.
  - **Random Forest Regressor**: Non-linear ensemble model capturing complex interactions.
  - **Gradient Boosting Regressor**: High-performance gradient boosted trees.
- **Rich Feature Engineering**:
  - Technical Moving Averages ($SMA_{7}, SMA_{20}, SMA_{50}$) & distance indicators
  - Momentum & Multi-day Returns ($1\text{d}, 3\text{d}, 5\text{d}$)
  - Price Lags ($t-1, t-2, t-3, t-5$)
  - Relative Strength Index ($RSI_{14}$)
  - 10-day Return Volatility
  - Volume Ratio
- **Multi-Step Future Forecasting**: Autoregressive day-by-day rollout into the future (1 to 30 days).
- **Interactive UI Dashboard**:
  - Dynamic interactive price chart with glowing trajectory lines.
  - Toggles for SMA 20, SMA 50, Backtest Fit, and Trading Volume.
  - KPI cards: Target Price, Expected % Change, Signal (Bullish/Bearish/Neutral), $R^2$ Accuracy, MAE, MAPE, Directional Hit Ratio.
  - Day-by-day forecast table with daily and cumulative returns.
  - Feature importance / driver weights visualization.
  - Quick-click chips for popular stocks.

---

## 🚀 Quick Start

### 1. Run the Application
Open your terminal in this directory:
```bash
python run.py
```
Or directly with Uvicorn:
```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Open the Dashboard
Navigate to:
```
http://127.0.0.1:8000
```
Interactive Swagger API documentation is available at:
```
http://127.0.0.1:8000/docs
```

---

## 🧪 Running Tests
Run the automated test suite with:
```bash
python -m unittest test_app.py
```

---

## 📁 Project Structure

```
stock-price-predictor/
├── main.py              # FastAPI server, endpoints, static routing
├── model.py             # Feature engineering, ML training, multi-step forecaster
├── data_fetcher.py      # Market data fetcher (yfinance, direct Yahoo API, caching, offline fallback)
├── run.py               # Launcher script with auto-browser opening
├── test_app.py          # Unit & integration tests
├── requirements.txt     # Dependencies
├── static/
│   ├── index.html       # Clean Tailwind CSS UI layout
│   ├── style.css        # Dashboard styling & glowing effects
│   └── app.js           # Chart.js rendering & API client logic
└── README.md            # Documentation
```
