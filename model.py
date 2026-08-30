"""
Stock Price Prediction ML Engine.
Implements feature engineering, model training (Linear Regression, Ridge, Random Forest, Gradient Boosting),
train/test evaluation, and recursive multi-step future forecasting.
"""

import datetime
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def safe_val(val, default: float = 0.0) -> float:
    """Ensures value is a clean float without NaN or Inf."""
    if val is None or pd.isna(val) or np.isnan(val) or np.isinf(val):
        return default
    return float(val)


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates standard Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def extract_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extracts informative technical features from OHLCV data for regression models.
    """
    data = df.copy()
    data = data.sort_index()

    # Base price series
    close = data["Close"].ffill().bfill()
    data["Close"] = close

    # Moving Averages
    data["SMA_7"] = close.rolling(window=7, min_periods=1).mean()
    data["SMA_20"] = close.rolling(window=20, min_periods=1).mean()
    data["SMA_50"] = close.rolling(window=50, min_periods=1).mean()

    # Ratios relative to moving averages
    data["Dist_SMA_7"] = (close - data["SMA_7"]) / (data["SMA_7"] + 1e-9)
    data["Dist_SMA_20"] = (close - data["SMA_20"]) / (data["SMA_20"] + 1e-9)
    data["Dist_SMA_50"] = (close - data["SMA_50"]) / (data["SMA_50"] + 1e-9)

    # Momentum / Returns
    data["Return_1d"] = close.pct_change(1).fillna(0)
    data["Return_3d"] = close.pct_change(3).fillna(0)
    data["Return_5d"] = close.pct_change(5).fillna(0)

    # Lags (Close prices t-1, t-2, t-3, t-5)
    data["Lag_1"] = close.shift(1)
    data["Lag_2"] = close.shift(2)
    data["Lag_3"] = close.shift(3)
    data["Lag_5"] = close.shift(5)

    # Volatility
    data["Vol_10"] = data["Return_1d"].rolling(window=10, min_periods=1).std().fillna(0)

    # Technical indicator: RSI
    data["RSI_14"] = calculate_rsi(close, 14)

    # Volume ratio
    if "Volume" in data.columns:
        vol_clean = data["Volume"].fillna(1_000_000)
        vol_ma = vol_clean.rolling(window=20, min_periods=1).mean()
        data["Vol_Ratio"] = (vol_clean / (vol_ma + 1e-9)).fillna(1.0)
    else:
        data["Vol_Ratio"] = 1.0

    feature_cols = [
        "Lag_1", "Lag_2", "Lag_3", "Lag_5",
        "SMA_7", "SMA_20", "SMA_50",
        "Dist_SMA_7", "Dist_SMA_20",
        "Return_1d", "Return_3d", "Return_5d",
        "Vol_10", "RSI_14", "Vol_Ratio"
    ]

    # Drop rows with NaN from lags
    clean_data = data.dropna(subset=feature_cols).copy()
    return clean_data, feature_cols


def get_model(model_name: str):
    """Factory to instantiate the selected ML algorithm."""
    name = model_name.lower().strip()
    if name in ["ridge", "ridge_regression"]:
        return Ridge(alpha=1.0)
    elif name in ["random_forest", "rf", "randomforest"]:
        return RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    elif name in ["gradient_boosting", "gbm", "gradientboosting"]:
        return GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    else:  # Default Linear Regression
        return LinearRegression()


def train_and_forecast(
    df: pd.DataFrame,
    model_type: str = "linear_regression",
    forecast_days: int = 7,
    test_size_ratio: float = 0.2
) -> Dict[str, Any]:
    """
    Processes dataframe, trains model, computes metrics, and generates future forecast.
    """
    clean_df, feature_cols = extract_features(df)
    if len(clean_df) < 25:
        raise ValueError(f"Insufficient historical data points ({len(clean_df)}). Need at least 25 bars.")

    X = clean_df[feature_cols].values
    y = clean_df["Close"].values

    # Chronological Train-Test Split (no future leakage)
    split_idx = max(int(len(clean_df) * (1.0 - test_size_ratio)), len(clean_df) - 60)
    split_idx = max(split_idx, 15)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = get_model(model_type)
    model.fit(X_train, y_train)

    # Predictions on test set for evaluation
    y_test_pred = model.predict(X_test) if len(X_test) > 0 else np.array([])

    # Evaluation Metrics
    if len(y_test) > 0:
        r2 = max(-1.0, safe_val(r2_score(y_test, y_test_pred), 0.0))
        rmse = safe_val(np.sqrt(mean_squared_error(y_test, y_test_pred)), 0.0)
        mae = safe_val(mean_absolute_error(y_test, y_test_pred), 0.0)
        mape = safe_val(np.mean(np.abs((y_test - y_test_pred) / (y_test + 1e-9))) * 100.0, 0.0)

        # Directional Accuracy (predicting up vs down movement)
        if len(y_test) > 1:
            actual_dir = np.sign(np.diff(y_test))
            pred_dir = np.sign(np.diff(y_test_pred))
            dir_acc = safe_val(np.mean(actual_dir == pred_dir) * 100.0, 50.0)
        else:
            dir_acc = 50.0
    else:
        r2, rmse, mae, mape, dir_acc = 0.0, 0.0, 0.0, 0.0, 50.0

    # Feature Importance or Model Coefficients
    feature_importance = {}
    if hasattr(model, "coef_"):
        coefs = model.coef_
        feature_importance = {col: round(safe_val(coefs[i]), 4) for i, col in enumerate(feature_cols)}
    elif hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
        feature_importance = {col: round(safe_val(imps[i]), 4) for i, col in enumerate(feature_cols)}

    # Multi-Step Autoregressive Future Forecasting
    future_forecast = []
    future_dates = []
    
    last_date = clean_df.index[-1]
    running_df = clean_df.copy()

    for _ in range(forecast_days):
        # Next trading day (skip weekends)
        next_date = last_date + datetime.timedelta(days=1)
        while next_date.weekday() >= 5:  # Saturday/Sunday
            next_date += datetime.timedelta(days=1)
        last_date = next_date

        # Re-compute features for latest row
        latest_feats, _ = extract_features(running_df)
        x_next = latest_feats[feature_cols].iloc[-1:].values
        
        pred_close = safe_val(model.predict(x_next)[0])
        # Guard against runaway negative values or explosion
        current_last_close = safe_val(running_df["Close"].iloc[-1])
        # Clamp daily variation to max 12% to stay physically plausible
        pred_close = max(current_last_close * 0.88, min(current_last_close * 1.12, pred_close))
        pred_close = round(pred_close, 2)

        future_forecast.append(pred_close)
        future_dates.append(next_date.strftime("%Y-%m-%d"))

        # Append step to running dataframe to auto-regressively predict next step
        new_row = pd.DataFrame({
            "Open": [pred_close],
            "High": [pred_close * 1.005],
            "Low": [pred_close * 0.995],
            "Close": [pred_close],
            "Volume": [int(running_df["Volume"].iloc[-1])] if "Volume" in running_df else [1000000]
        }, index=[pd.to_datetime(next_date)])
        
        running_df = pd.concat([running_df, new_row])

    # Format Historical Series for Visualization
    history_records = []
    for idx, row in clean_df.iterrows():
        close_v = safe_val(row["Close"])
        history_records.append({
            "date": idx.strftime("%Y-%m-%d"),
            "open": round(safe_val(row.get("Open"), close_v), 2),
            "high": round(safe_val(row.get("High"), close_v), 2),
            "low": round(safe_val(row.get("Low"), close_v), 2),
            "close": round(close_v, 2),
            "volume": int(safe_val(row.get("Volume", 0))),
            "sma_20": round(safe_val(row.get("SMA_20")), 2) if "SMA_20" in row and not pd.isna(row["SMA_20"]) else None,
            "sma_50": round(safe_val(row.get("SMA_50")), 2) if "SMA_50" in row and not pd.isna(row["SMA_50"]) else None,
        })

    # In-sample fitted predictions aligned with test set dates
    test_dates = [d.strftime("%Y-%m-%d") for d in clean_df.index[split_idx:]]
    fitted_test = [
        {"date": test_dates[i], "actual": round(safe_val(y_test[i]), 2), "predicted": round(safe_val(y_test_pred[i]), 2)}
        for i in range(len(y_test))
    ]

    current_price = safe_val(clean_df["Close"].iloc[-1])
    target_price = safe_val(future_forecast[-1]) if future_forecast else current_price
    expected_change_pct = round(((target_price - current_price) / (current_price + 1e-9)) * 100.0, 2)
    
    if expected_change_pct > 1.5:
        signal = "BULLISH"
    elif expected_change_pct < -1.5:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    return {
        "model_type": model_type,
        "metrics": {
            "r2_score": round(r2, 4),
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "mape_pct": round(mape, 2),
            "directional_accuracy_pct": round(dir_acc, 1),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        },
        "feature_importance": feature_importance,
        "summary": {
            "current_price": round(current_price, 2),
            "forecast_price": round(target_price, 2),
            "forecast_change_pct": expected_change_pct,
            "forecast_days": forecast_days,
            "signal": signal,
        },
        "history": history_records,
        "fitted_test": fitted_test,
        "forecast": [
            {"date": future_dates[i], "predicted_close": future_forecast[i]}
            for i in range(len(future_forecast))
        ]
    }
