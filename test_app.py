"""
Automated unit & integration tests for Stock Price Predictor (Indian Stocks & INR).
"""

import sys
import unittest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

from data_fetcher import fetch_stock_data, _generate_synthetic_stock_data, POPULAR_STOCKS
from model import extract_features, train_and_forecast, get_model
from main import app


class TestStockPredictor(unittest.TestCase):

    def test_data_fetcher_popular_stocks(self):
        """Test popular stocks list is well-formed with Indian companies."""
        self.assertGreaterEqual(len(POPULAR_STOCKS), 5)
        self.assertEqual(POPULAR_STOCKS[0]["symbol"], "RELIANCE")

    def test_synthetic_data_generation(self):
        """Test synthetic data generator produces valid OHLCV dataframe in INR."""
        df, meta = _generate_synthetic_stock_data("RELIANCE", period="1y")
        self.assertFalse(df.empty)
        self.assertGreater(len(df), 50)
        self.assertIn("Close", df.columns)
        self.assertIn("Volume", df.columns)
        self.assertEqual(meta["symbol"], "RELIANCE")
        self.assertEqual(meta["currency"], "INR")
        self.assertEqual(meta["currency_symbol"], "₹")
        self.assertGreater(meta["current_price"], 0)

    def test_data_fetcher_integration(self):
        """Test fetch_stock_data returns non-empty dataset for Indian ticker."""
        df, meta = fetch_stock_data("RELIANCE", period="6mo")
        self.assertFalse(df.empty)
        self.assertGreater(len(df), 20)
        self.assertIn("Close", df.columns)
        self.assertIn("symbol", meta)
        self.assertEqual(meta["currency_symbol"], "₹")

    def test_feature_extraction(self):
        """Test technical features extraction."""
        df, _ = _generate_synthetic_stock_data("TCS", period="1y")
        clean_df, feature_cols = extract_features(df)
        self.assertFalse(clean_df.empty)
        self.assertIn("Lag_1", feature_cols)
        self.assertIn("SMA_20", feature_cols)
        self.assertIn("RSI_14", feature_cols)
        self.assertTrue(all(col in clean_df.columns for col in feature_cols))

    def test_ml_models_training_and_forecasting(self):
        """Test training and multi-step forecasting across different model types."""
        df, _ = _generate_synthetic_stock_data("INFY", period="1y")

        models = ["linear_regression", "ridge", "random_forest"]
        for model_type in models:
            res = train_and_forecast(df, model_type=model_type, forecast_days=5)
            self.assertEqual(res["model_type"], model_type)
            self.assertIn("metrics", res)
            self.assertIn("r2_score", res["metrics"])
            self.assertIn("rmse", res["metrics"])
            self.assertIn("mae", res["metrics"])
            self.assertEqual(len(res["forecast"]), 5)
            self.assertIn("summary", res)
            self.assertIn(res["summary"]["signal"], ["BULLISH", "BEARISH", "NEUTRAL"])

    def test_fastapi_endpoints(self):
        """Test FastAPI REST endpoints with Indian stocks."""
        client = TestClient(app)

        # Health endpoint
        res_health = client.get("/api/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json()["status"], "ok")

        # Popular stocks
        res_popular = client.get("/api/stocks/popular")
        self.assertEqual(res_popular.status_code, 200)
        self.assertIn("stocks", res_popular.json())

        # History endpoint
        res_hist = client.get("/api/stock/RELIANCE/history?period=6mo")
        self.assertEqual(res_hist.status_code, 200)
        hist_data = res_hist.json()
        self.assertIn("history", hist_data)
        self.assertGreater(hist_data["count"], 0)

        # Predict endpoint
        payload = {
            "ticker": "RELIANCE",
            "model_type": "linear_regression",
            "forecast_days": 7,
            "period": "1y",
            "force_refresh": False
        }
        res_pred = client.post("/api/stock/predict", json=payload)
        self.assertEqual(res_pred.status_code, 200)
        pred_json = res_pred.json()
        self.assertEqual(pred_json["status"], "success")
        self.assertEqual(len(pred_json["result"]["forecast"]), 7)


if __name__ == "__main__":
    unittest.main()
