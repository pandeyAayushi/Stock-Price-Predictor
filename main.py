"""
FastAPI application for Stock Price Predictor.
Provides REST APIs for stock data fetching, machine learning training, and forecast visualization.
"""

import os
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from data_fetcher import fetch_stock_data, POPULAR_STOCKS, clean_val
from model import train_and_forecast

app = FastAPI(
    title="Stock Price Predictor API",
    description="Interactive Stock Price Forecasting Engine using Machine Learning & FastAPI",
    version="1.0.0",
)

# CORS middleware for seamless local and remote access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static directory setup
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PredictRequest(BaseModel):
    ticker: str = Field(default="AAPL", description="Stock ticker symbol, e.g. AAPL, NVDA, MSFT")
    model_type: str = Field(default="linear_regression", description="linear_regression, ridge, random_forest, gradient_boosting")
    forecast_days: int = Field(default=7, ge=1, le=60, description="Number of trading days into the future to predict")
    period: str = Field(default="1y", description="Historical period: 6mo, 1y, 2y, 5y")
    force_refresh: bool = Field(default=False, description="Bypass cache and force re-fetch")


@app.get("/")
async def root():
    """Serves the main interactive dashboard interface."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Stock Price Predictor API is running. Access /static/index.html"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "stock-price-predictor"}


@app.get("/api/stocks/popular")
async def get_popular_stocks():
    """Returns curated list of popular stocks."""
    return {"stocks": POPULAR_STOCKS}


@app.get("/api/stock/{ticker}/history")
async def get_stock_history(
    ticker: str,
    period: str = Query(default="1y", description="Data range (6mo, 1y, 2y, 5y)")
):
    """Fetches raw historical stock prices and meta information."""
    try:
        df, meta = fetch_stock_data(ticker=ticker, period=period)
        history = []
        for idx, row in df.iterrows():
            close_val = clean_val(row.get("Close", 0.0))
            open_val = clean_val(row.get("Open"), close_val)
            high_val = clean_val(row.get("High"), close_val)
            low_val = clean_val(row.get("Low"), close_val)
            vol_val = int(clean_val(row.get("Volume", 0)))
            history.append({
                "date": idx.strftime("%Y-%m-%d"),
                "open": round(open_val, 2),
                "high": round(high_val, 2),
                "low": round(low_val, 2),
                "close": round(close_val, 2),
                "volume": vol_val,
            })
        return {
            "meta": meta,
            "count": len(history),
            "history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stock history: {str(e)}")


@app.post("/api/stock/predict")
async def predict_stock_price(req: PredictRequest):
    """
    Trains ML model on historical data and returns evaluation metrics + future forecast.
    """
    try:
        # Step 1: Fetch historical data
        df, meta = fetch_stock_data(
            ticker=req.ticker,
            period=req.period,
            force_refresh=req.force_refresh
        )

        if df.empty or len(df) < 25:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough historical data available for '{req.ticker}'. Found only {len(df)} records."
            )

        # Step 2: Train ML model and generate multi-step prediction
        forecast_result = train_and_forecast(
            df=df,
            model_type=req.model_type,
            forecast_days=req.forecast_days
        )

        return {
            "status": "success",
            "meta": meta,
            "result": forecast_result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import socket
    import uvicorn

    def get_free_port():
        for p in (8000, 8080, 8050, 5000, 5050, 8001):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", p))
                    return p
                except OSError:
                    continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    port = get_free_port()
    print(f"\n🚀 Server starting on http://127.0.0.1:{port}\n")
    uvicorn.run(app, host="127.0.0.1", port=port)

