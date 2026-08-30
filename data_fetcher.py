"""
Data Fetcher module for Indian & Global stock price data.
Supports NSE/BSE symbols (e.g. RELIANCE, TCS, INFY, HDFCBANK) with automatic .NS/.BO resolution,
INR currency formatting, direct Yahoo Finance v8 API, and realistic offline fallback.
"""

import time
import datetime
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import requests

# Memory cache: key -> (timestamp, data_df, metadata)
_CACHE: Dict[str, Tuple[float, pd.DataFrame, Dict[str, Any]]] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes cache

POPULAR_STOCKS = [
    {"symbol": "RELIANCE", "name": "Reliance Industries Ltd.", "category": "Energy/Retail"},
    {"symbol": "TCS", "name": "Tata Consultancy Services", "category": "IT Services"},
    {"symbol": "HDFCBANK", "name": "HDFC Bank Ltd.", "category": "Banking"},
    {"symbol": "INFY", "name": "Infosys Ltd.", "category": "IT Services"},
    {"symbol": "ICICIBANK", "name": "ICICI Bank Ltd.", "category": "Banking"},
    {"symbol": "TATAMOTORS", "name": "Tata Motors Ltd.", "category": "Auto/EV"},
    {"symbol": "SBIN", "name": "State Bank of India", "category": "PSU Bank"},
    {"symbol": "ITC", "name": "ITC Limited", "category": "FMCG"},
    {"symbol": "BHARTIARTL", "name": "Bharti Airtel Ltd.", "category": "Telecom"},
    {"symbol": "LT", "name": "Larsen & Toubro", "category": "Infrastructure"},
    {"symbol": "ZOMATO", "name": "Zomato Ltd.", "category": "Tech"},
    {"symbol": "^NSEI", "name": "NIFTY 50 Index", "category": "NSE Index"},
]


def clean_val(val, default: float = 0.0) -> float:
    """Sanitizes floats to prevent JSON NaN serialization errors."""
    if val is None or pd.isna(val) or np.isnan(val) or np.isinf(val):
        return default
    return float(val)


def normalize_ticker_symbol(ticker: str) -> list:
    """
    Expands ticker search candidates.
    If ticker has no suffix, tries .NS (NSE) first, then .BO (BSE), then bare symbol.
    """
    raw = ticker.strip().upper()
    if raw.startswith("^"):
        return [raw]
    if raw == "NIFTY" or raw == "NIFTY50":
        return ["^NSEI", "NIFTYBEES.NS", raw]
    if raw == "BANKNIFTY":
        return ["^NSEBANK", "BANKBEES.NS", raw]
    if "." in raw:
        return [raw]
    return [f"{raw}.NS", f"{raw}.BO", raw]


def _generate_synthetic_stock_data(ticker: str, period: str = "1y") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generates realistic synthetic stock data in INR if live market API is unavailable."""
    days_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
    total_days = days_map.get(period.lower(), 365)

    seed = sum(ord(c) for c in ticker.upper())
    np.random.seed(seed)

    # Typical Indian stock price range ₹500 - ₹3500
    base_price = 500.0 + (seed % 2800)
    mu = 0.0005  # daily upward drift
    sigma = 0.016  # daily volatility

    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=int(total_days * 1.45))
    date_range = pd.bdate_range(start=start_date, end=end_date)
    if len(date_range) > total_days:
        date_range = date_range[-total_days:]

    n = len(date_range)
    daily_returns = np.random.normal(mu, sigma, n)
    trend = np.sin(np.linspace(0, 3 * np.pi, n)) * 0.04
    prices = base_price * np.exp(np.cumsum(daily_returns + trend / n))

    df = pd.DataFrame(index=date_range)
    df["Close"] = np.round(prices, 2)
    spread = np.random.uniform(0.005, 0.018, n) * prices
    df["Open"] = np.round(prices + np.random.normal(0, 0.004, n) * prices, 2)
    df["High"] = np.round(np.maximum(df["Open"], df["Close"]) + spread, 2)
    df["Low"] = np.round(np.minimum(df["Open"], df["Close"]) - spread, 2)
    df["Volume"] = np.random.randint(500_000, 15_000_000, n)

    display_symbol = ticker.upper()
    meta = {
        "symbol": display_symbol,
        "name": f"{display_symbol} (NSE India Simulator)",
        "currency": "INR",
        "currency_symbol": "₹",
        "current_price": float(df["Close"].iloc[-1]),
        "previous_close": float(df["Close"].iloc[-2]) if len(df) > 1 else float(df["Close"].iloc[-1]),
        "source": "Synthetic / Offline Engine (NSE)",
    }
    return df, meta


def _fetch_from_yfinance_candidate(ticker: str, period: str) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
    """Attempts to fetch a specific ticker candidate from yfinance."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval="1d", auto_adjust=True)
        if df is None or df.empty or len(df) < 15:
            return None

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.dropna(subset=["Close"])
        df = df.ffill().bfill()
        if df.empty or len(df) < 15:
            return None

        info = {}
        curr = "INR" if (".NS" in ticker or ".BO" in ticker or "^NSE" in ticker) else "USD"
        name = ticker
        try:
            info = t.fast_info
            curr = getattr(info, "currency", curr) or curr
            curr_price = clean_val(getattr(info, "last_price", None), float(df["Close"].iloc[-1]))
            prev_close = clean_val(getattr(info, "previous_close", None), float(df["Close"].iloc[-2]) if len(df) > 1 else curr_price)
        except Exception:
            curr_price = float(df["Close"].iloc[-1])
            prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else curr_price

        # Map currency symbol
        curr_sym = "₹" if (curr.upper() in ["INR", "₹"] or ".NS" in ticker or ".BO" in ticker) else "$"

        meta = {
            "symbol": ticker,
            "name": name,
            "currency": curr.upper() if curr else "INR",
            "currency_symbol": curr_sym,
            "current_price": round(curr_price, 2),
            "previous_close": round(prev_close, 2),
            "source": f"NSE/Yahoo Finance ({ticker})",
        }
        return df, meta
    except Exception:
        return None


def _fetch_from_yahoo_api_candidate(ticker: str, period: str) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
    """Direct HTTP request to Yahoo Finance Chart API endpoint for a candidate."""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"range": period, "interval": "1d", "indicators": "quote", "includeTimestamps": "true"}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, params=params, headers=headers, timeout=6)
        if resp.status_code != 200:
            return None

        data = resp.json()
        result = data.get("chart", {}).get("result")
        if not result or len(result) == 0:
            return None

        chart_data = result[0]
        timestamps = chart_data.get("timestamp", [])
        indicators = chart_data.get("indicators", {}).get("quote", [{}])[0]

        closes = indicators.get("close", [])
        opens = indicators.get("open", [])
        highs = indicators.get("high", [])
        lows = indicators.get("low", [])
        volumes = indicators.get("volume", [])

        if not timestamps or not closes or len(closes) < 15:
            return None

        valid_dates, valid_opens, valid_highs, valid_lows, valid_closes, valid_vols = [], [], [], [], [], []
        for i in range(len(timestamps)):
            c = closes[i] if i < len(closes) else None
            if c is None or np.isnan(c):
                continue
            o = opens[i] if i < len(opens) and opens[i] is not None and not np.isnan(opens[i]) else c
            h = highs[i] if i < len(highs) and highs[i] is not None and not np.isnan(highs[i]) else c
            l = lows[i] if i < len(lows) and lows[i] is not None and not np.isnan(lows[i]) else c
            v = volumes[i] if i < len(volumes) and volumes[i] is not None and not np.isnan(volumes[i]) else 0

            valid_dates.append(datetime.datetime.fromtimestamp(timestamps[i]).date())
            valid_opens.append(float(o))
            valid_highs.append(float(h))
            valid_lows.append(float(l))
            valid_closes.append(float(c))
            valid_vols.append(int(v))

        if len(valid_closes) < 15:
            return None

        df = pd.DataFrame(
            {
                "Open": valid_opens,
                "High": valid_highs,
                "Low": valid_lows,
                "Close": valid_closes,
                "Volume": valid_vols,
            },
            index=pd.to_datetime(valid_dates),
        )

        meta_info = chart_data.get("meta", {})
        curr = meta_info.get("currency", "INR" if (".NS" in ticker or ".BO" in ticker) else "USD")
        curr_sym = "₹" if (curr.upper() in ["INR", "₹"] or ".NS" in ticker or ".BO" in ticker) else "$"

        meta = {
            "symbol": ticker,
            "name": meta_info.get("shortName") or meta_info.get("symbol") or ticker,
            "currency": curr.upper() if curr else "INR",
            "currency_symbol": curr_sym,
            "current_price": round(float(df["Close"].iloc[-1]), 2),
            "previous_close": round(float(df["Close"].iloc[-2]) if len(df) > 1 else float(df["Close"].iloc[-1]), 2),
            "source": f"Yahoo Finance ({ticker})",
        }
        return df, meta
    except Exception:
        return None


def fetch_stock_data(ticker: str, period: str = "1y", force_refresh: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fetches stock data for a given ticker and period.
    Automatically resolves Indian tickers (.NS, .BO) and formats currency as ₹ INR.
    Uses caching and cascading fallbacks: yfinance -> Yahoo HTTP -> Synthetic Engine.
    """
    raw_ticker = ticker.strip().upper()
    candidates = normalize_ticker_symbol(raw_ticker)
    cache_key = f"{raw_ticker}_{period}"

    now = time.time()
    if not force_refresh and cache_key in _CACHE:
        ts, df, meta = _CACHE[cache_key]
        if now - ts < CACHE_TTL_SECONDS:
            return df.copy(), meta

    result = None

    # Try each candidate with yfinance
    for cand in candidates:
        result = _fetch_from_yfinance_candidate(cand, period)
        if result is not None:
            break

    # If yfinance failed, try Yahoo HTTP API
    if result is None:
        for cand in candidates:
            result = _fetch_from_yahoo_api_candidate(cand, period)
            if result is not None:
                break

    # Fallback to realistic synthetic data generator
    if result is None:
        result = _generate_synthetic_stock_data(raw_ticker, period)

    df, meta = result
    _CACHE[cache_key] = (now, df, meta)
    return df.copy(), meta
