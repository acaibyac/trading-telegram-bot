import pandas as pd, numpy as np

# ---- Setări implicite ----
DEFAULT_PERIOD   = "1y"   # 6mo, 1y, 2y, 5y, max
DEFAULT_INTERVAL = "1d"   # 1d, 1h, 15m (Stooq = doar daily)
SMA_FAST = 20
SMA_SLOW = 50
FEE_BPS  = 5              # 0.05%/tranzacție
# --------------------------

def guess_ticker(text: str, default="AAPL") -> str:
    """Folosește exact simbolul trimis de utilizator; altfel implicit."""
    if not text:
        return default
    return text.strip().upper()

def load_prices_safe(ticker: str, period: str = DEFAULT_PERIOD, interval: str = DEFAULT_INTERVAL):
    """
    Încearcă Yahoo (yfinance). Dacă nu merge sau e gol, revine la Stooq (daily).
    Returnează (DataFrame cu coloana 'close', 'Yahoo'|'Stooq').
    """
    # 1) Yahoo (yfinance)
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if not df.empty:
            df = df.rename(columns={"Close": "close"})[["close"]].dropna()
            return df, "Yahoo"
        raise RuntimeError("yfinance empty")
    except Exception:
        pass

    # 2) Stooq fallback (daily)
    try:
        import pandas_datareader.data as web
        df = web.DataReader(ticker, "stooq").sort_index()
        df = df.rename(columns={"Close": "close"})[["close"]].dropna()
        return df, "Stooq"
    except Exception as e:
        raise RuntimeError(f"Nu pot încărca date pentru {ticker}: {e}")

def sma_crossover(df: pd.DataFrame, fast: int = SMA_FAST, slow: int = SMA_SLOW) -> pd.DataFrame:
    """Generează semnale long-only: 1 când SMA(fast) > SMA(slow), altfel 0."""
    out = df.copy()
    out["sma_fast"] = out["close"].rolling(fast).mean()
    out["sma_slow"] = out["close"].rolling(slow).mean()
    out["signal"]   = (out["sma_fast"] > out["sma_slow"]).astype(int)
    return out.dropna()

def backtest_long_only(df_sig: pd.DataFrame, fee_bps: float = FEE_BPS, ann_factor: int = 252):
    """
    Execută la bara următoare: pos(t) = signal(t-1).
    Returnează dict cu metrici: CAGR, MaxDD, Sharpe, Trades.
    """
    bt = df_sig.copy()
    bt["pos"] = bt["signal"].shift(1).fillna(0)
    bt["ret"] = bt["close"].pct_change().fillna(0.0)

    trades = (bt["pos"].diff().abs().fillna(0) > 0).astype(int)
    bt["strategy_ret"] = bt["pos"] * bt["ret"] - trades * (fee_bps / 10000.0)

    curve = (1.0 + bt["strategy_ret"]).cumprod()

    if len(bt) <= 1:
        return {"CAGR": 0.0, "MaxDD": 0.0, "Sharpe": 0.0, "Trades": int(trades.sum())}

    cagr = curve.iloc[-1] ** (ann_factor / len(bt)) - 1.0
    dd = (curve / curve.cummax() - 1.0).min()
    mu, sigma = bt["strategy_ret"].mean(), bt["strategy_ret"].std() + 1e-12
    sharpe = (mu / sigma) * np.sqrt(ann_factor)

    return {
        "CAGR": float(cagr),
        "MaxDD": float(dd),
        "Sharpe": float(sharpe),
        "Trades": int(trades.sum()),
    }

def format_metrics(ticker: str, source: str, m: dict,
                   period: str = DEFAULT_PERIOD, interval: str = DEFAULT_INTERVAL) -> str:
    """Formatează rezultatul pentru mesajul de Telegram."""
    return (
        f"Ticker: {ticker}\n"
        f"Perioadă: {period} @ {interval}{' (dacă Stooq: daily)' if source=='Stooq' else ''}\n"
        f"Trades: {m['Trades']}\n"
        f"CAGR: {m['CAGR']*100:.2f}%\n"
        f"Max Drawdown: {m['MaxDD']*100:.2f}%\n"
        f"Sharpe: {m['Sharpe']:.2f}\n"
        f"Sursa date: {source}"
    )