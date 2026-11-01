"""
AGENTS WAR - Alpha Arena (Meraz 7.0) Trading Agent
Team: AlphaForge
AI-assisted: Yes (Google Antigravity / Claude Opus 4.6 Thinking)

Architecture: Volatility-Weighted Buy-and-Hold with Adaptive Rebalancing
  - Core idea: stay invested in all stocks, weighted inversely by volatility
  - Rebalance only every ~20 trading days (reduces transaction costs dramatically)
  - Gentle trend tilt: slightly overweight stocks above their long-term average
  - Crash guard: reduce exposure when recent market volatility spikes

Why this works:
  - All 12 stocks have positive long-term drift (avg ~25% annualized)
  - Transaction costs (0.1% per trade) destroy active strategies
  - Diversification across 12 low-correlated stocks (mean rho ~ 0.32) is strong
  - Simple strategies generalize; complex ones overfit
"""
import numpy as np
import pandas as pd

TEAM_NAME = "Harshad's employee"
TRAIN_WINDOW = (252, 2771)   # days 252-2771 (10 years)
MODEL = {}


# ─────────────────────────── TRAIN ───────────────────────────

def train(data):
    """Called once with TRAIN_WINDOW data.
    Learn volatility characteristics and optimal rebalance parameters."""
    close = data["close"]
    daily_ret = close.pct_change().dropna()
    stocks = list(close.columns)
    n = len(stocks)

    # 1. Learn per-stock annualized volatilities
    stock_vol = (daily_ret.std() * np.sqrt(252)).to_dict()
    MODEL["stock_vol"] = stock_vol

    # 2. Learn market volatility distribution for regime detection
    mkt_ret = daily_ret.mean(axis=1)
    roll_vol = mkt_ret.rolling(21).std() * np.sqrt(252)
    MODEL["vol_median"] = float(roll_vol.median())
    MODEL["vol_high"] = float(roll_vol.quantile(0.85))
    MODEL["vol_crisis"] = float(roll_vol.quantile(0.95))

    # 3. Test rebalance frequencies to find cost-optimal one
    next_day = daily_ret.shift(-1)
    best_sharpe, best_period = -np.inf, 20
    for period in [10, 15, 20, 25, 30, 40]:
        inv_vol = 1.0 / daily_ret.rolling(60).std().replace(0, np.nan).fillna(1.0)
        rp = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(1/n)
        # Only rebalance every 'period' days
        rebal = rp.copy()
        for i in range(1, len(rebal)):
            if i % period != 0:
                rebal.iloc[i] = rebal.iloc[i-1]
        turnover = rebal.diff().abs().sum(axis=1)
        cost = 0.003 * turnover
        ret = (next_day * rebal).sum(axis=1) - cost
        ret = ret.iloc[60:].dropna()
        if len(ret) > 30 and ret.std() > 0:
            sh = float(ret.mean() / ret.std() * np.sqrt(252))
            if sh > best_sharpe:
                best_sharpe, best_period = sh, period

    MODEL["rebalance_period"] = best_period

    # 4. Learn trend filter lookback
    best_trend_sh, best_trend_lb = -np.inf, 100
    for lb in [50, 80, 100, 150, 200]:
        if len(close) < lb + 60:
            continue
        above = close > close.rolling(lb).mean()
        inv_vol = 1.0 / daily_ret.rolling(60).std().replace(0, np.nan).fillna(1.0)
        rp = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(1/n)
        tilt = above.astype(float)
        tilt = tilt.div(tilt.sum(axis=1).replace(0, 1), axis=0)
        combined = 0.70 * rp + 0.30 * tilt
        combined = combined.div(combined.abs().sum(axis=1).clip(lower=0.01), axis=0) * 0.92
        for i in range(1, len(combined)):
            if i % 20 != 0:
                combined.iloc[i] = combined.iloc[i-1]
        turnover = combined.diff().abs().sum(axis=1)
        cost = 0.003 * turnover
        ret = (next_day * combined).sum(axis=1) - cost
        ret = ret.iloc[lb:].dropna()
        if len(ret) > 30 and ret.std() > 0:
            sh = float(ret.mean() / ret.std() * np.sqrt(252))
            if sh > best_trend_sh:
                best_trend_sh, best_trend_lb = sh, lb

    MODEL["trend_lookback"] = best_trend_lb

    print(f"  Rebalance period: {MODEL['rebalance_period']} days")
    print(f"  Trend lookback: {MODEL['trend_lookback']} days")
    print(f"  Vol thresholds: median={MODEL['vol_median']:.3f}, "
          f"high={MODEL['vol_high']:.3f}, crisis={MODEL['vol_crisis']:.3f}")


# ─────────────────────────── STRATEGIES ───────────────────────────


def decide_equal_weight(history):
    """Baseline: 1/N in every stock, every day. Minimal turnover."""
    stocks = history["close"].columns
    return {s: 1 / len(stocks) for s in stocks}


def decide_momentum(history):
    """Buy the top-3 stocks by 20-day return."""
    close = history["close"]
    if len(close) < 21:
        return {}
    past_return = close.iloc[-1] / close.iloc[-21] - 1
    winners = past_return.nlargest(3).index
    return {stock: 1 / 3 for stock in winners}


def decide_risk_parity(history):
    """Inverse-volatility weighted, rebalanced every ~20 days."""
    close = history["close"]
    stocks = list(close.columns)
    n = len(stocks)

    if len(close) < 63:
        return {s: 1/n for s in stocks}

    if not hasattr(decide_risk_parity, '_cnt'):
        decide_risk_parity._cnt = 0
        decide_risk_parity._prev = None

    decide_risk_parity._cnt += 1
    rebal_period = MODEL.get("rebalance_period", 20)

    if decide_risk_parity._prev is not None and decide_risk_parity._cnt % rebal_period != 1:
        return decide_risk_parity._prev

    daily_ret = close.pct_change().dropna()
    vol = daily_ret.iloc[-60:].std()
    inv_vol = 1.0 / vol.replace(0, np.nan).fillna(0)
    total = inv_vol.sum()
    if total == 0:
        return {s: 1/n for s in stocks}
    weights = inv_vol / total * 0.92
    result = {s: float(weights[s]) for s in stocks}
    decide_risk_parity._prev = result
    return result


def decide_trend_rp(history):
    """Risk-parity base + gentle trend tilt: overweight uptrending stocks.
    Rebalances every ~20 days to minimize costs."""
    close = history["close"]
    stocks = list(close.columns)
    n = len(stocks)
    trend_lb = MODEL.get("trend_lookback", 100)

    if len(close) < max(63, trend_lb):
        return {s: 1/n for s in stocks}

    if not hasattr(decide_trend_rp, '_cnt'):
        decide_trend_rp._cnt = 0
        decide_trend_rp._prev = None

    decide_trend_rp._cnt += 1
    rebal_period = MODEL.get("rebalance_period", 20)

    if decide_trend_rp._prev is not None and decide_trend_rp._cnt % rebal_period != 1:
        return decide_trend_rp._prev

    daily_ret = close.pct_change().dropna()

    # Risk-parity base
    vol = daily_ret.iloc[-60:].std()
    inv_vol = 1.0 / vol.replace(0, np.nan).fillna(0)
    total_iv = inv_vol.sum()
    if total_iv == 0:
        rp = pd.Series(1/n, index=stocks)
    else:
        rp = inv_vol / total_iv

    # Trend tilt: stocks above their long-term MA get a boost
    ma = close.iloc[-trend_lb:].mean()
    above = (close.iloc[-1] > ma).astype(float)
    n_above = above.sum()
    if n_above > 0:
        trend_w = above / n_above
    else:
        trend_w = pd.Series(1/n, index=stocks)

    # Blend: 70% risk-parity + 30% trend tilt
    combined = 0.70 * rp + 0.30 * trend_w

    # Scale to 92% gross exposure
    gross = combined.abs().sum()
    if gross > 0:
        combined = combined / gross * 0.92

    result = {s: float(combined[s]) for s in stocks}
    decide_trend_rp._prev = result
    return result


def decide_vol_regime(history):
    """FINAL AGENT: Volatility-regime-aware risk-parity with trend tilt.

    Core idea: Stay invested in a diversified risk-parity portfolio,
    tilted toward uptrending stocks. Reduce exposure during market crises.
    Rebalance infrequently (~20 days) to minimize transaction costs.

    Scoring targets:
    - High return: stay invested in positively-drifting stocks
    - Good Sharpe: vol-weight for diversification
    - Low drawdown: crisis-mode de-risking
    - Low costs: infrequent rebalancing
    - High accuracy: trend filter biases toward winners
    - Good generalization: simple, few parameters
    """
    close = history["close"]
    stocks = list(close.columns)
    n = len(stocks)
    trend_lb = MODEL.get("trend_lookback", 100)

    if len(close) < max(63, trend_lb):
        return {s: 1/n for s in stocks}

    # Rebalance counter
    if not hasattr(decide_vol_regime, '_cnt'):
        decide_vol_regime._cnt = 0
        decide_vol_regime._prev = None

    decide_vol_regime._cnt += 1
    rebal_period = MODEL.get("rebalance_period", 20)

    # Check for volatility regime change
    daily_ret = close.pct_change().dropna()
    mkt_ret = daily_ret.mean(axis=1)

    force_rebalance = False
    if len(mkt_ret) >= 21:
        recent_vol = float(mkt_ret.iloc[-21:].std() * np.sqrt(252))
        vol_crisis = MODEL.get("vol_crisis", 0.30)
        vol_high = MODEL.get("vol_high", 0.22)

        # Force rebalance if entering/exiting crisis
        if decide_vol_regime._prev is not None:
            prev_exposure = sum(abs(v) for v in decide_vol_regime._prev.values())
            if recent_vol > vol_crisis and prev_exposure > 0.60:
                force_rebalance = True
            elif recent_vol < vol_high and prev_exposure < 0.60:
                force_rebalance = True
    else:
        recent_vol = MODEL.get("vol_median", 0.15)
        vol_crisis = MODEL.get("vol_crisis", 0.30)
        vol_high = MODEL.get("vol_high", 0.22)

    # Skip rebalance if not on schedule and no regime change
    if (decide_vol_regime._prev is not None
            and decide_vol_regime._cnt % rebal_period != 1
            and not force_rebalance):
        return decide_vol_regime._prev

    # ── Build target weights ──

    # 1. Risk-parity base
    vol = daily_ret.iloc[-60:].std()
    inv_vol = 1.0 / vol.replace(0, np.nan).fillna(0)
    total_iv = inv_vol.sum()
    if total_iv == 0:
        rp = pd.Series(1/n, index=stocks)
    else:
        rp = inv_vol / total_iv

    # 2. Trend tilt
    ma = close.iloc[-trend_lb:].mean()
    above = (close.iloc[-1] > ma).astype(float)
    n_above = above.sum()
    if n_above > 0:
        trend_w = above / n_above
    else:
        trend_w = pd.Series(1/n, index=stocks)

    # 3. Blend
    combined = 0.70 * rp + 0.30 * trend_w

    # 4. Volatility regime scaling
    if recent_vol > vol_crisis:
        target_exposure = 0.45  # crisis: significant de-risking
    elif recent_vol > vol_high:
        target_exposure = 0.70  # elevated vol: moderate de-risking
    else:
        target_exposure = 0.92  # normal: near-full investment

    # 5. Scale to target exposure
    gross = combined.abs().sum()
    if gross > 0:
        combined = combined / gross * target_exposure

    result = {s: float(combined[s]) for s in stocks}
    decide_vol_regime._prev = result
    return result


def decide_mean_reversion(history):
    """Cross-sectional mean reversion: buy recent losers, sell recent winners.
    Rebalances every ~10 days."""
    close = history["close"]
    stocks = list(close.columns)
    n = len(stocks)

    if len(close) < 40:
        return {s: 1/n for s in stocks}

    if not hasattr(decide_mean_reversion, '_cnt'):
        decide_mean_reversion._cnt = 0
        decide_mean_reversion._prev = None

    decide_mean_reversion._cnt += 1
    if decide_mean_reversion._prev is not None and decide_mean_reversion._cnt % 10 != 1:
        return decide_mean_reversion._prev

    daily_ret = close.pct_change().dropna()
    past_ret = close.iloc[-1] / close.iloc[-6] - 1
    vol = daily_ret.iloc[-20:].std()
    z = past_ret / (vol * np.sqrt(5)).replace(0, np.nan)
    z = z.fillna(0)
    ranks = z.rank(pct=True) - 0.5
    signal = -ranks
    signal = signal + 1/n  # positive bias since all stocks trend up
    gross = signal.abs().sum()
    if gross > 0:
        signal = signal / gross * 0.85
    result = {s: float(signal[s]) for s in stocks}
    decide_mean_reversion._prev = result
    return result


def decide_conservative(history):
    """Ultra-conservative: equal-weight with crash protection only.
    Rebalances monthly. Minimum turnover."""
    close = history["close"]
    stocks = list(close.columns)
    n = len(stocks)

    if len(close) < 30:
        return {s: 1/n for s in stocks}

    if not hasattr(decide_conservative, '_cnt'):
        decide_conservative._cnt = 0
        decide_conservative._prev = None

    decide_conservative._cnt += 1
    if decide_conservative._prev is not None and decide_conservative._cnt % 30 != 1:
        return decide_conservative._prev

    daily_ret = close.pct_change().dropna()
    mkt_21d = float(daily_ret.mean(axis=1).iloc[-21:].sum()) if len(daily_ret) >= 21 else 0.0

    if mkt_21d < -0.15:
        exposure = 0.50
    elif mkt_21d < -0.08:
        exposure = 0.70
    else:
        exposure = 0.92

    w = exposure / n
    result = {s: w for s in stocks}
    decide_conservative._prev = result
    return result


# ─────────────────────────── FINAL AGENT ───────────────────────────

decide = decide_vol_regime  # <- the agent we submit


# ─────────────────────────── MAIN ───────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    import engine

    data_path = Path(__file__).resolve().parent / "data" / "train.csv"
    if not data_path.exists():
        data_path = Path(__file__).resolve().parent / "train.csv"
    data = engine.load(data_path)

    # Reset all persistent state
    for fn in [decide_risk_parity, decide_trend_rp, decide_vol_regime,
               decide_mean_reversion, decide_conservative]:
        if hasattr(fn, '_cnt'):
            fn._cnt = 0
            fn._prev = None

    window = engine.train_agent(sys.modules[__name__], data)
    result = engine.backtest(decide, data, verbose=False)
    log = result["log"]
    inside = (log.index >= window[0]) & (log.index <= window[1])

    print(f"\n{TEAM_NAME}: trained on days {window[0]}-{window[1]}")
    print(f"Strategy: {decide.__name__}")
    print("\nInside the training window:")
    engine.print_metrics(engine.metrics(log[inside]))
    print("Outside the training window (validation):")
    engine.print_metrics(engine.metrics(log[~inside]))

    m_in = engine.metrics(log[inside])
    m_out = engine.metrics(log[~inside])
    print(f"\nGeneralization gap (val Sharpe - train Sharpe): "
          f"{m_out['sharpe'] - m_in['sharpe']:.3f}")
    print(f"Total costs: {log['cost'].sum()*100:.2f}% of capital")
    print(f"Errors: {int(log['error'].sum())}")
