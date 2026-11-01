# Harshad's employee — Agents War Trading Agent (Alpha Arena, Meraz 7.0)

> **Team Name**: Harshad's employee  
> **Submitted Strategy**: `decide_vol_regime`  
> **Selected Training Window**: `TRAIN_WINDOW = (252, 2771)` (Days 252–2771, 2,520 days = 10 years)  
> **AI Assistance**: Developed with Google Antigravity / Claude Opus 4.6 (Thinking)

---

## 1. Executive Summary

**Harshad's employee** uses a **volatility-regime-aware inverse-volatility portfolio with trend tilt and infrequent rebalancing**. The design philosophy is simple: stay invested in diversified, positively-drifting stocks and minimize the amount of trading you do, because at 0.3% per trade, turnover is the #1 destroyer of returns.

### Key Results (Window 252–2771, 0.3% cost)

| Metric | Inside Training | Outside (Validation) |
|---|---|---|
| **Total Return** | +626% | **+145%** |
| **Sharpe Ratio** | 1.34 | **1.52** |
| **Sortino Ratio** | 2.00 | **2.20** |
| **Calmar Ratio** | 1.21 | **0.86** |
| **Max Drawdown** | -18.1% | **-25.5%** |
| **Costs Paid** | 6.54% | **2.33%** |
| **Avg Exposure** | 87.2% | **90.7%** |
| **Generalization Gap** | — | **+0.18** |
| **Errors** | 0 | **0** |
| **Total Runtime** | — | **6.02 seconds** |

> [!IMPORTANT]
> The validation Sharpe (1.52) is **higher** than the in-sample Sharpe (1.34), giving a positive generalization gap of +0.18. This means the strategy does not overfit to its training data.

---

## 2. Strategy Architecture

### Core Idea: Keep It Simple, Stay Invested

All 12 stocks in the dataset have positive long-term drift (17–35% annualized). The market's equal-weight portfolio compounds at ~25% per year. At 0.3% transaction costs, active daily trading can easily consume 50–130% of capital over a multi-year period (as demonstrated by our momentum strategy, which lost 58% of its value due to 132% in fees).

The winning formula is:
1. **Stay invested** — capture the positive drift
2. **Diversify by volatility** — lower-vol stocks get higher weight (risk parity)
3. **Tilt toward trends** — gently overweight stocks above their 200-day moving average
4. **Rebalance rarely** — only every ~20 trading days
5. **De-risk in crises** — reduce exposure when market volatility spikes

### How `decide_vol_regime` Works

```
Each rebalance day (every ~20 days):
  1. Compute 60-day rolling volatility for each stock
  2. Set base weights ∝ 1/volatility (risk parity)
  3. Check which stocks are above their 200-day moving average
  4. Blend: 70% risk-parity + 30% trend-above-MA tilt
  5. Check market volatility regime:
     - Normal (vol < 25.6%):  target 92% exposure
     - Elevated (vol > 25.6%): target 70% exposure
     - Crisis (vol > 32.5%):   target 45% exposure
  6. Scale weights to target exposure

On non-rebalance days: hold previous weights (zero turnover)
Exception: force rebalance when entering/exiting a crisis regime
```

### Why No ML / Complex Models

We tested a RidgeCV cross-sectional alpha model and found it **reduced** performance after costs. The ML overlay generates small alpha signals that require frequent rebalancing to act on, and the 0.3% cost per trade wipes out any edge. Simpler strategies generalize better to unseen market regimes.

---

## 3. Training Window Selection

### Why `TRAIN_WINDOW = (252, 2771)`?

We tested all 6 standard 10-year windows using `python compare.py windows`:

| Window | Val Return | Val Sharpe | Val Calmar | Val MaxDD | Costs | Gen Gap |
|---|---|---|---|---|---|---|
| days 0-2519 | +59% | 0.80 | 0.41 | -25.2% | 2.92% | -0.84 |
| **days 252-2771** ← | **+153%** | **1.57** | **0.91** | **-25.1%** | **2.38%** | **+0.18** |
| days 504-3023 | +170% | 1.52 | 0.96 | -25.6% | 2.69% | +0.15 |
| days 756-3275 | +216% | 1.64 | 1.17 | -24.9% | 2.45% | +0.28 |
| days 1008-3527 | +291% | 1.96 | 1.40 | -25.3% | 2.90% | +0.91 |
| days 1188-3707 | +359% | 2.24 | 3.10 | -13.0% | 2.49% | +1.30 |

> [!WARNING]  
> Later windows (1008-3527, 1188-3707) show higher validation Sharpe, but this is misleading. Their "validation" period is days 50-1007 — the easiest bull market in the data (Sharpe 2–4.5). The equal-weight benchmark also gets Sharpe 2.04 on those same days.

### Selection Rationale

1. **Balanced validation**: Window (252, 2771) validates on both early data (days 50-251) AND the hard late period (days 2772-3707). This gives an honest generalization test.

2. **Hard-period performance**: We independently tested each window's performance on the hard period (days 2520-3707). Window (252, 2771) achieved the **best hard-period Sharpe of 0.88**.

3. **Good parameter calibration**: The learned volatility thresholds (crisis=32.5%) reflect a market that includes both strong rallies and corrections — a balanced calibration for unknown future conditions.

4. **Moderate, positive generalization gap**: +0.18 means the strategy works slightly *better* out of sample than in-sample, without being suspiciously good (which would indicate the validation was too easy).

---

## 4. Strategy Comparison (`comparison.csv`)

All strategies tested on validation data (days outside training window 252-2771), with 0.3% transaction cost:

| Strategy | Val Return | Sharpe | Calmar | Max DD | Costs | Gen Gap |
|---|---|---|---|---|---|---|
| **decide_vol_regime** ← submitted | +145% | **1.52** | 0.86 | **-25.5%** | 2.33% | **+0.18** |
| decide_conservative | +153% | 1.42 | 0.82 | -27.7% | 0.41% | +0.02 |
| decide_risk_parity | +154% | 1.35 | 0.71 | -32.5% | 1.76% | -0.07 |
| decide_trend_rp | +147% | 1.30 | 0.68 | -32.9% | 2.32% | -0.09 |
| BENCHMARK equal weight | +169% | 1.30 | 0.71 | -34.4% | 0.30% | -0.11 |
| decide_mean_reversion | +6% | 0.20 | 0.07 | -20.8% | **39.8%** | +0.06 |
| decide_momentum | **-58%** | -0.77 | -0.26 | **-69%** | **132%** | -0.43 |

### Key takeaways:
- **Vol-regime is #1 by Sharpe** (1.52) and has the best generalization (+0.18)
- **Vol-regime has the best max drawdown** among investable strategies (-25.5% vs -34.4% for EW)
- **Momentum is destroyed** by 0.3% costs: 132% in fees, loses more than half the portfolio
- **Mean reversion** also fails: 39.8% in fees despite rebalancing every 10 days
- The equal-weight benchmark has higher raw return but **worse Sharpe and much worse drawdown**

---

## 5. Risk Management

1. **Volatility regime detection**: 21-day rolling market volatility compared to learned thresholds from training. Exposure reduced to 70% in elevated regimes, 45% in crisis.

2. **Inverse-volatility weighting**: Naturally allocates less to volatile stocks, preventing any single position from dominating risk.

3. **Maximum per-stock weight**: ~21% (enforced by the risk-parity + trend structure), well below the 1.0 per-stock limit.

4. **Infrequent rebalancing**: Only rebalances every 20 trading days. Forces rebalance only on regime transitions (entering/exiting crisis mode).

5. **No leverage**: Gross exposure capped at 92% in normal times, with a guaranteed floor of 45% (always above the 30% minimum rule).

---

## 6. Assumptions & Limitations

1. **Stationarity assumption**: We assume the cross-sectional properties of volatility and trend persistence continue into the secret period. This is a standard assumption but could break if the market fundamentally changes character.

2. **No short positions used**: Although shorts are allowed, our strategy only holds long positions. This means we cannot profit from declining stocks.

3. **Regime thresholds are static**: The vol thresholds are learned once in train() and don't adapt during the test period. A sustained shift in volatility regime could make the thresholds stale.

4. **The strategy cannot beat buy-and-hold in a pure bull market**: By de-risking in high-vol periods, we sacrifice some upside. This is a deliberate tradeoff for lower drawdown and better risk-adjusted returns.

5. **Window bias uncertainty**: Despite our best efforts, we cannot know which window truly prepares best for the secret period. Our analysis suggests (252, 2771) is a balanced choice, but this could be wrong.

---

## 7. Files

| File | Description |
|---|---|
| `strategy.py` | Trading agent: TRAIN_WINDOW, train(), 7 decide functions, final decide = decide_vol_regime |
| `visualization.py` | Portfolio growth graph with 4 lines (benchmark + top 3 strategies) |
| `engine.py` | Simulator (organiser's code, COST_PER_TRADE = 0.003) |
| `compare.py` | Strategy and window comparison tool (organiser's code) |
| `comparison.csv` | Strategy comparison results |
| `windows.csv` | Training window comparison results |
| `portfolio_growth.png` | Portfolio growth visualization |

---

## 8. Presentation Outline (3-5 Slides)

### Slide 1: One-Sentence Summary
*"Harshad's employee stays invested in a volatility-balanced portfolio, tilts toward uptrending stocks, and trades only twice a month — because at 0.3% cost, the best trade is the one you don't make."*

### Slide 2: Strategies We Tried (`comparison.csv`)
- Table of 7 strategies + benchmark
- Key insight: momentum loses 58% to fees; simple beats complex
- Vol-regime: best Sharpe (1.52), best drawdown (-25.5%), best generalization (+0.18)

### Slide 3: Training Window Selection (`windows.csv`)
- Window (252, 2771) chosen for balanced validation across both easy and hard periods
- Later windows show inflated validation Sharpe because they test on easy early data
- Hard-period analysis: window (252, 2771) achieves best hard-period Sharpe of 0.88

### Slide 4: Risk Management
- Volatility regime detection: 92% → 70% → 45% exposure scaling
- Inverse-vol weighting: automatic diversification
- 20-day rebalancing: keeps costs at 2.33% vs 132% for daily strategies

### Slide 5: Learnings
- At 0.3% transaction cost, **turnover control matters more than alpha generation**
- Simple strategies generalize; ML/complex strategies overfit and churn
- The biggest risk is not market drops — it's paying fees to react to noise
