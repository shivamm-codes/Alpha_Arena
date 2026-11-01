"""
visualization.py - Cumulative Portfolio Growth Graph
Alpha Arena (Meraz 7.0) - Team Harshad's employee

Requirements:
  - X-axis: Day (data has no dates)
  - Y-axis: Portfolio value in Rs, starting from Rs 1,00,000
  - At most 10 years (2520 days) on one graph
  - 4 lines: equal-weight benchmark + top 3 strategies
  - Saves portfolio_growth.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

import engine
import strategy

HERE = Path(__file__).resolve().parent

# The 3 best strategies by validation Sharpe (with 0.3% costs)
BEST_STRATEGIES = [
    ("Harshad's employee: Vol-Regime (submitted)", strategy.decide_vol_regime),
    ("Conservative EW", strategy.decide_conservative),
    ("Risk Parity", strategy.decide_risk_parity),
]

COLORS = {
    "Harshad's employee: Vol-Regime (submitted)": "#2196F3",  # blue
    "Conservative EW": "#4CAF50",                       # green
    "Risk Parity": "#9C27B0",                           # purple
    "Benchmark (Equal Weight)": "#FF9800",              # orange
}


def _reset_state():
    """Reset all persistent state in strategy functions."""
    for fn in [strategy.decide_risk_parity, strategy.decide_trend_rp,
               strategy.decide_vol_regime, strategy.decide_mean_reversion,
               strategy.decide_conservative]:
        if hasattr(fn, '_cnt'):
            fn._cnt = 0
            fn._prev = None


def plot_portfolio_growth():
    # 1. Load data
    data_path = HERE / "data" / "train.csv"
    if not data_path.exists():
        data_path = HERE / "train.csv"
    data = engine.load(data_path)

    # 2. Train agent on TRAIN_WINDOW
    _reset_state()
    window = engine.train_agent(strategy, data)
    print(f"Agent trained on days {window[0]}-{window[1]}")

    # 3. Select the 10-year window to display (training window)
    plot_start = window[0]
    plot_end = min(window[1], window[0] + 2519)  # at most 2520 days

    # 4. Run backtests and collect equity curves
    curves = {}

    # -- Benchmark --
    res_bm = engine.backtest(engine.equal_weight, data, verbose=False)
    curves["Benchmark (Equal Weight)"] = res_bm["equity"]

    # -- Top 3 strategies --
    for label, fn in BEST_STRATEGIES:
        _reset_state()
        engine.train_agent(strategy, data)  # retrain to reset MODEL state
        res = engine.backtest(fn, data, verbose=False)
        curves[label] = res["equity"]

    # 5. Slice to plot window and normalize to Rs 1,00,000
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 10),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    for label, equity in curves.items():
        sub = equity.loc[plot_start:plot_end]
        if len(sub) == 0:
            continue
        normed = 100000 * (sub / sub.iloc[0])
        color = COLORS.get(label, "gray")
        lw = 2.2 if "submitted" in label else 1.5
        ls = "--" if "Benchmark" in label else "-"
        ax1.plot(normed.index, normed, label=label,
                 color=color, linewidth=lw, linestyle=ls)

        # Drawdown
        peak = normed.cummax()
        dd = (normed - peak) / peak * 100
        if "submitted" in label:
            ax2.fill_between(dd.index, dd, 0, color=color, alpha=0.3,
                             label=f"{label} DD")
        elif "Benchmark" in label:
            ax2.plot(dd.index, dd, color=color, linestyle="--",
                     linewidth=1.0, label=f"{label} DD")

    # Annotate final values for all curves
    for label, equity in curves.items():
        sub = equity.loc[plot_start:plot_end]
        if len(sub) == 0:
            continue
        normed = 100000 * (sub / sub.iloc[0])
        final = normed.iloc[-1]
        color = COLORS.get(label, "gray")
        short_label = label.split(":")[0] if ":" in label else label.split("(")[0].strip()
        ax1.annotate(
            f"Rs {final:,.0f} ({(final/100000-1)*100:+.0f}%)",
            xy=(normed.index[-1], final),
            fontsize=8, fontweight="bold", color=color,
            ha="right", va="bottom")

    ax1.axhline(y=100000, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)

    ax1.set_title(
        f"Portfolio Growth: Top 3 Strategies vs Benchmark\n"
        f"(Training Window: Days {window[0]}-{window[1]}, Cost: 0.3% per trade)",
        fontsize=14, fontweight="bold", pad=15)
    ax1.set_ylabel("Portfolio Value (Rs)", fontsize=11, fontweight="bold")
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"Rs {int(x):,}"))
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2.set_ylabel("Drawdown (%)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Day", fontsize=11, fontweight="bold")
    ax2.legend(loc="lower left", frameon=True, fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    out_path = HERE / "portfolio_growth.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Graph saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    plot_portfolio_growth()
