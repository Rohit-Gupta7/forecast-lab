"""Command line entry point for the backtest and report."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .core import BacktestConfig, backtest, generate_demo, scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest retail demand forecasts")
    parser.add_argument("--data", type=Path, help="CSV with date, sku, units, promo columns")
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--min-train", type=int, default=180)
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--step", type=int, default=28)
    args = parser.parse_args()
    data = pd.read_csv(args.data) if args.data else generate_demo()
    predictions = backtest(data, BacktestConfig(args.min_train, args.horizon, args.step))
    summary = scores(predictions)
    args.out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.out / "predictions.csv", index=False)
    summary.to_csv(args.out / "scores.csv", index=False)
    if not args.data:
        data.to_csv(args.out / "synthetic_sales.csv", index=False)

    pivot = summary.pivot(index="sku", columns="model", values="wape_pct")
    ax = pivot.plot.bar(figsize=(10, 5), color=["#8e9aaf", "#24a879", "#4472c4"])
    ax.set_ylabel("WAPE (%) · lower is better")
    ax.set_xlabel("")
    ax.set_title("Rolling-origin demand forecasting · synthetic data")
    ax.grid(axis="y", alpha=.2)
    ax.legend(title="Model", frameon=False)
    plt.tight_layout()
    plt.savefig(args.out / "model_comparison.png", dpi=160)
    plt.close()
    print(summary.to_string(index=False))
    print(f"\nReports written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
