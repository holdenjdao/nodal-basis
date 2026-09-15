"""Walk-forward DART virtual-trading backtest -> results/backtest_dart.csv."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from nodal import backtest, basis, dart, data


def main() -> None:
    da = basis.filter_complete(data.da_panel())
    rt = data.rt_hourly_panel()
    d = dart.dart_panel(da, rt)
    print(f"dart panel: {d.shape[0]} hours x {d.shape[1]} locations "
          f"({d.index.min().date()} .. {d.index.max().date()})")

    primary = backtest.run_dart_backtest(d, **backtest.PRIMARY)
    print("\n=== primary (pre-registered) parameters:", backtest.PRIMARY, "===")
    for k, v in primary.stats.items():
        print(f"  {k:32s} {v:,.3f}" if isinstance(v, float) else f"  {k:32s} {v}")

    monthly = primary.pnl_daily.groupby(primary.pnl_daily.index.to_period("M")).sum()
    print("\nnet P&L by month ($/MW, summed over traded nodes):")
    print(monthly.round(0).to_string())

    table = backtest.sensitivity(d)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n=== sensitivity ===")
        cols = ["lookback_days", "entry", "cost", "net_pnl", "net_per_node_hour", "sharpe_daily_ann",
                "max_drawdown", "hit_rate_days", "top1pct_hours_share_of_gross", "avg_nodes_held", "primary"]
        print(table[cols].round(3).to_string(index=False))

    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "backtest_dart.csv", index=False)
    primary.pnl_daily.rename("net_pnl").to_csv(out / "backtest_dart_daily.csv")
    print(f"\nwrote {out / 'backtest_dart.csv'}")


if __name__ == "__main__":
    main()
