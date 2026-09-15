"""Walk-forward DART virtual-trading backtests.

Absolute (INC/DEC at the node) and hub-relative (node leg against the hub,
stripping the market-wide forward premium). Writes results/backtest_dart*.csv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from nodal import backtest, basis, dart, data

OUT = Path(__file__).resolve().parents[1] / "results"


def report(d: pd.DataFrame, label: str, stem: str, hub_relative: bool) -> None:
    primary = backtest.run_dart_backtest(d, **backtest.PRIMARY, hub_relative=hub_relative)
    print(f"\n##### {label} #####")
    print("=== primary (pre-registered) parameters:", backtest.PRIMARY, "===")
    for k, v in primary.stats.items():
        print(f"  {k:32s} {v:,.3f}" if isinstance(v, float) else f"  {k:32s} {v}")

    monthly = primary.pnl_daily.groupby(primary.pnl_daily.index.to_period("M")).sum()
    print("\nnet P&L by month ($/MW, summed over traded nodes):")
    print(monthly.round(0).to_string())

    table = backtest.sensitivity(d, hub_relative=hub_relative)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n=== sensitivity ===")
        cols = ["lookback_days", "entry", "cost", "net_pnl", "net_per_node_hour", "sharpe_daily_ann",
                "max_drawdown", "hit_rate_days", "top1pct_hours_share_of_gross", "avg_nodes_held", "primary"]
        print(table[cols].round(3).to_string(index=False))

    table.to_csv(OUT / f"{stem}.csv", index=False)
    primary.pnl_daily.rename("net_pnl").to_csv(OUT / f"{stem}_daily.csv")
    print(f"\nwrote {OUT / f'{stem}.csv'}")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    da = basis.filter_complete(data.da_panel())
    rt = data.rt_hourly_panel()
    d = dart.dart_panel(da, rt)
    print(f"dart panel: {d.shape[0]} hours x {d.shape[1]} locations "
          f"({d.index.min().date()} .. {d.index.max().date()})")
    report(d, "ABSOLUTE DART: INC/DEC at node", "backtest_dart", hub_relative=False)
    report(d, "HUB-RELATIVE DART: node leg vs hub leg", "backtest_dart_hubrel", hub_relative=True)


if __name__ == "__main__":
    main()
