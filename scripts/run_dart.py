"""Run the DART bias analysis on the local store and write results/dart.csv."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from nodal import basis, dart, data


def main() -> None:
    da = data.da_panel()
    rt = data.rt_hourly_panel()
    da = basis.filter_complete(da)

    d = dart.dart_panel(da, rt)
    print(f"dart panel: {d.shape[0]} hours x {d.shape[1]} locations")
    print(f"range: {d.index.min()} .. {d.index.max()}")

    min_obs = max(120, int(0.7 * len(d)))
    maxlags = min(48, len(d) // 4)
    table = dart.dart_table(d, maxlags=maxlags, min_obs=min_obs)
    n_disc = int(table["discovery"].sum())
    print(f"\nnodes tested: {len(table)}   BH discoveries at 5% FDR: {n_disc}")

    with pd.option_context("display.width", 160, "display.max_columns", 10):
        print("\n=== strongest DA-rich (INC side) ===")
        print(table[table["mean_dart"] > 0].head(10).round(3))
        print("\n=== strongest DA-cheap (DEC side) ===")
        print(table[table["mean_dart"] < 0].head(10).round(3))

    persist = dart.split_persistence(d, min_obs=max(40, len(d) // 4))
    print(f"\nfirst-half vs second-half rank correlation (Spearman): "
          f"{persist.attrs['spearman']:.3f}")

    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "dart.csv")
    print(f"\nwrote {out / 'dart.csv'}")


if __name__ == "__main__":
    main()
