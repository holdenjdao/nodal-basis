"""Run the DART bias analysis on the local store.

Writes results/dart.csv (absolute DA - RT per node) and results/dart_hubrel.csv
(node dart minus hub dart: the locational part), plus month-to-month
persistence tables for both.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from nodal import basis, dart, data

OUT = Path(__file__).resolve().parents[1] / "results"


def report(d: pd.DataFrame, label: str, stem: str) -> None:
    min_obs = max(120, int(0.7 * len(d)))
    maxlags = min(48, len(d) // 4)
    table = dart.dart_table(d, maxlags=maxlags, min_obs=min_obs)
    n_disc = int(table["discovery"].sum())
    n_inc = int((table["discovery"] & (table["mean_dart"] > 0)).sum())
    print(f"\n##### {label} #####")
    print(f"nodes tested: {len(table)}   BH discoveries at 5% FDR: {n_disc} "
          f"({n_inc} DA-rich / {n_disc - n_inc} DA-cheap)")

    cols = ["n", "mean_dart", "mean_dart_raw", "median_dart", "share_da_rich", "t_hac", "p_bh"]
    with pd.option_context("display.width", 160, "display.max_columns", 12):
        print("\n=== strongest DA-rich (INC side) ===")
        print(table[table["mean_dart"] > 0].head(10)[cols].round(3))
        print("\n=== strongest DA-cheap (DEC side) ===")
        print(table[table["mean_dart"] < 0].head(10)[cols].round(3))

    persist = dart.split_persistence(d, min_obs=max(40, len(d) // 4))
    print(f"\nfirst-half vs second-half rank correlation (Spearman): {persist.attrs['spearman']:.3f}")

    if len(d) > 24 * 60:
        monthly = dart.monthly_persistence(d)
        with pd.option_context("display.width", 120):
            print("\n=== month-to-month persistence of the node premium cross-section ===")
            print(monthly.round(3).to_string())
            print(f"mean rho lag1: {monthly['rho_lag1'].mean():.3f}   lag2: {monthly['rho_lag2'].mean():.3f}"
                  f"   lag3: {monthly['rho_lag3'].mean():.3f}")
        monthly.to_csv(OUT / f"{stem}_monthly_persistence.csv")

    table.to_csv(OUT / f"{stem}.csv")
    print(f"\nwrote {OUT / f'{stem}.csv'}")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    da = basis.filter_complete(data.da_panel())
    rt = data.rt_hourly_panel()
    d = dart.dart_panel(da, rt)
    print(f"dart panel: {d.shape[0]} hours x {d.shape[1]} locations")
    print(f"range: {d.index.min()} .. {d.index.max()}")

    hub = d["HB_HUBAVG"].dropna()
    mean, t, p = dart._hac_test(dart.winsorize(hub.to_frame(), 0.01).iloc[:, 0].to_numpy(), 48)
    print(f"\nsystem-wide forward premium at HB_HUBAVG: mean DA-RT = ${mean:.2f}/MWh "
          f"(raw ${hub.mean():.2f}, median ${hub.median():.2f}), HAC t = {t:.2f}, "
          f"DA rich in {100 * (hub > 0).mean():.0f}% of hours")

    report(d, "ABSOLUTE DART (DA - RT per node)", "dart")
    report(dart.hub_relative(d), "HUB-RELATIVE DART (node dart - hub dart)", "dart_hubrel")


if __name__ == "__main__":
    main()
