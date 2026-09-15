"""Build node profiles -> results/node_profiles.csv and print the interesting slices.

Usage: python scripts/run_profiles.py [start_date] [end_date]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from nodal import basis, data, factors, nodes


def main() -> None:
    da = data.da_panel()
    if len(sys.argv) > 1:
        da = da.loc[da.index >= pd.Timestamp(sys.argv[1], tz="US/Central")]
    if len(sys.argv) > 2:
        da = da.loc[da.index < pd.Timestamp(sys.argv[2], tz="US/Central")]
    da = basis.filter_complete(da)
    b = basis.basis_panel(da)
    print(f"basis panel: {b.shape[0]} hours x {b.shape[1]} locations "
          f"({b.index.min().date()} .. {b.index.max().date()})")

    model = factors.fit(b, n_factors=8)
    prof = nodes.profile_table(b, model)

    with pd.option_context("display.width", 170, "display.max_columns", 12):
        print("\n=== asset classes: mean basis and spike share ===")
        print(prof.groupby("asset")[["mean_basis", "abs_basis", "spike_share", "factor_r2"]]
              .agg(["mean", "count"]).round(3).to_string())

        print("\n=== most idiosyncratic nodes (lowest factor R^2, high dispersion) ===")
        idio = prof[prof["std_basis"] > prof["std_basis"].median()].nsmallest(10, "factor_r2")
        print(idio[["asset", "mean_basis", "std_basis", "spike_share", "factor_r2", "resid_std"]]
              .round(3).to_string())

        print("\n=== spikiest nodes ===")
        print(prof.nlargest(10, "spike_share")[["asset", "mean_basis", "std_basis", "spike_share",
                                                "peak_hour", "trough_hour"]].round(3).to_string())

    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    prof.to_csv(out / "node_profiles.csv")
    print(f"\nwrote {out / 'node_profiles.csv'}")


if __name__ == "__main__":
    main()
