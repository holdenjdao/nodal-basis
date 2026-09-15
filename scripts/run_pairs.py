"""Scan locational spread pairs -> results/pairs.csv.

Usage: python scripts/run_pairs.py [start_date] [end_date]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from nodal import basis, data, factors, pairs


def main() -> None:
    da = data.da_panel()
    if len(sys.argv) > 1:
        da = da.loc[da.index >= pd.Timestamp(sys.argv[1], tz="US/Central")]
    if len(sys.argv) > 2:
        da = da.loc[da.index < pd.Timestamp(sys.argv[2], tz="US/Central")]
    da = basis.filter_complete(da)
    b = basis.basis_panel(da)
    print(f"basis panel: {b.shape[0]} hours x {b.shape[1]} locations")

    model = factors.fit(b, n_factors=8)
    cands = pairs.candidate_pairs(model.loadings, k=5)
    print(f"candidate pairs (5-NN in loading space): {len(cands)}")

    table = pairs.scan(b, cands)
    tested = len(table)
    disc = table[table["discovery"]]
    print(f"pairs tested: {tested}   BH discoveries at 5% FDR: {len(disc)}")

    # persistence: among discoveries, does the in-sample half-life survive out of sample?
    ok = disc[np.isfinite(disc["half_life_oos"])]
    if len(ok):
        rho = ok["half_life_is"].corr(ok["half_life_oos"], method="spearman")
        still_fast = (ok["half_life_oos"] < 2 * ok["half_life_is"]).mean()
        print(f"discoveries with finite OOS half-life: {len(ok)}  "
              f"IS/OOS half-life rank corr: {rho:.3f}  "
              f"share still reverting at <2x IS speed: {still_fast:.2f}")

    # economic bar: the *typical* hour has to move enough to clear costs, and
    # dislocations have to happen often enough (in and out of sample) to harvest
    min_sigma, min_opp = 3.0, 0.10
    econ = disc[(disc["robust_sigma"] >= min_sigma) & (disc["opportunity_share"] >= min_opp)
                & (disc["opportunity_share_oos"] >= min_opp) & np.isfinite(disc["half_life_oos"])]
    print(f"\nstationary pairs with robust sigma >= ${min_sigma} and >= {min_opp:.0%} of hours "
          f"dislocated > $5 (in and out of sample): {len(econ)}")

    with pd.option_context("display.width", 190, "display.max_columns", 14):
        print("\n=== top by reversion yield (robust $ of spread reverting per hour) ===")
        cols = ["spread_median", "robust_sigma", "spread_std", "opportunity_share", "opportunity_share_oos",
                "half_life_is", "half_life_oos", "reversion_yield"]
        print(econ.sort_values("reversion_yield", ascending=False).head(15)[cols].round(3).to_string())

    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "pairs.csv")
    print(f"\nwrote {out / 'pairs.csv'}")


if __name__ == "__main__":
    main()
