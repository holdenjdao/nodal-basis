"""DART bias: day-ahead minus real-time premia per settlement point.

Convention: dart = DA - RT (hourly, RT averaged from 15-min intervals).
Positive mean dart means day-ahead systematically clears rich at that node —
gross of costs, that is what an INC (virtual supply: sell DA, buy back RT)
would have captured; negative dart favors DECs.

Hourly darts are strongly autocorrelated (diurnal structure, multi-hour
congestion events), so plain t-stats overstate significance. We use
Newey-West (HAC) standard errors, then Benjamini-Hochberg across the node
universe to control the false discovery rate of the cross-sectional scan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


def dart_panel(da: pd.DataFrame, rt_hourly: pd.DataFrame) -> pd.DataFrame:
    """Aligned DA - RT panel on the intersection of hours and locations."""
    idx = da.index.intersection(rt_hourly.index)
    cols = da.columns.intersection(rt_hourly.columns)
    return da.loc[idx, cols] - rt_hourly.loc[idx, cols]


def _hac_test(series: np.ndarray, maxlags: int) -> tuple[float, float, float]:
    """Mean, HAC t-stat, and p-value for H0: mean == 0."""
    model = sm.OLS(series, np.ones_like(series))
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(res.params[0]), float(res.tvalues[0]), float(res.pvalues[0])


def dart_table(
    dart: pd.DataFrame,
    maxlags: int = 48,
    min_obs: int = 400,
    fdr: float = 0.05,
) -> pd.DataFrame:
    """Per-node DART premium with HAC inference and BH discovery flags.

    Returns one row per location: n, mean_dart, t_hac, p, p_bh, discovery,
    plus context columns (mean |dart|, share of hours DA rich).
    """
    rows = []
    for loc in dart.columns:
        s = dart[loc].dropna().to_numpy()
        if len(s) < min_obs:
            continue
        mean, t, p = _hac_test(s, maxlags)
        rows.append(
            {
                "location": loc,
                "n": len(s),
                "mean_dart": mean,
                "abs_dart": float(np.mean(np.abs(s))),
                "share_da_rich": float(np.mean(s > 0)),
                "t_hac": t,
                "p": p,
            }
        )
    if not rows:
        raise ValueError(
            f"no location has >= {min_obs} aligned DA/RT hours; "
            "lower min_obs or fetch more overlapping history"
        )
    table = pd.DataFrame(rows).set_index("location")
    reject, p_bh, _, _ = multipletests(table["p"], alpha=fdr, method="fdr_bh")
    table["p_bh"] = p_bh
    table["discovery"] = reject
    return table.sort_values("t_hac", key=lambda s: s.abs(), ascending=False)


def dart_by_hour(dart: pd.DataFrame) -> pd.DataFrame:
    """Mean dart per (location, hour-of-day): where in the day the bias lives."""
    stacked = dart.stack().rename("dart").reset_index()
    stacked.columns = ["interval", "location", "dart"]
    stacked["hour"] = stacked["interval"].dt.hour
    return stacked.pivot_table(index="location", columns="hour", values="dart")


def split_persistence(dart: pd.DataFrame, min_obs: int = 200) -> pd.DataFrame:
    """First-half vs second-half mean dart per node.

    If node premia are real structure rather than noise, the cross-section of
    first-half means should predict the second half (positive rank
    correlation). Returns per-node halves plus the Spearman rho as attrs.
    """
    halfway = dart.index[len(dart.index) // 2]
    first = dart.loc[dart.index < halfway].mean()
    second = dart.loc[dart.index >= halfway].mean()
    counts1 = dart.loc[dart.index < halfway].notna().sum()
    counts2 = dart.loc[dart.index >= halfway].notna().sum()
    ok = (counts1 >= min_obs) & (counts2 >= min_obs)
    out = pd.DataFrame({"first_half": first[ok], "second_half": second[ok]})
    out.attrs["spearman"] = float(out["first_half"].corr(out["second_half"], method="spearman"))
    return out
