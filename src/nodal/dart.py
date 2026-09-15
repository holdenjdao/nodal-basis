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


def hub_relative(dart: pd.DataFrame, hub: str = "HB_HUBAVG") -> pd.DataFrame:
    """Node dart minus hub dart: the day-ahead mispricing of the *congestion*
    component alone.

    Day-ahead clears above real-time almost everywhere (the electricity
    forward premium), so absolute darts are one market-wide fact wearing
    960 node names. Subtracting the hub's dart isolates the locational part:
    (DA_node - RT_node) - (DA_hub - RT_hub) = DA basis - RT basis. Traded as
    an INC at the node against a DEC at the hub, it is immune to the
    system-wide premium and to system-wide RT spikes.
    """
    if hub not in dart.columns:
        raise KeyError(f"hub column {hub!r} not in dart panel")
    return dart.sub(dart[hub], axis=0).drop(columns=[hub])


def _hac_test(series: np.ndarray, maxlags: int) -> tuple[float, float, float]:
    """Mean, HAC t-stat, and p-value for H0: mean == 0."""
    model = sm.OLS(series, np.ones_like(series))
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(res.params[0]), float(res.tvalues[0]), float(res.pvalues[0])


def winsorize(panel: pd.DataFrame, q: float = 0.01) -> pd.DataFrame:
    """Clip each column at its own q / 1-q quantiles."""
    return panel.clip(lower=panel.quantile(q), upper=panel.quantile(1 - q), axis=1)


def dart_table(
    dart: pd.DataFrame,
    maxlags: int = 48,
    min_obs: int = 400,
    fdr: float = 0.05,
    winsor: float = 0.01,
) -> pd.DataFrame:
    """Per-node DART premium with HAC inference and BH discovery flags.

    Inference runs on the winsorized series: HAC handles autocorrelation,
    not a single $20,000 hour. The raw mean is reported alongside so the
    tail's contribution is visible rather than hidden.

    Returns one row per location: n, mean_dart (winsorized), mean_dart_raw,
    median_dart, t_hac, p, p_bh, discovery, plus context columns.
    """
    clipped = winsorize(dart, winsor) if winsor else dart
    rows = []
    for loc in dart.columns:
        raw = dart[loc].dropna()
        if len(raw) < min_obs:
            continue
        s = clipped[loc].dropna().to_numpy()
        mean, t, p = _hac_test(s, maxlags)
        rows.append(
            {
                "location": loc,
                "n": len(s),
                "mean_dart": mean,
                "mean_dart_raw": float(raw.mean()),
                "median_dart": float(raw.median()),
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


def split_persistence(dart: pd.DataFrame, min_obs: int = 200, winsor: float = 0.01) -> pd.DataFrame:
    """First-half vs second-half (winsorized) mean dart per node.

    If node premia are real structure rather than noise, the cross-section of
    first-half means should predict the second half (positive rank
    correlation). Returns per-node halves plus the Spearman rho as attrs.
    """
    d = winsorize(dart, winsor) if winsor else dart
    halfway = d.index[len(d.index) // 2]
    first = d.loc[d.index < halfway].mean()
    second = d.loc[d.index >= halfway].mean()
    counts1 = d.loc[d.index < halfway].notna().sum()
    counts2 = d.loc[d.index >= halfway].notna().sum()
    ok = (counts1 >= min_obs) & (counts2 >= min_obs)
    out = pd.DataFrame({"first_half": first[ok], "second_half": second[ok]})
    out.attrs["spearman"] = float(out["first_half"].corr(out["second_half"], method="spearman"))
    return out


def monthly_persistence(dart: pd.DataFrame, min_obs: int = 300, winsor: float = 0.01) -> pd.DataFrame:
    """Does last month's cross-section of node premia predict this month's?

    Per-node winsorized mean dart by calendar month, then the Spearman rank
    correlation between each consecutive pair of months. A trader acting on
    last month's ranking needs this to be reliably positive; the lag-2 and
    lag-3 columns say how fast the information decays.
    """
    d = winsorize(dart, winsor) if winsor else dart
    month = d.index.tz_localize(None).to_period("M")
    means = d.groupby(month).mean()
    counts = d.groupby(month).count()
    means = means.where(counts >= min_obs)
    months = means.index
    rows = []
    for i in range(1, len(months)):
        row = {"month": str(months[i])}
        for lag in (1, 2, 3):
            if i - lag < 0:
                row[f"rho_lag{lag}"] = np.nan
                continue
            pair = pd.concat([means.iloc[i - lag], means.iloc[i]], axis=1).dropna()
            row[f"rho_lag{lag}"] = float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")) if len(pair) > 30 else np.nan
        row["nodes"] = int(means.iloc[i].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("month")
