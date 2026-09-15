"""Locational spread pairs: stationary node-minus-node spreads.

Two nodes with similar constraint exposure should have a spread that mean-
reverts; a spread that is stationary with a short half-life is a candidate
inefficiency. Pipeline:

    1. candidates   nearest neighbours in factor-loading space (a physical
                    prior, and it keeps the scan far below the ~600k possible
                    pairs so the multiple-testing burden stays sane)
    2. test         ADF stationarity test on the spread a - b with the hedge
                    ratio fixed at 1. A locational spread is a physical MWh-
                    for-MWh quantity; an OLS hedge ratio would just fit noise.
    3. dynamics     AR(1) fit of the spread -> OU half-life, equilibrium, sigma
    4. validate     out-of-sample: is the half-life still short, and does the
                    spread still revert, on data the test never saw?
    5. correct      Benjamini-Hochberg across every pair tested

Pairs of settlement points that share an electrical bus have identical prices
and a zero spread; those are excluded by a minimum spread-std filter rather
than celebrated as perfect cointegration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller


def candidate_pairs(loadings: pd.DataFrame, k: int = 5) -> list[tuple[str, str]]:
    """Each node's k nearest neighbours in loading space, de-duplicated."""
    x = loadings.dropna().to_numpy()
    names = loadings.dropna().index.to_list()
    sq = (x**2).sum(axis=1)
    dist = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * x @ x.T, 0.0))
    np.fill_diagonal(dist, np.inf)
    pairs: set[tuple[str, str]] = set()
    for i, row in enumerate(dist):
        for j in np.argpartition(row, k)[:k]:
            a, b = sorted((names[i], names[int(j)]))
            pairs.add((a, b))
    return sorted(pairs)


def ou_fit(spread: pd.Series) -> dict:
    """AR(1) on levels: ds_t = a + b*s_{t-1} + e  ->  theta=-b, mu=-a/b, half-life=ln2/theta."""
    s = spread.dropna()
    lag = s.shift(1).dropna()
    ds = s.diff().dropna()
    x = np.column_stack([np.ones(len(lag)), lag.to_numpy()])
    coef, *_ = np.linalg.lstsq(x, ds.to_numpy(), rcond=None)
    a, b = coef
    theta = -b
    half_life = np.log(2) / theta if theta > 0 else np.inf
    mu = -a / b if b != 0 else np.nan
    return {"theta": theta, "half_life": half_life, "mu": mu, "sigma": float(ds.std())}


def robust_sigma(x: pd.Series) -> float:
    """1.4826 * MAD: the std of the typical hour, immune to a single $20,000 spike."""
    med = x.median()
    return float(1.4826 * (x - med).abs().median())


def test_pair(
    a: pd.Series,
    b: pd.Series,
    split: float = 0.7,
    winsor: float = 0.01,
    dislocation: float = 5.0,
) -> dict:
    """Stationarity + OU dynamics in-sample, dynamics again out-of-sample.

    Tests run on the winsorized spread: one spike hour otherwise dominates
    the variance, the AR(1) fit, and any ranking. `dislocation` is the $/MWh
    departure from the median that counts as a tradeable opportunity; its
    frequency is what a strategy can actually harvest.
    """
    raw = (a - b).dropna()
    lo, hi = raw.quantile(winsor), raw.quantile(1 - winsor)
    spread = raw.clip(lo, hi)
    n = len(spread)
    cut = int(n * split)
    ins, oos = spread.iloc[:cut], spread.iloc[cut:]
    # fixed daily lag: hourly data, and AIC search over ~40 lags x 10k rows x
    # thousands of pairs is the difference between minutes and an hour
    adf_stat, adf_p, *_ = adfuller(ins.to_numpy(), maxlag=24, autolag=None, result_object=False)
    fit_in = ou_fit(ins)
    fit_out = ou_fit(oos) if len(oos) > 50 else {"half_life": np.nan, "theta": np.nan}
    sig_in = robust_sigma(ins)
    med_in = float(ins.median())
    return {
        "n": n,
        "spread_median": med_in,
        "spread_std": float(ins.std()),
        "robust_sigma": sig_in,
        "opportunity_share": float(((ins - med_in).abs() > dislocation).mean()),
        "opportunity_share_oos": float(((oos - med_in).abs() > dislocation).mean()) if len(oos) else np.nan,
        "adf_stat": float(adf_stat),
        "adf_p": float(adf_p),
        "half_life_is": fit_in["half_life"],
        "half_life_oos": fit_out["half_life"],
        "theta_is": fit_in["theta"],
        "theta_oos": fit_out["theta"],
        # typical dollars of spread that revert per hour: stationarity is cheap, size x speed is not
        "reversion_yield": sig_in / fit_in["half_life"] if np.isfinite(fit_in["half_life"]) else 0.0,
    }


def scan(
    basis: pd.DataFrame,
    pairs: list[tuple[str, str]],
    min_spread_std: float = 0.75,
    min_obs: int = 300,
    fdr: float = 0.05,
) -> pd.DataFrame:
    """Test every candidate pair; BH-correct the ADF p-values across the scan."""
    rows = []
    for a, b in pairs:
        if a not in basis.columns or b not in basis.columns:
            continue
        spread = (basis[a] - basis[b]).dropna()
        if len(spread) < min_obs:
            continue
        # filter on the in-sample window: two points on one bus can be identical
        # for a year and diverge later, which is a constant series to ADF
        ins_std = spread.iloc[: int(len(spread) * 0.7)].std()
        if not np.isfinite(ins_std) or ins_std < min_spread_std:
            continue
        try:
            res = test_pair(basis[a], basis[b])
        except ValueError:
            continue
        res.update({"a": a, "b": b})
        rows.append(res)
    table = pd.DataFrame(rows).set_index(["a", "b"])
    if table.empty:
        return table
    reject, p_bh, _, _ = multipletests(table["adf_p"], alpha=fdr, method="fdr_bh")
    table["p_bh"] = p_bh
    table["discovery"] = reject
    return table.sort_values("adf_p")
