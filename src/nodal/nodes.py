"""Node profiles: one row per settlement point summarizing its personality.

A node is an electrical location. Its price is hub energy plus the shadow
prices of whichever transmission constraints it is exposed to (weighted by
its shift factors) plus losses. Everything we measure about a node is
therefore a statement about its exposure to constraints:

    level        mean basis (structurally cheap / rich vs hub)
    dispersion   basis std, mean |basis|, skew, spike share
    shape        hour-of-day basis profile (solar nodes sag midday, etc.)
    exposure     factor loadings (which constraint axes move it)
    residual     variance unexplained by the factor model (idiosyncratic risk)
    dart         DA-RT premium stats from `dart.py`, if supplied
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from nodal.factors import FactorModel

_ASSET_PATTERNS = [
    (r"_SLR|SOLAR|_SUN", "solar"),
    (r"_BESS|_ESR|_ESS|_BAT|STOR", "storage"),
    (r"_WND|WIND|_ALL$|_BRZ|BREEZE", "wind"),
    (r"_CC\d|_CC$|_CT\d|_GT\d|_UNIT|_G\d|_ST\d|_CCU", "thermal"),
    (r"^HB_", "hub"),
    (r"^LZ_", "load_zone"),
    (r"^DC_", "dc_tie"),
]


def asset_type(location: str) -> str:
    """Heuristic asset class from ERCOT naming conventions; 'other' when unclear."""
    for pattern, label in _ASSET_PATTERNS:
        if re.search(pattern, location):
            return label
    return "other"


def _factor_fit_stats(basis: pd.DataFrame, model: FactorModel) -> pd.DataFrame:
    """Per-node R^2 of the factor reconstruction (on the model's transformed
    scale) and residual std in $/MWh (rescaled back by the node's sigma)."""
    z = model.transform(basis.loc[model.scores.index]).ffill().dropna()
    resid = z.to_numpy() - model.reconstruct().loc[z.index].to_numpy()
    total_var = np.nanvar(z.to_numpy(), axis=0)
    resid_var = np.nanvar(resid, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(total_var > 0, 1.0 - resid_var / total_var, np.nan)
    resid_std_dollars = np.sqrt(resid_var) * model.sigma.loc[z.columns].to_numpy()
    return pd.DataFrame({"factor_r2": r2, "resid_std": resid_std_dollars}, index=z.columns)


def profile_table(
    basis: pd.DataFrame,
    model: FactorModel,
    dart_table: pd.DataFrame | None = None,
    spike_threshold: float = 25.0,
) -> pd.DataFrame:
    """One row per location: level, dispersion, shape, exposure, residual, dart."""
    hours = basis.index.hour
    by_hour = basis.groupby(hours).mean()

    prof = pd.DataFrame(
        {
            "asset": [asset_type(c) for c in basis.columns],
            "mean_basis": basis.mean(),
            "std_basis": basis.std(),
            "abs_basis": basis.abs().mean(),
            "skew": basis.skew(),
            "spike_share": (basis.abs() > spike_threshold).mean(),
            "neg_share": (basis < -1.0).mean(),
            "peak_hour": by_hour.idxmax(),
            "trough_hour": by_hour.idxmin(),
            "diurnal_range": by_hour.max() - by_hour.min(),
        },
        index=basis.columns,
    )
    prof = prof.join(model.loadings, how="left")
    prof = prof.join(_factor_fit_stats(basis, model), how="left")
    if dart_table is not None:
        prof = prof.join(
            dart_table[["mean_dart", "t_hac", "p_bh", "discovery"]], how="left"
        )
    return prof


def similar_nodes(profiles: pd.DataFrame, location: str, k: int = 10) -> pd.Series:
    """Nearest nodes in factor-loading space (Euclidean); the physical neighbourhood."""
    cols = [c for c in profiles.columns if re.fullmatch(r"F\d+", c)]
    load = profiles[cols].dropna()
    target = load.loc[location]
    dist = np.sqrt(((load - target) ** 2).sum(axis=1))
    return dist.drop(location).nsmallest(k)
