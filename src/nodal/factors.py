"""Factor structure of the basis panel.

PCA on the (demeaned) node-minus-hub basis panel. Congestion is
low-dimensional: a handful of binding transmission constraints move whole
regions of nodes together, so the leading principal components correspond to
physical constraint axes and loadings group nodes into congestion zones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FactorModel:
    explained: pd.Series          # variance ratio per factor
    loadings: pd.DataFrame        # location x factor
    scores: pd.DataFrame          # interval x factor


def fit(basis: pd.DataFrame, n_factors: int = 10, standardize: bool = False) -> FactorModel:
    x = basis.dropna(axis=1, thresh=int(0.99 * len(basis)))
    x = x.ffill().dropna()
    values = x.to_numpy(dtype=float)
    values = values - values.mean(axis=0)
    if standardize:
        std = values.std(axis=0)
        std[std == 0] = 1.0
        values = values / std

    u, s, vt = np.linalg.svd(values, full_matrices=False)
    var = s**2
    k = min(n_factors, len(s))
    names = [f"F{i+1}" for i in range(k)]
    return FactorModel(
        explained=pd.Series(var[:k] / var.sum(), index=names),
        loadings=pd.DataFrame(vt[:k].T, index=x.columns, columns=names),
        scores=pd.DataFrame(u[:, :k] * s[:k], index=x.index, columns=names),
    )


def top_loaders(model: FactorModel, factor: str, n: int = 8) -> pd.DataFrame:
    """The nodes at each pole of a factor: the two sides of the constraint."""
    load = model.loadings[factor].sort_values()
    return pd.DataFrame(
        {"negative_pole": load.head(n).index, "neg_loading": load.head(n).round(3).values,
         "positive_pole": load.tail(n).index[::-1], "pos_loading": load.tail(n).round(3).values[::-1]}
    )
