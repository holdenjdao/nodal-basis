"""Factor structure of the basis panel.

PCA on the node-minus-hub basis panel. Congestion is low-dimensional: a
handful of binding transmission constraints move whole regions of nodes
together, so the leading principal components correspond to physical
constraint axes and loadings group nodes into congestion zones.

Default preprocessing is winsorize (1st/99th percentile per node) then
standardize. Power prices have violent tails: on raw variance a single node
with a few $5,000 hours captures two "factors" by itself, and the result
describes magnitude, not co-movement. On the correlation scale the leading
factors recover ERCOT's known congestion geography (West export, South
coastal export) with loadings spread across hundreds of nodes.
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
    lo: pd.Series                 # winsor floor per location (or -inf)
    hi: pd.Series                 # winsor cap per location (or +inf)
    mu: pd.Series                 # per-location mean removed
    sigma: pd.Series              # per-location scale (1.0 if not standardized)

    def transform(self, basis: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted preprocessing to a panel, restricted to model columns."""
        x = basis.loc[:, self.loadings.index]
        x = x.clip(lower=self.lo, upper=self.hi, axis=1)
        return (x - self.mu) / self.sigma

    def reconstruct(self) -> pd.DataFrame:
        """Fitted (transformed-scale) panel from scores and loadings."""
        return pd.DataFrame(
            self.scores.to_numpy() @ self.loadings.to_numpy().T,
            index=self.scores.index,
            columns=self.loadings.index,
        )


def fit(
    basis: pd.DataFrame,
    n_factors: int = 10,
    standardize: bool = True,
    winsor: float | None = 0.01,
) -> FactorModel:
    x = basis.dropna(axis=1, thresh=int(0.99 * len(basis)))
    x = x.ffill().dropna()

    if winsor:
        lo, hi = x.quantile(winsor), x.quantile(1 - winsor)
        x = x.clip(lower=lo, upper=hi, axis=1)
    else:
        lo = pd.Series(-np.inf, index=x.columns)
        hi = pd.Series(np.inf, index=x.columns)

    mu = x.mean()
    sigma = x.std().replace(0.0, 1.0) if standardize else pd.Series(1.0, index=x.columns)
    z = (x - mu) / sigma

    u, s, vt = np.linalg.svd(z.to_numpy(dtype=float), full_matrices=False)
    var = s**2
    k = min(n_factors, len(s))
    names = [f"F{i+1}" for i in range(k)]
    return FactorModel(
        explained=pd.Series(var[:k] / var.sum(), index=names),
        loadings=pd.DataFrame(vt[:k].T, index=x.columns, columns=names),
        scores=pd.DataFrame(u[:, :k] * s[:k], index=x.index, columns=names),
        lo=lo, hi=hi, mu=mu, sigma=sigma,
    )


def top_loaders(model: FactorModel, factor: str, n: int = 8) -> pd.DataFrame:
    """The nodes at each pole of a factor: the two sides of the constraint."""
    load = model.loadings[factor].sort_values()
    return pd.DataFrame(
        {"negative_pole": load.head(n).index, "neg_loading": load.head(n).round(3).values,
         "positive_pole": load.tail(n).index[::-1], "pos_loading": load.tail(n).round(3).values[::-1]}
    )
