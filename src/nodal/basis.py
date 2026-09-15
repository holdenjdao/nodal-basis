"""Basis panel construction: node price minus hub reference."""

from __future__ import annotations

import pandas as pd

HUB_AVG = "HB_HUBAVG"


def basis_panel(panel: pd.DataFrame, hub: str = HUB_AVG) -> pd.DataFrame:
    """Node minus hub for every column, hub reference column dropped.

    `panel` is a wide price frame (index: interval, columns: location).
    """
    if hub not in panel.columns:
        raise KeyError(f"hub column {hub!r} not in panel")
    out = panel.sub(panel[hub], axis=0)
    return out.drop(columns=[hub])


def filter_complete(panel: pd.DataFrame, min_coverage: float = 0.98) -> pd.DataFrame:
    """Drop locations with sparse history (new/retired nodes distort stats)."""
    coverage = panel.notna().mean()
    return panel.loc[:, coverage >= min_coverage]
