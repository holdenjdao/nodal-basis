"""Cost-aware, walk-forward backtests for the inefficiency scans.

Every strategy is evaluated the same way:

    * walk-forward: signals for day D use only information that was
      available when D's day-ahead bids were due
    * costs: a per-MWh charge on every MWh traded (bid fees plus an
      allowance for ERCOT uplift on virtuals); spread trades pay both legs
    * capacity honesty: results are per MW per node, never scaled to a
      notional; the number of node-days traded is reported alongside
    * reporting: net P&L, annualized Sharpe from daily P&L, max drawdown,
      hit rate, and the share of gross P&L earned in the top 1% of hours
      (tail dependence is the real risk story in power)

Pre-registered primary parameters for the DART strategy, fixed before any
result was seen: lookback 30 days, entry threshold $2/MWh, cost $1/MWh.
Other settings are reported as sensitivity, not selection.

Settlement convention (matches dart.py): dart = DA - RT. An INC at a node
sells in DA and buys back in RT, earning +dart per MWh; a DEC earns -dart.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nodal.dart import winsorize

PRIMARY = {"lookback_days": 30, "entry": 2.0, "cost": 1.0}


@dataclass
class BacktestResult:
    pnl_hourly: pd.Series            # net $/MW summed across traded nodes, per hour
    pnl_daily: pd.Series             # same, per day
    positions: pd.DataFrame          # day x node in {-1, 0, +1}
    stats: dict = field(default_factory=dict)


def _daily_index(dart: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(dart.index.normalize().unique())


def dart_signal(
    dart: pd.DataFrame,
    lookback_days: int = 30,
    entry: float = 2.0,
    min_days: int = 20,
    winsor: float = 0.01,
    settle_lag_days: int = 2,
) -> pd.DataFrame:
    """Walk-forward node positions: +1 INC, -1 DEC, 0 flat, one row per day.

    Positions for day D come from the trailing `lookback_days` of daily mean
    dart ending at D - settle_lag_days. Bids for D are due the morning of
    D-1, when RT for D-1 is not yet complete, so D-2 is the last fully known
    day. The trailing series is winsorized per node first so one $20,000
    hour cannot flip a month of signal.
    """
    clipped = winsorize(dart, winsor) if winsor else dart
    daily = clipped.groupby(clipped.index.normalize()).mean()
    trailing = daily.rolling(lookback_days, min_periods=min_days).mean().shift(settle_lag_days)
    pos = pd.DataFrame(0, index=daily.index, columns=daily.columns, dtype="int8")
    pos[trailing > entry] = 1
    pos[trailing < -entry] = -1
    return pos


def settle(
    positions: pd.DataFrame,
    dart: pd.DataFrame,
    cost: float = 1.0,
) -> BacktestResult:
    """Apply daily positions to every hour of that day and settle at DA - RT.

    Uses the raw (un-winsorized) dart for settlement: the strategy really
    would have eaten that $20,000 hour if it was on the wrong side of it.
    """
    day_of = dart.index.normalize()
    pos_hourly = positions.reindex(day_of).set_axis(dart.index)
    cols = pos_hourly.columns.intersection(dart.columns)
    pos_hourly = pos_hourly[cols].fillna(0)
    d = dart[cols]
    gross = (pos_hourly * d).where(d.notna(), 0.0)
    fees = pos_hourly.abs().where(d.notna(), 0) * cost
    net = gross - fees
    pnl_hourly = net.sum(axis=1)
    pnl_daily = pnl_hourly.groupby(day_of).sum()
    traded = (pos_hourly.abs() > 0) & d.notna()

    daily_active = pnl_daily[(traded.groupby(day_of).sum().sum(axis=1)) > 0]
    sharpe = (
        float(daily_active.mean() / daily_active.std() * np.sqrt(365))
        if len(daily_active) > 1 and daily_active.std() > 0
        else np.nan
    )
    cum = pnl_daily.cumsum()
    drawdown = float((cum - cum.cummax()).min())
    gross_pos = gross[gross > 0].stack()
    top1 = gross_pos.nlargest(max(1, int(0.01 * len(gross_pos)))).sum() / gross_pos.sum() if len(gross_pos) else np.nan

    stats = {
        "net_pnl": float(pnl_hourly.sum()),
        "gross_pnl": float(gross.sum().sum()),
        "fees": float(fees.sum().sum()),
        "node_hours_traded": int(traded.sum().sum()),
        "net_per_node_hour": float(pnl_hourly.sum() / max(traded.sum().sum(), 1)),
        "avg_nodes_held": float(traded.sum(axis=1).mean()),
        "days_active": int(len(daily_active)),
        "hit_rate_days": float((daily_active > 0).mean()) if len(daily_active) else np.nan,
        "sharpe_daily_ann": sharpe,
        "max_drawdown": drawdown,
        "top1pct_hours_share_of_gross": float(top1),
        "inc_share": float((pos_hourly[traded] > 0).sum().sum() / max(traded.sum().sum(), 1)),
    }
    return BacktestResult(pnl_hourly=pnl_hourly, pnl_daily=pnl_daily, positions=positions, stats=stats)


def run_dart_backtest(dart: pd.DataFrame, lookback_days: int, entry: float, cost: float) -> BacktestResult:
    pos = dart_signal(dart, lookback_days=lookback_days, entry=entry)
    return settle(pos, dart, cost=cost)


def sensitivity(dart: pd.DataFrame, grid: list[dict] | None = None) -> pd.DataFrame:
    """Stats for the primary parameters plus a small pre-set sensitivity grid."""
    grid = grid or [
        PRIMARY,
        {"lookback_days": 60, "entry": 2.0, "cost": 1.0},
        {"lookback_days": 90, "entry": 2.0, "cost": 1.0},
        {"lookback_days": 30, "entry": 1.0, "cost": 1.0},
        {"lookback_days": 30, "entry": 4.0, "cost": 1.0},
        {"lookback_days": 30, "entry": 2.0, "cost": 2.0},
        {"lookback_days": 30, "entry": 2.0, "cost": 0.0},
    ]
    rows = []
    for params in grid:
        res = run_dart_backtest(dart, **params)
        rows.append({**params, **res.stats, "primary": params == PRIMARY})
    return pd.DataFrame(rows)
