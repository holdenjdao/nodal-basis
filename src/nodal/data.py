"""ERCOT settlement point price ingestion.

Downloads day-ahead hourly and real-time 15-minute SPPs via gridstatus and keeps a
local parquet store, one file per market per month. All timestamps are stored
tz-aware in US/Central (ERCOT market time). Analysis code should read through
`load_da` / `load_rt`, which return long-format frames with a stable schema:

    interval_start (datetime, tz-aware), location (str), location_type (str),
    price (float)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

MARKETS = {
    "da": "DAY_AHEAD_HOURLY",
    "rt": "REAL_TIME_15_MIN",
}

log = logging.getLogger(__name__)


def _month_path(market: str, year: int, month: int) -> Path:
    return DATA_DIR / market / f"{year:04d}-{month:02d}.parquet"


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Map a gridstatus SPP frame to the canonical long schema."""
    out = pd.DataFrame(
        {
            "interval_start": raw["Interval Start"],
            "location": raw["Location"].astype("string"),
            "location_type": raw["Location Type"].astype("string"),
            "price": raw["SPP"].astype("float64"),
        }
    )
    return out.sort_values(["interval_start", "location"]).reset_index(drop=True)


def fetch_day(market: str, date: str) -> pd.DataFrame:
    """Fetch one day of SPPs ('da' or 'rt') from ERCOT, normalized."""
    import gridstatus

    iso = gridstatus.Ercot()
    raw = iso.get_spp(date=date, market=MARKETS[market], location_type="ALL")
    return _normalize(raw)


def fetch_range(market: str, start: str, end: str) -> None:
    """Incrementally download [start, end) into the monthly parquet store.

    Skips days already present, so it is safe to re-run; a partial month is
    topped up in place.
    """
    days = pd.date_range(start, end, freq="D", inclusive="left")
    by_month: dict[tuple[int, int], list[pd.Timestamp]] = {}
    for day in days:
        by_month.setdefault((day.year, day.month), []).append(day)

    for (year, month), month_days in by_month.items():
        path = _month_path(market, year, month)
        existing = pd.read_parquet(path) if path.exists() else None
        have = (
            set(existing["interval_start"].dt.date.unique())
            if existing is not None
            else set()
        )
        todo = [d for d in month_days if d.date() not in have]
        if not todo:
            continue
        frames = [] if existing is None else [existing]
        for day in todo:
            log.info("fetching %s %s", market, day.date())
            try:
                frames.append(fetch_day(market, day.strftime("%Y-%m-%d")))
            except Exception as exc:  # noqa: BLE001 - one bad day shouldn't kill a month
                log.warning("failed %s %s: %s", market, day.date(), exc)
        merged = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(["interval_start", "location"])
            .sort_values(["interval_start", "location"])
            .reset_index(drop=True)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(path, index=False)
        log.info("wrote %s (%d rows)", path.name, len(merged))


def _load(market: str) -> pd.DataFrame:
    paths = sorted((DATA_DIR / market).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no local {market} data; run scripts/fetch_data.py")
    return pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)


def load_da() -> pd.DataFrame:
    return _load("da")


def load_rt() -> pd.DataFrame:
    return _load("rt")


def da_panel(location_types: list[str] | None = None) -> pd.DataFrame:
    """Wide hourly panel: index interval_start, columns location, values DA price."""
    df = load_da()
    if location_types:
        df = df[df["location_type"].isin(location_types)]
    return df.pivot_table(index="interval_start", columns="location", values="price")


def rt_hourly_panel(location_types: list[str] | None = None) -> pd.DataFrame:
    """RT 15-min prices averaged to the hour, as a wide panel aligned with DA.

    Hours with fewer than four 15-minute intervals are masked: a partial-day
    fragment averaged into an "hourly" price would silently corrupt DART stats.
    """
    df = load_rt()
    if location_types:
        df = df[df["location_type"].isin(location_types)]
    df = df.assign(hour=df["interval_start"].dt.floor("h"))
    grouped = df.groupby(["hour", "location"])["price"].agg(["mean", "count"])
    complete = grouped.loc[grouped["count"] >= 4, "mean"]
    return complete.unstack("location")
