"""ERCOT settlement point price ingestion.

Two sources feed one local parquet store (one file per market per month):

* MIS (public, no auth): rolling window only — ~30 days of day-ahead files,
  ~7 days of real-time files. Used by the daily accumulation job.
* ERCOT Public API (free account): years of history. Used for backfill.
  Credentials come from the environment or a git-ignored `.env` file:
  ERCOT_API_USERNAME, ERCOT_API_PASSWORD, ERCOT_PUBLIC_API_SUBSCRIPTION_KEY.

All timestamps are tz-aware US/Central (ERCOT market time). Analysis code
reads through `load_da` / `load_rt` / the panel builders, which expose a
stable long schema regardless of source:

    interval_start (datetime, tz-aware), location (str), location_type (str),
    price (float)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

MARKETS = {
    "da": "DAY_AHEAD_HOURLY",
    "rt": "REAL_TIME_15_MIN",
}

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- store


def _month_path(market: str, year: int, month: int) -> Path:
    return DATA_DIR / market / f"{year:04d}-{month:02d}.parquet"


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Map a gridstatus SPP frame (MIS or API — same columns) to the canonical schema."""
    out = pd.DataFrame(
        {
            "interval_start": raw["Interval Start"],
            "location": raw["Location"].astype("string"),
            "location_type": raw["Location Type"].astype("string"),
            "price": raw["SPP"].astype("float64"),
        }
    )
    return out.sort_values(["interval_start", "location"]).reset_index(drop=True)


def _upsert(market: str, frame: pd.DataFrame) -> None:
    """Merge normalized rows into the monthly store, de-duplicated on (interval, location)."""
    if frame.empty:
        return
    for (year, month), chunk in frame.groupby(
        [frame["interval_start"].dt.year, frame["interval_start"].dt.month]
    ):
        path = _month_path(market, int(year), int(month))
        frames = [pd.read_parquet(path), chunk] if path.exists() else [chunk]
        merged = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(["interval_start", "location"], keep="last")
            .sort_values(["interval_start", "location"])
            .reset_index(drop=True)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(path, index=False)
        log.info("wrote %s/%s (%d rows)", market, path.name, len(merged))


def stored_days(market: str) -> set:
    """Dates already present in the store for a market."""
    days: set = set()
    for path in (DATA_DIR / market).glob("*.parquet"):
        days |= set(pd.read_parquet(path, columns=["interval_start"])["interval_start"].dt.date.unique())
    return days


# ------------------------------------------------------------------- MIS (public)


def fetch_day(market: str, date: str) -> pd.DataFrame:
    """Fetch one day of SPPs ('da' or 'rt') from public MIS, normalized."""
    import gridstatus

    iso = gridstatus.Ercot()
    raw = iso.get_spp(date=date, market=MARKETS[market], location_type="ALL")
    return _normalize(raw)


def fetch_range(market: str, start: str, end: str) -> None:
    """Pull [start, end) day by day from MIS, skipping days already stored.

    Safe to re-run. Days outside MIS's retention window fail individually and
    are logged, not fatal.
    """
    have = stored_days(market)
    for day in pd.date_range(start, end, freq="D", inclusive="left"):
        if day.date() in have:
            continue
        log.info("fetching %s %s", market, day.date())
        try:
            _upsert(market, fetch_day(market, day.strftime("%Y-%m-%d")))
        except Exception as exc:  # noqa: BLE001 - one purged day shouldn't kill the run
            log.warning("failed %s %s: %s", market, day.date(), exc)


# --------------------------------------------------------------- ERCOT Public API


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal KEY=VALUE loader so credentials never live in the repo or shell history."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _api():
    _load_dotenv()
    from gridstatus.ercot_api.ercot_api import ErcotAPI

    return ErcotAPI()


def fetch_range_api(market: str, start: str, end: str, chunk_days: int = 7) -> None:
    """Backfill [start, end) from the ERCOT Public API in weekly chunks.

    Chunks already fully present in the store are skipped, so an interrupted
    backfill resumes where it stopped.
    """
    api = _api()
    have = stored_days(market)
    getter = {
        "da": api.get_spp_day_ahead_hourly,
        "rt": api.get_spp_real_time_15_min,
    }[market]
    edges = list(pd.date_range(start, end, freq=f"{chunk_days}D"))
    if edges[-1] < pd.Timestamp(end):
        edges.append(pd.Timestamp(end))
    for lo, hi in zip(edges[:-1], edges[1:]):
        days = pd.date_range(lo, hi, freq="D", inclusive="left")
        if all(d.date() in have for d in days):
            continue
        log.info("backfilling %s %s .. %s", market, lo.date(), hi.date())
        try:
            raw = getter(date=lo.strftime("%Y-%m-%d"), end=hi.strftime("%Y-%m-%d"))
        except Exception as exc:  # noqa: BLE001
            log.warning("failed %s %s..%s: %s", market, lo.date(), hi.date(), exc)
            continue
        _upsert(market, _normalize(raw))


# ------------------------------------------------------------------------ readers


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


def _rt_month_to_hourly(path: Path, location_types: list[str] | None) -> pd.Series:
    df = pd.read_parquet(path)
    if location_types:
        df = df[df["location_type"].isin(location_types)]
    # floor in UTC: flooring in local time is ambiguous on the DST fall-back hour
    hour = df["interval_start"].dt.tz_convert("UTC").dt.floor("h").dt.tz_convert("US/Central")
    df = df.assign(hour=hour)
    grouped = df.groupby(["hour", "location"])["price"].agg(["mean", "count"])
    return grouped.loc[grouped["count"] >= 4, "mean"]


def rt_hourly_panel(location_types: list[str] | None = None) -> pd.DataFrame:
    """RT 15-min prices averaged to the hour, as a wide panel aligned with DA.

    Aggregated one month file at a time (a year of 15-min nodal data is ~40M
    rows; hourly is a quarter of that). Hours with fewer than four 15-minute
    intervals are masked: a partial-day fragment averaged into an "hourly"
    price would silently corrupt DART stats. An hour straddling a month file
    boundary cannot occur because files split on interval_start's month.
    """
    paths = sorted((DATA_DIR / "rt").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError("no local rt data; run scripts/fetch_data.py")
    hourly = pd.concat(_rt_month_to_hourly(p, location_types) for p in paths)
    hourly = hourly[~hourly.index.duplicated(keep="last")]
    return hourly.unstack("location")
