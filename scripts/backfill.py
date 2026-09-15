"""Backfill nodal price history from the ERCOT Public API.

Usage:
    python scripts/backfill.py da 2025-01-01 2026-09-01
    python scripts/backfill.py rt 2025-01-01 2026-09-01
Needs credentials in .env (see .env.example). Resumable: stored weeks are skipped.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nodal import data


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    market, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
    data.fetch_range_api(market, start, end)


if __name__ == "__main__":
    main()
