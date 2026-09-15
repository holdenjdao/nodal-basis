"""Incrementally download ERCOT SPP history into the local parquet store.

Usage:
    python scripts/fetch_data.py da 2025-09-01 2026-09-01
    python scripts/fetch_data.py rt 2025-09-01 2026-09-01
Re-running is safe; days already stored are skipped.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nodal import data


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    market, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
    data.fetch_range(market, start, end)


if __name__ == "__main__":
    main()
