"""Top up the local store from the public MIS feed (no credentials needed).

Pulls whatever recent days are missing within MIS's retention window:
~30 days of day-ahead, ~7 days of real-time. Idempotent, so it is safe to run
daily from a scheduler; run by hand any time.

    python scripts/update.py
"""

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nodal import data


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    today = date.today()
    # DA for tomorrow is published ~13:30 CT; asking for it early just logs a miss
    data.fetch_range("da", str(today - timedelta(days=32)), str(today + timedelta(days=2)))
    data.fetch_range("rt", str(today - timedelta(days=9)), str(today))


if __name__ == "__main__":
    main()
