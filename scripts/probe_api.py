"""One-off probe: confirm what ERCOT data gridstatus gives us and in what shape."""

import sys

import pandas as pd

import gridstatus

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

iso = gridstatus.Ercot()

date = sys.argv[1] if len(sys.argv) > 1 else "2026-09-10"

print("=== Day-ahead hourly SPP ===")
da = iso.get_spp(date=date, market="DAY_AHEAD_HOURLY", location_type="ALL")
print(da.head(3))
print("rows:", len(da))
print("columns:", list(da.columns))
print("location types:", da["Location Type"].value_counts().to_dict())
print("unique locations:", da["Location"].nunique())

print("\n=== Real-time 15-min SPP ===")
rt = iso.get_spp(date=date, market="REAL_TIME_15_MIN", location_type="ALL")
print(rt.head(3))
print("rows:", len(rt))
print("unique locations:", rt["Location"].nunique())
