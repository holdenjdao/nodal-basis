# nodal-basis

Quantitative research on locational price inefficiencies in the ERCOT nodal market.

Every ERCOT settlement point trades at hub price ± congestion ± losses. The **basis**
(node − hub) carries the structure: persistent day-ahead vs. real-time premia,
mean-reverting locational spreads, and a low-dimensional factor structure that maps to
physical transmission constraints. This repo measures all three and tests — with
multiple-testing control and cost-aware backtests — which apparent inefficiencies
actually survive.

## Research questions

1. **DART bias.** For each node, is the day-ahead price systematically above or below
   realized real-time? Estimated per node and per hour-of-day; persistence tested
   out-of-sample with Benjamini–Hochberg control across the node universe.
2. **Basis pairs.** Which node pairs have stationary spreads (Engle–Granger /
   Johansen, OU half-life fits), and does z-score reversion survive realistic
   transaction costs?
3. **Factor structure.** PCA on the basis panel: how many factors explain congestion
   co-movement, which nodes load together, and when does the structure break (regime
   changes = new constraints = dislocations)?

## Methodology commitments

- Strict walk-forward splits; parameters chosen in-sample only.
- Multiple-testing correction on every cross-sectional scan.
- All strategy results reported net of estimated costs; the null result is a result.

## Layout

```
src/nodal/       research library
  data.py        ERCOT SPP ingestion (MIS rolling window + Public API backfill), parquet store
  basis.py       basis panel construction
  factors.py     PCA factor structure of the basis panel
  nodes.py       per-node profiles
  dart.py        DART bias scan (HAC, BH, persistence)
  pairs.py       cointegration & mean-reversion pairs
  backtest.py    walk-forward, cost-aware backtests
scripts/         CLI entry points
  fetch_data.py  pull recent days from public MIS (no auth)
  backfill.py    pull history from the ERCOT Public API (needs .env)
  run_dart.py    DART scan -> results/dart.csv
docs/DESIGN.md   what a node is, what "inefficiency" means, limitations, roadmap
data/            local parquet store (not committed)
results/         committed analysis outputs
```

See [docs/DESIGN.md](docs/DESIGN.md) for the conceptual model and methodology.

## Data

ERCOT day-ahead hourly and real-time 15-minute settlement point prices for all
~1,100 settlement points (hubs, load zones, resource nodes), via
[gridstatus](https://github.com/gridstatus/gridstatus).

- **Public MIS** (no account) keeps only a rolling window: ~30 days of day-ahead
  files, ~7 days of real-time files. `scripts/fetch_data.py` pulls it.
- **ERCOT Public API** (free account at apiexplorer.ercot.com) serves years of
  history. Put credentials in `.env` (copy `.env.example`) and run
  `scripts/backfill.py`.

## Setup

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env      # then fill in ERCOT API credentials
```
