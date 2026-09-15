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
  data.py        ERCOT SPP download + local parquet store
  basis.py       basis panel construction
  dart.py        DART bias analysis
  pairs.py       cointegration & mean-reversion pairs
  factors.py     PCA factor structure
  backtest.py    cost-aware backtests
scripts/         CLI entry points (data fetch, report build)
data/            local parquet store (not committed)
```

## Data

ERCOT day-ahead hourly and real-time 15-minute settlement point prices for all
settlement points (hubs, load zones, resource nodes), pulled via
[gridstatus](https://github.com/gridstatus/gridstatus) into a local parquet store.

## Setup

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```
