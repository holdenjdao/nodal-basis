# Design

## What a node is

ERCOT settles prices at ~1,100 **settlement points**. Physically, a settlement
point is a bus on the transmission network — a substation busbar where a
generator injects (a *resource node*) or, for retail settlement, a
load-weighted average of many buses (a *load zone*). *Trading hubs* are
averages over hundreds of buses, designed to be liquid, constraint-diluted
references. `HB_HUBAVG` is our reference.

Every nodal price decomposes as

    LMP(node, t) = energy(t) + congestion(node, t) + losses(node, t)

`energy` is system-wide and cancels in the basis. `congestion` is the sum over
binding transmission constraints of *shadow price × shift factor*: how much
of each constraint's marginal cost leaks into this bus, determined by network
topology. So **a node's identity is its vector of shift factors** — its
exposure to the set of constraints that bind. Two nodes are "similar" when
they load on the same constraints with the same signs. We cannot observe
shift factors directly from prices, but the PCA loadings on the basis panel
are an empirical proxy for them (`factors.py`), and the ERCOT SCED binding
constraint reports can later provide the physical ground truth.

A node also has an *owner*: the asset behind it. Wind and solar nodes sit at
the far end of long radial lines in West/South Texas and price *below* hub
when their own output saturates the wires; battery nodes see the sharpest
local price swings; load-side nodes in cities price *above* hub when imports
are constrained. The name suffix usually tells you (`_SLR`, `_BESS`/`_ESR`,
`_ALL` for aggregated wind, `_CC`/`_UNIT` for gas).

## What "inefficiency" means here

Three testable forms, each a different mechanism:

| form | mechanism | scan | module |
|---|---|---|---|
| DART premium | DA market systematically mis-forecasts RT at a node (risk premium, forecast bias, thin participation) | per-node mean(DA−RT), HAC t, BH across nodes, split-sample persistence | `dart.py` |
| spread reversion | two similarly-exposed nodes' spread departs from equilibrium and returns | cointegration + OU half-life on candidate pairs, walk-forward z-score | `pairs.py` |
| factor dislocation | a node's price departs from what its factor exposure implies | residual z-scores vs the factor model, regime-break detection | `factors.py` + `nodes.py` |

**Tradeability caveat for spreads.** All 24 hours of the day-ahead market
clear simultaneously in one auction, so hour-to-hour reversion *within* a
day's DA curve is a pattern, not a sequential trade. The tradeable versions
of a spread signal are (a) day-over-day: yesterday's spread level vs today's
clearing, and (b) DA-vs-RT: a spread that clears wide in DA and converges at
RT settlement, executed as an INC at the rich node and a DEC at the cheap
node. `backtest.py` frames spread strategies in those terms; the hourly OU
half-lives in `pairs.py` characterise dynamics, they are not a trade.

**Economic bar.** Stationarity is nearly free between neighbouring nodes
(98% of tested pairs passed ADF on the first window). A pair matters only if
the spread is large enough to clear costs *and* reverts fast: we rank by
reversion yield = spread std / half-life, with a minimum spread std.

A scan produces *candidates*. A candidate becomes a *finding* only after:
1. multiple-testing control across everything scanned (BH FDR),
2. persistence out of sample (the cross-section of premia must predict itself),
3. net-of-cost economics in a walk-forward backtest (`backtest.py`).

The first week of data already demonstrated why: 113 nodes passed BH on a
one-week DART scan, and the split-sample rank correlation was −0.05. Those
were congestion episodes, not structure.

## Pipeline

    data.py      MIS (rolling, daily) + ERCOT API (backfill)  ->  parquet store
    basis.py     DA panel, RT-hourly panel, basis = node − hub
    factors.py   PCA -> constraint axes, loadings, residuals
    nodes.py     per-node profile: level, dispersion, shape, exposure, dart
    dart.py      DA/RT premium scan with HAC + BH + persistence
    pairs.py     cointegrated spread scan on exposure-near pairs
    backtest.py  walk-forward, cost-aware evaluation of any signal
    scripts/     CLI entry points; results/ holds committed CSV outputs

## How each node is analysed

For a node `n`, over an analysis window:

1. **Profile** — basis level, std, skew, spike share (|basis| > $25),
   hour-of-day profile, factor loadings, residual variance. (`nodes.py`)
2. **DART** — mean(DA−RT), HAC t, BH-adjusted p, per-hour breakdown; then
   persistence across windows. (`dart.py`)
3. **Neighbourhood** — nearest nodes in loading space; spreads to each are
   tested for cointegration. (`pairs.py`)
4. **Dislocation** — residual from the factor model, z-scored on a rolling
   window; flagged when it departs and tracked to see whether it reverts or
   the structure has changed (new line, new asset). (`factors.py`)
5. **Economics** — any signal surviving 1–4 is backtested net of costs.

## Limitations (known, accepted)

- **Prices, not physics.** We infer constraints from prices. Shift factors,
  outages and topology changes are unobserved; ERCOT's binding-constraint
  and outage reports are the natural enrichment.
- **Non-stationarity.** The grid changes: new lines relieve constraints,
  new batteries and solar farms create them. Old history can be actively
  misleading; every result is window-dated, and regime breaks are a first-
  class object, not a nuisance.
- **Node churn.** Nodes appear (new interconnections) and vanish
  (retirements, renames). Coverage filters and careful alignment matter.
- **Hourly aggregation.** RT is 15-min settled and 5-min dispatched; hourly
  averaging hides intra-hour structure that is real to real traders.
- **Costs are estimates.** Virtual bid fees, uplift charges and — above all
  — the thinness of nodal virtual markets are approximated. Results are
  reported per MW with capacity caps, never scaled to fantasy notional.
- **Multiple testing forever.** ~1,100 nodes × 24 hours × many windows ×
  many parameterisations is a false-discovery machine. Discipline is the
  product; a null result reported honestly is a valid deliverable.
- **We cannot trade it.** Trading ERCOT virtuals requires QSE registration
  and collateral. This is research; the deliverable is knowledge, code and
  a defensible write-up.

## Breadth: where it can grow

- **Binding constraints** (SCED shadow prices, public): turn the PCA proxy
  into named physical constraints; know *why* a node moves.
- **FTR/CRR auction prices** (public, monthly): compare market-implied
  congestion expectations to realised basis — is the congestion-hedge
  market itself efficient?
- **Weather and load joins**: condition DART premia on temperature, wind and
  solar forecast error; the premia are probably regime-dependent.
- **Real-time price adders and ancillary services**: scarcity mechanics that
  drive the fat tails.
- **Other ISOs**: PJM/MISO publish LMPs with the congestion component
  broken out explicitly — the same code with better-labelled data.
