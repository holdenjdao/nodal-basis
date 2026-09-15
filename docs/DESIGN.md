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

## Findings log

Dated, so later readers know which regime each claim was made in.

- **2026-09-15, 20 months of DA (Jan 2025 – Sep 2026), 959 full-coverage nodes.**
  - *Raw-variance PCA is hijacked by tails.* One node (`PALACIOS_RN`, basis
    std $274) captured two factors on its own, and half-year sub-fits
    looked like a different grid each period. Winsorized + standardized PCA
    instead recovers the West-export constraint as F1 and South-coastal
    export as F2, with diffuse loadings — the physically sensible answer.
    Honest five-factor explained variance: ~76%, not 85%.
  - *The dominant constraint axis is stable; its intensity is not.* On the
    standardized scale each half-year's F1 is the same West-export axis
    (|corr| with full-sample F1 = 0.97, 0.97, 0.99). Its share of variance
    rises 33% → 37% → 45% into 2026: West export congestion became more
    dominant. Rolling refits are still required for the lower factors and
    for new nodes, but the map's main road does not move.
  - *Node statistics are tail-dominated.* Spike shares of 18–26% and basis
    std of $60–75 at the extreme nodes (Russek, Junction, Appaloosa) mean
    sample means are fragile; report robust (winsorized/median) statistics
    alongside means everywhere. Extreme case: `PALACIOS_RN` has a median
    basis of $1 and one hour at +$21,068 (price $22,618/MWh); that hour
    alone contributes ~$170 of its $274 std. Its 143 hours above +$100
    cluster in Jul 2025 and Nov 2025–Jan 2026: a coastal load pocket whose
    local constraint binds in peak events.
  - *Stationarity is not a finding.* 98–99% of neighbour-pair spreads pass
    ADF on both the 30-day and 20-month windows. Ranking by std / half-life
    then put `PALACIOS_RN` pairs on top with spread std of $300+: one spike
    hour that is gone the next hour scores as "fast reversion". Pair tests
    now run on winsorized spreads, rank by MAD-based robust sigma, and
    require dislocations > $5 in at least 10% of hours both in and out of
    sample — frequency of opportunity, not size of the largest one.
  - *99 spread pairs survive the full bar* (of 1,883 stationary). Robust
    sigma $6–12/MWh, half-lives 4–9 h and stable in/out of sample, >$5
    dislocations in 40–75% of hours. They sit in two physical pockets: the
    Junction/Olney/Potosi Hill Country pocket and a Del Rio-area cluster
    (Amistad, Indian Energy, Hamilton BESS, Appaloosa, Fermi, Russek).
    Reversion yields ~$1.2–2.0/MWh per hour: modest, real, and only
    tradeable in the day-over-day or DA-vs-RT framing (see caveat above).
- **2026-09-15, 17 months of DA∩RT (Jan 2025 – Jun 2026), 960 nodes.**
  - *Absolute DART is one fact, not 865.* 865/960 nodes are BH discoveries,
    all DA-rich; the best DA-cheap t-stat is −1.6. Day-ahead clears above
    real-time essentially everywhere: the electricity forward premium,
    a hub-level phenomenon. Absolute node darts are not nodal findings.
  - *Hub-relative DART isolates the locational part.* Node dart minus hub
    dart (= DA basis − RT basis) leaves 200 discoveries, 140 DA-rich and
    60 DA-cheap. DA-rich: the Hill Country / coastal scarcity pockets
    (Palacios, RHESS2, NF_BRP, Medina) — DA underprices local scarcity.
    DA-cheap: South Texas coastal wind (Foxtrot, Sparta, Karankawa,
    Algodón, Nueces) — DA over-discounts congestion relative to real time.
    The tradeable form is a node leg against a hub leg.
  - *Persistence is seasonal.* Hub-relative split-half rank correlation
    0.56; month-to-month mean 0.22 (lag 1), 0.14 (lag 2), 0.08 (lag 3).
    Within summer it is 0.55–0.68; it goes negative around Dec–Jan and
    May. The nodal mispricing cross-section persists inside a season and
    turns over at season boundaries; a signal must be regime-aware.
  - *Raw vs winsorized disagree at the wild nodes.* RHESS2 hub-relative:
    +5.4 winsorized, −7.4 raw. A handful of RT spike hours would have
    destroyed an INC there; the backtest settles on raw darts for exactly
    that reason.
  - *Backtest, absolute DART (pre-registered: 30-day lookback, $2 entry,
    $1 fee).* Net $0.60/MWh over 5.06M node-hours; Sharpe 0.34; hit rate
    49% of days; max drawdown −$5.2M against +$3.0M total. January 2026
    alone made +$5.7M — every other month nets to a loss — and 27% of
    gross P&L came from the top 1% of hours. Sensitivity: lookback 60/90,
    entry $4, or fee $2 all turn it negative; fee $0 gives +$8M. Verdict:
    the forward premium is a risk premium collected by selling tail
    insurance, fragile inside the cost band. Not an inefficiency.
- **2026-09-14, one week of DA∩RT.** 113 nodes passed BH on the DART scan;
  split-sample persistence was −0.05. Congestion episodes, not structure.

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
