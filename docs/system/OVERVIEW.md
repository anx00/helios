# HELIOS Project + System Reference

This document is the short, practical reference for how HELIOS is built today.

It consolidates the most relevant information spread across:
- architecture docs
- prediction/math docs
- phase design docs
- market integration docs
- operational notes and fixes

Use this first when you need to answer:
- "where does this behavior live?"
- "what is the source of truth for this subsystem?"
- "what loop updates this data and how often?"
- "which doc is canonical vs historical?"

## 1. Project purpose (current scope)

HELIOS is a weather-driven forecasting and trading system for Polymarket temperature markets.

The project combines:
- real-time weather ingestion (METAR, PWS, environmental features)
- nowcast generation (distribution, not only point forecast)
- live market data ingestion (Polymarket CLOB WS + local orderbook mirror)
- replay/backtest infrastructure
- paper-first autotrader runtime
- web dashboards and API endpoints
- Atenea (AI copilot with evidence-based context)

## 2. High-level architecture (what runs together)

The current architecture is split into logical layers:

1. World / ingestion layer
- Collects METAR, PWS, and environmental signals.
- Applies QC and freshness tracking.
- Publishes state/events used by UI and nowcast.

2. Forecast / nowcast layer
- Builds/updates per-station distributions.
- Produces `NowcastDistribution` outputs (mean, sigma, bucket probabilities, confidence, etc.).

3. Market layer (Polymarket)
- Uses Gamma API for event discovery / metadata.
- Uses CLOB WebSocket for live microstructure (orderbook/trades).
- Maintains local L2 mirror by token ID.

4. Storage / replay / backtest layer
- Records tapes and snapshots (live and derived state).
- Replays historical sessions with a virtual clock.
- Builds backtest datasets and execution-aware simulations.

5. Trading / decision layer
- Autotrader consumes nowcast + market orderbook + risk gates.
- Paper broker simulates execution.
- Live execution adapter exists but is still a stub (feature-flagged).

6. Presentation / API layer
- FastAPI server exposes dashboards and internal APIs (`/api/...`).
- Frontends consume nowcast, market, replay, autotrader, and diagnostics endpoints.

## 3. Runtime loops and cadences (operationally relevant)

HELIOS starts multiple background loops from `web_server.py` startup.
These are the loops you should check first when data looks stale.

### Main loops (as implemented)

- `pws_loop()`:
  - refreshes PWS cluster data
  - cadence: about every 120s

- `prediction_loop()`:
  - runs heavier prediction refreshes
  - cadence: about every 5 min

- `snapshot_loop()`:
  - captures near-realtime snapshots for analytics / monitoring
  - cadence: about every 3s
  - note: some slow sources are cached and refreshed with TTL in `monitor.py`

- `websocket_loop()`:
  - maintains Polymarket CLOB WS connection and token subscriptions
  - refreshes subscriptions periodically (about every 5 min)

- `nowcast_loop()`:
  - coordinates nowcast engines and event-driven updates
  - nowcast internals also have periodic refresh (~60s)

- `recorder_loop()`:
  - records event streams for replay/backtest

- `autotrader_loop()`:
  - paper-first trading runtime

- `learning_nightly_loop()`:
  - offline learning cycle

## 4. Data source hierarchy (source of truth by function)

This is one of the most important design decisions in HELIOS.

### Settlement / official truth
- METAR (NOAA / Aviation Weather) is the settlement authority for airport markets.

Important operational rule:
- if a METAR observation has no `T-group` precision field, treat temperature as a range (not a single exact point)
- do not trust third-party decoded Fahrenheit point values blindly

Related docs:
- `docs/weather/METAR_RACING.md`
- `docs/weather/METAR_SETTLEMENT_RANGE_NOTE.md`

### Validation / confirmation layer
- PWS cluster (MADIS/CWOP + WU/weather.com support/fallbacks depending station/path)
- used for consensus, drift detection, QC confidence, and UI context

Related docs:
- `docs/weather/PWS_WUNDERGROUND.md`
- `docs/weather/PWS_MADIS_CWOP_MIGRATION_NOTE.md`

### Environmental features
- SST (NDBC) for coastal effects
- AOD for aerosol/solar attenuation
- used as adjustment signals, not settlement truth

### Base forecast / model layer
- HRRR/GFS and other forecast products (NBM/LAMP/UKV depending station/use case)
- provide base curve and anchors for nowcast

Canonical detail:
- `docs/weather/DATA_SOURCES.md`

## 5. Prediction and nowcast model (what the output means)

HELIOS no longer treats prediction as only a single Tmax number.
The important output is a distribution over settlement-relevant buckets.

### Canonical output (conceptually)

`NowcastDistribution` includes:
- central estimate (`tmax_mean_f`)
- uncertainty (`tmax_sigma_f`)
- `p_bucket` probabilities
- `p_ge_strike` cumulative probabilities
- time-of-peak distribution (`t_peak_bins`)
- confidence and explanation fields

### Key modeling ideas

- Bias correction relative to a base forecast
- Time decay of bias as peak approaches
- Hard constraints (reality floor, post-peak cap)
- Sigma (uncertainty) penalized by staleness / QC degradation
- Distribution generation for market-comparable buckets

Canonical docs:
- `docs/weather/PREDICTION_SYSTEM.md`
- `docs/weather/MATH_REFERENCE.md`
- `docs/weather/NOWCAST_ENGINE_IMPLEMENTATION.md`

## 6. Polymarket integration (how market data is handled)

HELIOS uses a two-layer market architecture:

1. Gamma API (metadata / discovery)
- event discovery by slug or date window
- bracket names, volumes, `outcomePrices`, `clobTokenIds`

2. CLOB WebSocket (hot path)
- orderbook snapshots/deltas
- best bid/ask
- trades / last trade
- local L2 mirror in memory for low-latency reads

This separation was a major architectural correction documented in the phase/refactor docs.

Canonical docs:
- `docs/market/POLYMARKET_INTEGRATION.md`
- `docs/market/POLYMARKET_TOKEN_IDS.md`
- `docs/system/ARCHITECTURE_EVOLUTION.md`

## 7. Replay, backtest, and autotrader (how development and trading loop closes)

### Replay / recorder / compactor
- live events are captured and later replayed for debugging and development
- replay is a first-class subsystem, not only logging

### Backtest
- supports signal-only and execution-aware evaluation
- focuses on label correctness, as-of joins, policy simulation, and calibration

### Autotrader
- paper-first runtime
- combines nowcast, live orderbook, risk gates, strategy selection (LinUCB)
- persists decisions/orders/fills to SQLite
- live execution adapter is currently a stub

Canonical docs:
- `docs/trading/BACKTEST_REPLAY.md`
- `docs/trading/AUTOTRADER.md`
- `docs/trading/STORAGE_REPLAY_IMPLEMENTATION.md`
- `docs/trading/BACKTEST_CALIBRATION_IMPLEMENTATION.md`

## 8. UI/API surfaces (where people and bots read the system)

The FastAPI app is the integration point for dashboards and tooling.

Common categories:
- station/world status
- nowcast outputs
- Polymarket snapshots + orderbook context
- replay sessions
- autotrader status / positions / performance / fills
- Atenea context endpoints

For Polymarket specifically, the important internal endpoints are documented in:
- `docs/market/POLYMARKET_INTEGRATION.md`

## 9. Which docs to use (current implementation)

### Core implementation docs
- `docs/system/*`
- `docs/weather/*`
- `docs/market/*`
- `docs/trading/*`
- `docs/ai/*`
- `docs/operations/*`

### Chronological engineering trace
- `docs/engineering/*`

### Legacy context only
- `docs/legacy/*`

## 10. Recommended workflow when changing the system

### If you change market ingestion / trading context
1. Read `docs/market/POLYMARKET_INTEGRATION.md`
2. Read `docs/market/POLYMARKET_TOKEN_IDS.md`
3. Check notes in `docs/market/rollouts/` and `docs/weather/`
4. Update `docs/engineering/CHANGE_JOURNAL.md`

### If you change nowcast / prediction logic
1. Read `docs/weather/PREDICTION_SYSTEM.md`
2. Read `docs/weather/MATH_REFERENCE.md`
3. Read `docs/weather/DATA_SOURCES.md`
4. Update `docs/engineering/CHANGE_JOURNAL.md`

### If you change replay/backtest/autotrader
1. Read `docs/trading/BACKTEST_REPLAY.md` and/or `docs/trading/AUTOTRADER.md`
2. Check implementation notes in `docs/trading/`
3. Add a note near the affected subsystem and update `docs/engineering/CHANGE_JOURNAL.md`
