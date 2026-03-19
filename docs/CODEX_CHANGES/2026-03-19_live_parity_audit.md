# 2026-03-19 - Live parity audit and experimental comparison contract

## already implemented

- `WorldState` already holds the live backbone: `official`, `aux`, `env`, `qc`, `health`, `pws_details`, `pws_metrics`, and tape metadata.
- `get_polymarket_dashboard_data()` already composes market state, WS enrichment, sentiment, and the current trading payload.
- `build_trading_signal()` and `NowcastEngine.get_state_snapshot()` already expose the live forecast and signal path.
- `ReplaySession` and the backtest stack already exist as separate replay and evaluation paths.

## not parity-ready

- There is no single persisted contract shared by live, replay, and backtest.
- The recorder/replay path does not preserve the full `nowcast_state` and all signal inputs needed for faithful reconstruction.
- Live and replay still derive their views from different payload shapes and freshness rules.
- Wunderground is still being used as review/settlement context, not as a separately versioned live truth source.

## contract required for future parity

- Required top-level contract fields:
  - `station_id`
  - `target_day`
  - `target_date`
  - `reference_utc`
  - `market_unit`
  - `official`
  - `auxiliary`
  - `forecast`
  - `market`
  - `signal`
  - `provenance`
  - `settlement_review`
- Contract invariants:
  - `official` stays separate from `settlement_review`
  - `METAR/SPECI` remains the live official confirmation source
  - `PWS/Synoptic/MADIS` remains auxiliary
  - market state remains separate from forecast state
  - provenance must carry timestamps, source age, and role/staleness markers
  - no second source of truth may be introduced
- Current parity blockers:
  - recorder does not persist the full state needed for a frozen replay contract
  - replay/live do not consume the same frozen payload
  - backtest uses a different contract path from live signal generation

## residual risk

- The comparison endpoint can still be useful as an inspection layer before full parity exists, but it should not be treated as the contract for replay or execution.
