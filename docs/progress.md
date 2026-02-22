# HELIOS Progress Log

This document tracks the evolution of the HELIOS Weather Lab project.

## 2026-02-22

### New markets added: Chicago, Miami, Dallas
**Goal**: add full market support (station config, slug resolution, PWS/WU, and forecast fetchers) for Chicago, Miami, and Dallas.

**Changes**:
- Added stations: `KORD`, `KMIA`, `KDAL` in `config.py` (coords/timezones/characteristics/enabled).
- Added Polymarket slug/unit mappings for new stations in `config.py` and `market/polymarket_checker.py`.
- Added Wunderground station URL mapping for the three stations in `collector/wunderground_fetcher.py`.
- Extended PWS pipeline defaults/fallbacks/coords/state grids in `collector/pws_fetcher.py`.
- Updated NBM/LAMP station metadata for new stations in:
  - `collector/nbm_fetcher.py`
  - `collector/lamp_fetcher.py`
- Refreshed WU PWS registry with discovered nearby stations:
  - `data/wu_pws_station_registry.json`
- Added implementation/research note:
  - `docs/NEW_MARKETS_CHICAGO_MIAMI_DALLAS_2026-02-22.md`

## 2026-02-21

### METAR decoding hardening (no T-group => range, not point)
**Goal**: remove false precision in METAR observations when `T group` is missing and keep a settlement-safe range.

**Changes**:
- Added unified METAR temperature parser from raw line: `collector/metar/temperature_parser.py`.
- Updated NOAA JSON/XML/TXT fetchers to use the same decode logic.
- Added range fields (`temp_f_low/high`, `settlement_f_low/high`, `has_t_group`) to METAR flow and world serialization.
- Added QC flag `TEMP_RANGE_<low>-<high>F_NO_T_GROUP` and `UNCERTAIN` status when settlement range spans more than one integer.
- Exposed range fields in realtime endpoint `/api/realtime/{station_id}`.
- Added parser tests: `tests/test_metar_temperature_parser.py`.

**Evidence**:
- Historical racing logs show this occurred in production (`81158` racing rows, `623` rows with NOAA pair disagreement `>=1F`).

**Reference**:
- Full technical note: `docs/FIX_METAR_T_GROUP_RANGE_2026-02-21.md`.

## 2026-01-27

### 00:55 - Integration of SEARA Logic (High-Speed Websockets)
**Goal**: Reduce latency and improve market data freshness by adopting the "Orderbook Mirror" architecture from the Rust-based SEARA project.

**Changes**:
- **New Core Component**: Created `market/orderbook.py` implementing `LocalOrderBook`. This maintains an in-memory copy of the orderbook using `O(1)` dictionary lookups.
- **WebSocket Upgrade**: Enhanced `market/polymarket_ws.py` to support real-time Snapshots and Deltas from the Polymarket CLOB API.
    - Implemented robust reconnection logic (`ws.state` checks).
    - Added `Book` channel subscription.
- **Refactoring**: Created `market/discovery.py` to handle market token discovery and resolve circular dependency issues between `web_server.py` and `polymarket_ws.py`.
- **API Real-time Bridge**: Updated `web_server.py` to:
    - Initialize the WebSocket client as a background task (`websocket_loop`).
    - Added new endpoint `GET /api/realtime/market/{station_id}` that serves the current memory state instantly, bypassing slow HTTP requests.
- **Verification**: Verified end-to-end functionality including correct handling of `ws.state` (Enum) attributes.

**Status**: ✅ Completed & Verified. Realtime endpoint is delivering live market depth.

### 01:55 - Phase 2: Event-Driven World Layer Architecture
**Goal**: Decouple data ingestion from consumption using a robust, event-driven "World State", ensuring data integrity, correct timestamping (UTC/NYC/Madrid), and quality control.

**Changes**:
- **Data Contracts (`core/models.py`)**: Defined strict schemas (`OfficialObs`, `AuxObs`) with dual timestamps and source tracking.
- **Judge Alignment (`core/judge.py`)**: Implemented standardized rounding logic (e.g., 26.5 → 27) to match settlement sources exactly.
- **Quality Control (`core/qc.py`)**: Added "Hard Rules" (physical limits) and "Spatial Rules" (MAD clustering) for outlier detection.
- **World State Engine (`core/world.py`)**: Created an in-memory Singleton ("World Tape") that acts as the Single Source of Truth, storing the latest state and a history buffer.
- **Event-Driven Ingestion**: Refactored `metar_fetcher.py` to publish `OfficialObs` events directly to the World State upon winning the "METAR Race".
- **Real-time API v2**: Added `GET /api/v2/world/snapshot` to serve the full World State instantly from RAM.
- **Runner Scripts**: Added `run_helios.bat` and `run_helios.ps1` for easy one-click startup.

**Status**: ✅ Completed & Verified. System is now feeding live events into the World State.

## 2026-01-28

### Phase 2 Completion: Full Event-Driven World Layer + Real-Time Dashboard
**Goal**: Complete the Phase 2 specification — full health tracking, QC integration, EnvFeature/AuxObs publishing, SSE streaming, and a real-time World Dashboard UI.

**Changes**:

- **WorldState Upgrade (`core/world.py`)**:
    - Added `SourceHealth` dataclass tracking per-source staleness, latency, update/error counts, and status (LIVE/OK/STALE/DEAD/NO_DATA).
    - Added `QCState` tracking per station with status flags and last check timestamp.
    - Implemented SSE (Server-Sent Events) subscriber management: `subscribe_sse()`, `unsubscribe_sse()`, `_notify_sse()` with queue-based fan-out.
    - Enhanced `get_snapshot()` with dual timestamps (UTC/NYC/Madrid), rich per-type serialization, QC state, health metrics, tape size, and SSE subscriber count.
    - Added `record_error()` for tracking source failures.
    - Added `set_qc_state()` for updating station QC status.

- **EnvFeature Publishers (`main.py`)**:
    - SST (Sea Surface Temp) from `ndbc_fetcher` now publishes `EnvFeature(type=SST)` to WorldState after each buoy fetch.
    - AOD (Aerosol Optical Depth) from `cams_fetcher` now publishes `EnvFeature(type=AOD)` to WorldState after each fetch.

- **AuxObs Publisher (`main.py`)**:
    - Upstream METAR data from `advection_fetcher` now publishes `AuxObs` with drift calculation to WorldState.

- **QC Integration (`collector/metar_fetcher.py`)**:
    - METAR publish flow now runs `QualityControl.check_hard_rules()` before creating `OfficialObs`.
    - QC result (passed/flags) embedded directly in the `OfficialObs` event.
    - Station QC state updated in WorldState via `set_qc_state()` on every METAR publish.
    - Error recording via `record_error()` on publish failure.

- **SSE Streaming Endpoint (`web_server.py`)**:
    - New `GET /api/v2/world/stream` SSE endpoint for real-time push from WorldState.
    - Sends initial full snapshot on connect, then incremental updates on each event.
    - 15-second keepalive heartbeat.
    - Proper cleanup on client disconnect.

- **World Dashboard UI (`templates/world.html`)**:
    - New full-featured real-time dashboard at `/world`:
        - **System Bar**: Dual timestamps (NYC + Madrid + UTC) with live SSE connection indicator.
        - **METAR Panel**: Current official observation with temp (F/C), station, source, source age, dewpoint, wind, sky condition, QC status + flags, and triple timestamps (UTC/NYC/Madrid).
        - **Upstream/Auxiliary Panel**: Shows all AuxObs with temperature, drift from official, support count, and source age.
        - **Environment Features Panel**: SST and AOD cards with value, source, age, location reference, and color-coded severity.
        - **QC Panel**: Per-station quality control status with color-coded dots (OK=green, UNCERTAIN=yellow, OUTLIER=red, SEVERE=pulsing red) and flag badges.
        - **Health & Latency Table**: Full-width table showing all sources with status dot, staleness, source age, update count, error count, and last seen time. Sorted by health priority.
    - All panels auto-update via SSE + 5-second polling fallback.
    - Multi-station support: Shows primary station prominently + additional stations inline.

- **Sidebar Navigation (`templates/base.html`)**:
    - Added "World State" link with globe icon to sidebar, accessible from all pages.

**Architecture**: The UI reads 100% from memory (WorldState) via SSE push. No database queries. No polling delays. All timestamps include NYC + Madrid + UTC for every event.

**Status**: ✅ Completed.

### Phase 2 Audit & Gap Completion
**Goal**: Full audit against the Phase 2 spec (sections 2.1–2.7). Fix all missing pieces.

**Gaps Found & Fixed**:

- **2.1 Data Contracts** (`core/models.py`):
    - Added `ingest_time_nyc` and `ingest_time_madrid` properties to base `ObservationEvent`.
    - Added `quality_flags` list and `qc_state` optional field to base contract.
    - Every event now carries dual timestamps for both `obs_time` and `ingest_time`.

- **2.2 METAR Race** (`collector/metar_fetcher.py`):
    - Added **dedupe by obs_time_utc**: if the same observation (within 5s) arrives again, WorldState publish is skipped.
    - Added **per-route latency tracking**: each METAR source is timed individually (`_route_latency_ms`). Latencies logged in `_race_latency_log` (bounded to 500 entries).
    - Race log now shows per-route ms: `NOAA_JSON_API=45F(120ms), AWC_TDS_XML=45F(340ms)`.

- **2.3 Judge Alignment** (`core/judge.py`):
    - Added **negative temperature handling** with explicit documentation (e.g., `-0.5 → -1` using round-half-away-from-zero).
    - Added **settlement day definition**: `settlement_date(utc_dt)` returns the NYC calendar date for any UTC datetime.
    - Added `is_same_settlement_day()` and `settlement_day_bounds()` for settlement window calculation.
    - Verified edge cases: `26.4→26`, `26.5→27`, `26.51→27`, `-0.5→-1`, `-1.5→-2`.

- **2.4 PWS Cluster + Consensus + QC** (`collector/pws_fetcher.py`) — **NEW MODULE**:
    - Created complete PWS fetcher using Open-Meteo grid points as pseudo-PWS cluster.
    - Station-specific config with radius, min_support, and 10 spatial offsets per station (KLGA, KATL, EGLC).
    - **Two-layer QC**:
        1. Hard rules (physical limits via `QualityControl.check_hard_rules`)
        2. Spatial MAD outlier detection (>3*MAD from median → OUTLIER)
    - Consensus output: `PWSConsensus` with `median_temp_c/f`, `mad`, `support`, `total_queried`, `outliers`, `drift_c`.
    - Publishes `AuxObs(is_aggregate=True, source="PWS_CLUSTER")` to WorldState.
    - PWS QC state pushed to WorldState (`PWS_OUTLIERS_N`, `PWS_LOW_SUPPORT`, `PWS_HIGH_SPREAD_MAD`).
    - Wired into `main.py` prediction loop: runs after advection fetch, uses official METAR temp for drift.

- **2.6 NDJSON Tape Logging** (`core/world.py`):
    - Added persistent NDJSON (Newline-Delimited JSON) logging to `logs/world_tape_YYYY-MM-DD.ndjson`.
    - Every event published to WorldState is simultaneously appended as one JSON line.
    - Auto-creates `logs/` directory, auto-rotates by date.
    - Designed for replay, debug, and future Parquet/ClickHouse ingestion.

- **UI: PWS Consensus Panel** (`templates/world.html`):
    - New dedicated "PWS Consensus" panel in World Dashboard showing:
        - Cluster median temperature (F + C)
        - Support count (number of valid stations)
        - Drift vs official METAR
        - Source age
    - Color-coded drift indicator (red=warm, blue=cold, gray=neutral).
    - Auto-filters AuxObs entries with `PWS_` prefix for display.

**Status**: ✅ Phase 2 Spec Fully Implemented.

## 2026-01-30

### v13.0: Post-Peak Cap Rule (Critical Logic Fix)
**Goal**: Fix a critical prediction logic gap where the system continued predicting temperatures significantly higher than observed maximum after the peak hour had passed.

**Problem Identified**:
The dashboard showed an illogical state at 15:19 NYC:
- Expected peak: 15:00
- Max observed: 19°F
- Predicted max: 24.0°F (5°F above observed!)

This was incorrect because once the peak hour passes, the temperature can only decrease - the prediction should converge toward the observed maximum.

**Root Cause Analysis**:
The system had a **logic gap** between 15:00 and 17:00:

| Time | Existing Logic |
|------|----------------|
| < 15:00 | Delta weight active, physics active |
| 15:00-17:00 | **GAP: No post-peak cap existed** |
| ≥ 17:00 | Sunset Hard Limit (cap = obs + 0.5°F) |

The "Sunset Hard Limit" (Rule 8) only activated at 17:00, but the typical temperature peak occurs around 15:00. During this 2-hour window, predictions could remain arbitrarily high above the observed maximum.

**Solution Implemented**:

- **New Parameter** (`synthesizer/physics.py:80`):
  - Added `peak_hour: Optional[int] = None` parameter to `calculate_physics_prediction()`.
  - Allows the physics engine to know when the expected peak occurs (from HRRR hourly data).

- **New Rule 7b: Post-Peak Cap** (`synthesizer/physics.py:351-381`):
  ```
  Activates when: local_hour >= peak_hour (or 15:00 if unknown)
  Cap formula: verified_floor_f + margin

  Margin calculation (linear decay):
  - At peak hour: margin = 2.0°F (some uncertainty remains)
  - At sunset (17:00): margin = 0.5°F (highly certain)
  ```

- **Peak Hour Extraction** (`main.py:121-133`):
  - Extracts `peak_hour` from HRRR hourly temperature data.
  - Finds the hour (0-23) with maximum temperature in the forecast.
  - Only calculated for day_offset=0 (today).

- **Integration** (`main.py:299, 477`):
  - Both calls to `calculate_physics_prediction()` now pass `peak_hour`.

**Example Fix Calculation**:
With the observed data (15:19 NYC, peak 15:00, observed max 19°F):
```
hours_to_sunset = 17 - 15 = 2
margin = 0.5 + 1.5 * (2/2) = 2.0°F
post_peak_cap = 19°F + 2.0°F = 21.0°F

Old prediction: 24.0°F → New prediction: 21.0°F
```

At 16:00 the cap would be even tighter:
```
hours_to_sunset = 1
margin = 0.5 + 1.5 * (1/2) = 1.25°F
post_peak_cap = 19°F + 1.25°F = 20.3°F
```

**Files Modified**:
- `synthesizer/physics.py`: Added `peak_hour` parameter and Rule 7b Post-Peak Cap
- `main.py`: Extract peak_hour from HRRR data and pass to physics engine
- `core/nowcast_engine.py`: Added Post-Peak Cap to `_calculate_adjusted_tmax()` method

**Important Discovery**: The dashboard uses the **Nowcast Engine** (`/api/v3/nowcast/`) for real-time predictions, which is separate from the physics engine. The fix needed to be applied to BOTH systems:
1. `synthesizer/physics.py` - for CLI predictions and database logging
2. `core/nowcast_engine.py` - for the real-time dashboard display

**Status**: ✅ Completed. Both physics engine and nowcast engine now apply Post-Peak Cap.

### Phase 3: Nowcast Engine Integration
**Goal**: Implement a real-time probabilistic nowcast engine that produces calibrated probability distributions over temperature buckets.

**Changes**:

- **NowcastEngine (`core/nowcast_engine.py`)**:
    - Created probabilistic nowcast engine with `NowcastDistribution` output containing:
        - `tmax_mean_f`: Point estimate of max temperature
        - `tmax_sigma_f`: Uncertainty (standard deviation)
        - `p_bucket`: List of bucket probabilities (market-aligned)
        - `t_peak_bins`: Probability distribution over peak timing bins
        - `confidence`: Overall confidence score
        - `bias_applied_f`: Bias correction applied
    - Implements EMA bias tracking with configurable decay
    - QC-aware uncertainty scaling (higher uncertainty when data quality is poor)

- **NowcastIntegration (`core/nowcast_integration.py`)**:
    - Singleton integration layer connecting METAR observations to nowcast outputs
    - **Trigger A (Periodic)**: Updates every 60 seconds
    - **Trigger B (Event-driven)**: Updates on new METAR observation via `on_metar(obs)`
    - **Trigger C (Auxiliary)**: Updates on PWS/upstream observations
    - Callback system for notifying consumers of distribution updates
    - SSE listener for real-time event streaming

- **API Endpoints (`web_server.py`)**:
    - `GET /api/v3/nowcast/{station_id}`: Get current nowcast distribution
    - `GET /api/v3/nowcast/snapshot`: Get all stations' nowcast state

- **UI Integration**:
    - Dashboard displays nowcast probabilities and confidence metrics
    - Real-time updates via SSE when distributions change

**Status**: ✅ Completed.

### Phase 4: Storage & Replay System
**Goal**: Implement persistent event recording and replay capability for backtesting and debugging.

**Changes**:

- **Recorder Module (`core/recorder.py`)**:
    - `Recorder` class with async NDJSON writers for persistence
    - Channels supported: `world`, `pws`, `features`, `nowcast`, `market`, `l2_snap`, `health`, `event_window`
    - Ring buffers for live UI (in-memory, configurable size)
    - Channel-specific recording methods:
        - `record_metar()`: METAR observations
        - `record_pws()`: PWS cluster consensus
        - `record_features()`: Feature vectors
        - `record_nowcast()`: Nowcast distributions
        - `record_health()`: System health metrics
        - `record_event_window()`: Event window markers
    - Directory structure: `data/recordings/date=YYYY-MM-DD/ch=channel/events.ndjson`
    - Stats tracking: events per channel, disk usage, recording duration

- **Compactor Module (`core/compactor.py`)**:
    - `Compactor` class for NDJSON → Parquet conversion (requires pyarrow)
    - `ParquetReader` for efficient columnar reads
    - **`HybridReader`** (key component):
        - Reads from both NDJSON (fresh recordings) and Parquet (compacted)
        - `list_available_dates()`: Get dates with recorded data
        - `list_channels_for_date()`: Get channels available for a date
        - `read_channel()`: Read events from a channel
        - `get_events_sorted()`: Get all events sorted by timestamp with station filtering

- **Replay Engine (`core/replay_engine.py`)**:
    - `ReplayEngine` singleton for coordinating replays
    - `ReplaySession` class for stateful playback:
        - Timeline playback with configurable speed (1x, 2x, 10x, etc.)
        - Pause/resume/seek functionality
        - Event-by-event iteration
        - Channel filtering
    - Uses HybridReader to access both NDJSON and Parquet data

- **API Endpoints (`web_server.py`)**:
    - `GET /api/v4/recorder/stats`: Recording statistics
    - `GET /api/v4/replay/dates`: List available dates
    - `GET /api/v4/replay/dates/{date}/channels`: List channels for a date
    - `POST /api/v4/replay/start`: Start a replay session
    - `POST /api/v4/replay/pause`: Pause replay
    - `POST /api/v4/replay/resume`: Resume replay
    - `POST /api/v4/replay/seek`: Seek to timestamp
    - `GET /api/v4/replay/events`: Get events (polling)
    - `GET /api/v4/replay/stream`: SSE stream for real-time playback

- **Replay UI (`templates/replay.html`)**:
    - Date selector with available dates from recordings
    - Channel filter checkboxes
    - Playback controls: Play, Pause, Speed (1x/2x/5x/10x)
    - Timeline scrubber with seek capability
    - Event log panel with syntax-highlighted JSON
    - Station state panel showing current values during playback

- **Navigation**:
    - Added "Replay" link to sidebar in `templates/base.html`

**Status**: ✅ Completed.

### Phase 5: Backtesting + Calibration Loop
**Goal**: Build a complete backtesting framework to evaluate model calibration and simulate trading execution.

**Architecture Overview**:
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  HybridReader   │────▶│ DatasetBuilder  │────▶│ BacktestEngine  │
│ (NDJSON+Parquet)│     │ (Timeline Join) │     │ (Orchestration) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                        ┌───────────────────────────────┼───────────────────────────────┐
                        ▼                               ▼                               ▼
              ┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
              │  LabelManager   │           │ MetricsCalculator│          │    Policy       │
              │ (Ground Truth)  │           │ (Brier/ECE/MAE) │           │ (State Machine) │
              └─────────────────┘           └─────────────────┘           └─────────────────┘
                                                                                    │
                                                                          ┌─────────────────┐
                                                                          │ExecutionSimulator│
                                                                          │(Taker/Maker)    │
                                                                          └─────────────────┘
```

**Core Modules Implemented**:

- **Labels Module (`core/backtest/labels.py`, ~320 lines)**:
    - `DayLabel` dataclass with:
        - `y_bucket_winner`: Winning bucket per market rules
        - `y_tmax_aligned`: Judge-aligned Tmax (contractual rounding)
        - `y_t_peak_bin`: Hour bin where maximum occurred
        - `y_tmax_physical`: Raw observed maximum
    - `LabelManager` class:
        - Default bucket definitions (Polymarket-style)
        - Judge alignment (round-half-up)
        - `derive_label_from_observations()`: Auto-derive labels from METAR data
        - Persistent storage in `data/labels/`

- **Dataset Module (`core/backtest/dataset.py`, ~350 lines)**:
    - `TimelineState` dataclass: World state at any timestamp
        - Last METAR, PWS, upstream observations with staleness
        - Current max observed, features, nowcast, market state
        - QC flags, active event windows
    - `BacktestDataset`: Complete dataset for one market day
        - Events by channel (world, pws, features, nowcast, market)
        - `iterate_timeline()`: Asof join with configurable interval
    - `DatasetBuilder`: Uses HybridReader to construct datasets

- **Metrics Module (`core/backtest/metrics.py`, ~450 lines)**:
    - **Calibration Metrics**:
        - `brier_score()` / `brier_score_multi()`: Squared error of probabilities
        - `log_loss()` / `log_loss_multi()`: Cross-entropy loss
        - `expected_calibration_error()`: ECE with binning
        - `reliability_diagram_data()`: For calibration curves
        - `sharpness()`: Distribution informativeness
    - **Point Metrics**:
        - `mae()`, `rmse()`, `bias()`: Tmax prediction errors
    - **Stability Metrics**:
        - `prediction_churn()`: Sum of |ΔP(bucket)| over time
        - `flip_count()`: Number of predicted winner changes
    - **T-Peak Metrics**:
        - `tpeak_accuracy()`, `tpeak_mass_near_truth()`
    - `MetricsCalculator` class: Aggregates metrics across days
    - `CalibrationMetrics` dataclass: All computed metrics

- **Policy Module (`core/backtest/policy.py`, ~420 lines)**:
    - `PolicyState` enum: `NEUTRAL`, `BUILD_POSITION`, `HOLD`, `REDUCE`, `FADE_EVENT`, `RISK_OFF`
    - `PolicyTable` dataclass:
        - Edge thresholds for entry/exit
        - Sizing parameters (base size, confidence scaling)
        - Risk limits (max position, daily loss limit)
        - Maker/taker preference
    - `Policy` class:
        - `evaluate()`: Produces `PolicySignal` from nowcast + market state
        - State machine transitions based on conditions
        - Position tracking per bucket
    - Factory functions: `create_conservative_policy()`, `create_aggressive_policy()`, `create_fade_policy()`

- **Simulator Module (`core/backtest/simulator.py`, ~400 lines)**:
    - `TakerModel`: Immediate execution with fees + slippage
        - `fee_rate`: Percentage fee (default 0.1%)
        - `base_slippage`: Fixed slippage (default 0.5%)
        - `size_impact`: Slippage scaling with size
    - `MakerModel`: Queue-based fill simulation
        - Queue position estimation from L2 depth
        - Fill probability based on queue consumption
        - Time-in-queue requirements
    - `Fill` dataclass with adverse selection tracking:
        - `price_1s_after`, `price_5s_after`, `price_30s_after`
    - `ExecutionSimulator`: Combines taker and maker models

- **Engine Module (`core/backtest/engine.py`, ~450 lines)**:
    - `BacktestMode` enum: `SIGNAL_ONLY`, `EXECUTION_AWARE`
    - `DayResult` dataclass: Per-day results with metrics and trading stats
    - `BacktestResult` dataclass: Aggregated results with:
        - Trading summary (PnL, fills, win rate, Sharpe, drawdown)
        - Coverage stats (days with data/labels)
        - `to_dict()`: JSON-serializable with `safe_float()` for inf/nan handling
        - `save()`: Persist to JSON file
        - `summary()`: Text summary generation
    - `BacktestEngine` class:
        - `run()`: Main orchestration for date range
        - `_run_day()`: Single day processing
        - Progress callback support
        - Convenience functions: `run_signal_backtest()`, `run_execution_backtest()`

- **Calibration Module (`core/backtest/calibration.py`, ~380 lines)**:
    - `ParameterSet`: Grid of parameter values for tuning
    - `ObjectiveFunction`: Weighted combination of metrics
        - Brier weight, ECE weight, PnL weight, drawdown penalty
    - `CalibrationLoop` class:
        - `grid_search()`: Exhaustive parameter search
        - `walk_forward()`: Rolling window validation
    - `NoLeakageValidator`: Ensures no future data leakage

- **Report Module (`core/backtest/report.py`, ~400 lines)**:
    - `ReportGenerator` class:
        - `generate_html()`: Full HTML report with charts
        - `generate_markdown()`: Markdown summary

**CLI (`backtest_cli.py`)**:
```bash
python backtest_cli.py run --station KLGA --start 2026-01-01 --end 2026-01-30
python backtest_cli.py calibrate --station KLGA --param bias_alpha
python backtest_cli.py report --input results.json --format html
python backtest_cli.py list --station KLGA
python backtest_cli.py validate --station KLGA --date 2026-01-15
```

**API Endpoints (`web_server.py`)**:
- `GET /api/v5/backtest/dates`: List dates with recorded data
- `POST /api/v5/backtest/run`: Run a backtest
- `GET /api/v5/backtest/labels/{station_id}`: Get labels for a station
- `GET /api/v5/backtest/metrics`: Get available metrics definitions
- `GET /api/v5/backtest/policies`: Get available policies

**Backtest UI (`templates/backtest.html`)**:
- Station selector (KLGA, KATL, EGLC)
- Date range pickers with available dates
- Mode selector: Signal Only vs Execution Aware
- Policy selector: Conservative, Aggressive
- Results panel:
    - Summary stats (PnL, Win Rate, Sharpe, Drawdown, Days, Fills)
    - Trading stats hidden as "N/A" in Signal Only mode
    - Coverage info: "X/Y days with labels"
- Calibration metrics panel:
    - Brier Score, Log Loss, ECE, Sharpness
- Point prediction panel:
    - Tmax MAE, RMSE, Bias
    - Avg Churn
- Day results table:
    - Date, Predicted bucket, Actual bucket
    - Pred Tmax, Actual Tmax, PnL

**Tests (`tests/test_backtest.py`)**:
- 30 comprehensive unit tests covering:
    - Metrics calculations (Brier, Log Loss, ECE, MAE, RMSE, Bias, Churn)
    - Label creation and management
    - Policy state machine and edge calculations
    - Simulator models (Taker, Maker)
    - Dataset timeline state
    - Calibration parameter grids
    - No-leakage validation
    - Integration tests

**Bug Fixes During Implementation**:
1. **JSON serialization error**: `float('inf')` values in staleness not JSON-serializable
   - Fix: Added `safe_float()` helper to convert inf/nan to None
2. **Station data mixing**: Dataset returned events from all stations
   - Fix: Added station_id filtering in `HybridReader.get_events_sorted()`
3. **p_bucket key mismatch**: Recorder used `prob`, engine expected `probability`
   - Fix: Support both keys in engine and metrics
4. **Metrics always zero**: Bucket labels didn't match between model and labels
   - Fix: Added temperature-based bucket matching when exact label match fails
5. **METAR not recording**: Only nowcast was being recorded
   - Fix: Added `register_metar_callback()` to NowcastIntegration

**Status**: ✅ Completed. Backtest framework fully operational with calibration metrics.

### Prediction Breakdown Display Fix (v13.1)
**Goal**: Fix incomplete prediction breakdown display in the Nowcast Dashboard.

**Problem Identified**:
The "Key Factors" section in the dashboard only showed 3 items:
- 1_BASE: Base HRRR forecast
- 2_BIAS: Bias correction
- 3_PEAK: Peak hour status

Missing were the critical steps:
- 4_OBS_MAX: Observed maximum (floor constraint)
- 5_CAP: Post-Peak Cap (the key constraint)
- 6_FINAL: Final prediction result

**Root Cause**:
In `core/nowcast_engine.py`, the `_build_explanations()` method had this line at the end:
```python
return explanations[:3]  # Top 3
```

This was truncating the full 6-step breakdown to only the first 3 items, regardless of how many were generated.

**Solution Implemented**:

1. **Removed truncation** (`core/nowcast_engine.py:771`):
   - Changed `return explanations[:3]` to `return explanations`
   - Now all 6 steps are returned to the frontend

2. **Added diagnostic messages for missing data** (`core/nowcast_engine.py:725-760`):
   - When `obs_max` is None (no METAR observations yet):
     - Step 4 shows: "⚠️ No METAR observations yet - awaiting data"
   - When peak has passed but no observations exist:
     - Step 5 shows: "⚠️ PEAK PASSED but no observations! Cap cannot apply"

**Expected Display After Fix**:
```
1_BASE     Base HRRR forecast: 24.0°F
2_BIAS     Bias correction: +0.00°F → 24.0°F
3_PEAK     Peak hour: 15:00 NYC (PASSED, now=17:00)
4_OBS_MAX  Observed max: 19°F
5_CAP      ⚠️ POST-PEAK CAP: 19°F + 1.5°F margin = 20.5°F (reduced from 24.0°F)
6_FINAL    ═══ FINAL PREDICTION: 20.5°F ═══
```

**Files Modified**:
- `core/nowcast_engine.py`: Removed `[:3]` truncation, added diagnostic messages

**Status**: ✅ Completed.

### Post-Peak Cap Trigger Fix (v13.2)
**Goal**: Fix overly aggressive Post-Peak Cap that was capping predictions too early.

**Problem Identified**:
Model predicted 17°F when Wunderground showed 24-25°F and markets bet on 24-27°F.

**Root Cause**:
The Post-Peak Cap condition `current_hour >= peak_hour` was triggering **at** the peak hour, not **after** it.

Example:
```
Peak hour: 14:00
Current time: 14:00 (justo a la hora pico)
Observed max: 17°F (de la mañana fría)
Base forecast: 26°F
Condition: 14 >= 14 → TRUE → Cap se aplica!
Cap = 17 + 2.0 = 19°F
Result: 26°F capped to 19°F (WRONG!)
```

**Solution Implemented**:

Changed the trigger condition to require **1+ hour AFTER** the peak:

1. **`core/nowcast_engine.py` (línea 376)**:
   ```python
   # ANTES:
   if state.max_so_far_aligned_f > -900 and current_hour >= peak_hour:

   # DESPUÉS:
   if state.max_so_far_aligned_f > -900 and current_hour >= peak_hour + 1:
   ```

2. **`synthesizer/physics.py` (línea 363)**:
   ```python
   # ANTES:
   if local_hour >= effective_peak_hour:

   # DESPUÉS:
   if local_hour >= effective_peak_hour + 1:
   ```

3. **Updated explanation messages** to reflect new timing:
   - "At peak hour - cap activates in Xh"
   - "Post-peak (+1h): prediction already ≤ observed max"

**New Behavior**:
- Peak hour 14:00 → Cap activates at 15:00
- Peak hour 15:00 → Cap activates at 16:00

**Files Modified**:
- `core/nowcast_engine.py`: Changed trigger from `>= peak_hour` to `>= peak_hour + 1`
- `synthesizer/physics.py`: Same change for Rule 7b

**Status**: ✅ Completed.


### New Markets Completion (2026-02-22, final pass)
**Goal**: Close remaining integration gaps after adding `KORD`, `KMIA`, `KDAL`.

**Additional work completed**:
- Added NDBC mapping for Miami:
  - `collector/ndbc_fetcher.py`: `KMIA -> VAKF1` (primario) con `41122` como fallback.
  - Added onshore sector for KMIA (`40-170 deg`).
  - Improved NDBC parser to skip newest rows with `WTMP=MM` and use the first recent valid `WTMP`.
- Updated WU discovery deployment template:
  - `deploy/systemd/helios-wu-discover.service` now refreshes
    `KLGA,KATL,KORD,KMIA,KDAL,EGLC,LTAC`.
- Updated web labels/fallbacks for new stations:
  - `templates/world.html`
  - `templates/nowcast.html`
  - `templates/replay.html`
  - `templates/autotrader.html`
  - `static/app.js` fallback station list.
- Updated docs to reflect expanded station set and fallback behavior:
  - `docs/DEPLOY.md`
  - `docs/WUNDERGROUND_PWS.md`
  - `docs/NEW_MARKETS_CHICAGO_MIAMI_DALLAS_2026-02-22.md`

**Status**: completed.

