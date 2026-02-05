# HELIOS Architecture — Technical Reference

## System Overview

HELIOS is a real-time temperature forecasting system designed for Polymarket prediction markets. It uses a **two-layer architecture** that combines stable model forecasts with fast-reacting observation adjustments.

### Two-Layer Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    HELIOS PREDICTION ENGINE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ LAYER 1: BASE FORECAST (Slow, Stable)                    │   │
│  │                                                          │   │
│  │  • Source: HRRR/GFS hourly temperature curves            │   │
│  │  • Update frequency: 1-3 hours (model runs)              │   │
│  │  • Provides: T_base_max, peak_hour, hourly profile       │   │
│  │  • Stability: High - anchors the prediction              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            ↓                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ LAYER 2: NOWCAST ADJUSTMENT (Fast, Event-Driven)         │   │
│  │                                                          │   │
│  │  • Source: METAR observations, PWS consensus             │   │
│  │  • Update frequency: Every METAR (~60 min) or event      │   │
│  │  • Provides: Bias correction, uncertainty estimation     │   │
│  │  • Reactivity: High - responds to observations           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            ↓                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ OUTPUT: NowcastDistribution                              │   │
│  │                                                          │   │
│  │  • tmax_mean_f: Best estimate (e.g., 48.5°F)            │   │
│  │  • tmax_sigma_f: Uncertainty (e.g., 1.8°F)              │   │
│  │  • p_bucket: Probability per temperature bracket        │   │
│  │  • confidence: Quality score (0.0-1.0)                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Pipeline

```
                              DATA SOURCES
                                   │
    ┌──────────────────────────────┼──────────────────────────────┐
    │                              │                              │
    ▼                              ▼                              ▼
┌─────────┐                 ┌─────────────┐                ┌─────────────┐
│ METAR   │                 │ PWS Cluster │                │ Env Features│
│ (NOAA)  │                 │ (Synoptic)  │                │ (SST, AOD)  │
└────┬────┘                 └──────┬──────┘                └──────┬──────┘
     │                             │                              │
     │         ┌───────────────────┴───────────────────┐          │
     │         │                                       │          │
     ▼         ▼                                       ▼          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         WORLDSTATE                                   │
│  (Central Event Bus + State Manager)                                │
│                                                                      │
│  • Receives all observation events                                  │
│  • Maintains current state per station                              │
│  • Logs to NDJSON tape for replay                                   │
│  • Broadcasts via SSE to web clients                                │
│  • Tracks source health (LIVE/OK/STALE/DEAD)                        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      NOWCAST INTEGRATION                            │
│  (Coordinator for All Stations)                                     │
│                                                                      │
│  • Manages per-station NowcastEngine instances                      │
│  • Triggers: Periodic (60s) or Event-driven (METAR, QC)            │
│  • Coordinates distribution regeneration                            │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
     ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
     │NowcastEngine│      │NowcastEngine│      │NowcastEngine│
     │   (KLGA)    │      │   (KATL)    │      │   (EGLC)    │
     └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
            │                    │                    │
            ▼                    ▼                    ▼
     ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
     │Distribution │      │Distribution │      │Distribution │
     │   Output    │      │   Output    │      │   Output    │
     └─────────────┘      └─────────────┘      └─────────────┘
```

## Key Components

### 1. WorldState (`core/world.py`)

**Role**: Central event bus and state management singleton.

**Responsibilities**:
- Receives and validates all observation events
- Maintains head state (latest values per station)
- Keeps ring buffer of recent events (1000 items)
- Writes persistent NDJSON log for replay
- Publishes events via SSE to web clients
- Tracks source health per data source

**State Structure**:
```python
WorldState:
    latest_official: Dict[station_id → OfficialObs]   # Latest METAR
    latest_aux: Dict[station_id → AuxObs]             # Latest PWS consensus
    env_features: Dict[feature_type → EnvFeature]     # SST, AOD, etc.
    source_health: Dict[source_name → SourceHealth]   # Health tracking
    tape: Deque[ObservationEvent]                     # Ring buffer
```

### 2. NowcastEngine (`core/nowcast_engine.py`)

**Role**: Per-station prediction engine that generates probability distributions.

**Key Methods**:
- `update_base_forecast()`: Ingests HRRR/GFS model data
- `update_observation()`: Processes METAR, updates bias
- `generate_distribution()`: Creates full prediction output

**State Tracked (DailyState)**:
- `max_so_far_aligned_f`: Highest observed temp (judge-aligned)
- `bias_state`: EMA-tracked bias for model correction
- `trend_f_per_hour`: Temperature change rate
- `sigma_f`: Current uncertainty estimate
- `sources_health`: Per-source freshness status

### 3. NowcastIntegration (`core/nowcast_integration.py`)

**Role**: Coordinator that manages all station engines.

**Triggers**:
- **Periodic (60s)**: Refreshes time-based calculations
- **METAR event**: Immediate distribution update
- **QC change**: Adjusts uncertainty based on data quality
- **Base forecast update**: New model run arrives

### 4. Quality Control (`core/qc.py`)

**Role**: Validates incoming data before use.

**Two-Layer Validation**:
1. **Hard Rules**: Physical bounds (-50°C to +60°C)
2. **Spatial Consistency**: MAD-based outlier detection

**QC States**:
- `OK`: Data passes all checks
- `UNCERTAIN`: Minor issues, reduced weight
- `OUTLIER`: Spatial inconsistency detected
- `SEVERE`: Physical impossibility, rejected

### 5. Judge Alignment (`core/judge.py`)

**Role**: Ensures predictions match settlement rounding rules.

**Key Functions**:
- `round_to_settlement()`: Integer rounding (half-up)
- `celsius_to_fahrenheit()`: NOAA-standard conversion
- `settlement_date()`: NYC timezone day boundary

## Update Triggers

### Trigger A: Periodic Updates (every 60 seconds)
```
Timer → NowcastIntegration._periodic_update_loop()
    → For each station: engine.generate_distribution()
    → Publish to SSE subscribers
```

### Trigger B: Event-Driven Updates

| Event Type | Source | Action |
|------------|--------|--------|
| METAR observation | NOAA API | Update bias, regenerate distribution |
| PWS consensus | Synoptic API | Confirm/flag METAR, adjust uncertainty |
| Base forecast | Open-Meteo | Update T_base_max, peak_hour |
| QC state change | QC system | Adjust sigma penalty |
| Environmental feature | NDBC/CAMS | Apply SST/AOD adjustments |

## Station Configuration

HELIOS supports multiple stations, each with specific characteristics:

| Station | Location | Timezone | Characteristics |
|---------|----------|----------|-----------------|
| KLGA | New York LaGuardia | America/New_York | Coastal, sea breeze sensitive |
| KATL | Atlanta Hartsfield | America/New_York | Continental, storm cooling |
| EGLC | London City | Europe/London | Urban riverside, Thames effect |

## File Structure

```
helios-temperature/
├── core/
│   ├── world.py              # WorldState singleton
│   ├── nowcast_engine.py     # Per-station prediction engine
│   ├── nowcast_integration.py # Multi-station coordinator
│   ├── nowcast_models.py     # Data structures
│   ├── qc.py                 # Quality control system
│   ├── judge.py              # Settlement rounding
│   └── models.py             # OfficialObs, AuxObs, EnvFeature
├── market/
│   ├── polymarket_ws.py      # WebSocket orderbook
│   ├── polymarket_checker.py # Market resolution
│   └── crowd_wisdom.py       # Sentiment analysis
├── opportunity/
│   └── detector.py           # Bet opportunity detection
└── main.py                   # Collection orchestrator
```

## Output Contract

Every `NowcastDistribution` contains:

| Field | Type | Description |
|-------|------|-------------|
| `tmax_mean_f` | float | Best temperature estimate |
| `tmax_sigma_f` | float | Uncertainty (std dev) |
| `p_bucket` | List[BucketProbability] | Probability per bracket |
| `p_ge_strike` | Dict[int, float] | Cumulative probabilities |
| `confidence` | float | Quality score (0.0-1.0) |
| `explanations` | List[NowcastExplanation] | Calculation breakdown |
| `inputs_used` | List[str] | Data lineage |
| `valid_until_utc` | datetime | Expiration time |
