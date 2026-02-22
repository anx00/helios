# HELIOS Data Sources — Technical Reference

This document describes all data sources used by HELIOS for temperature prediction.

## Overview: Data Source Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA SOURCE HIERARCHY                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PRIMARY (Settlement Authority)                                  │
│  └── METAR: Official airport observations (NOAA)                │
│                                                                  │
│  SECONDARY (Validation & Confirmation)                          │
│  ├── PWS Cluster: Personal weather station consensus           │
│  └── Upstream Advection: Virtual upwind location               │
│                                                                  │
│  ENVIRONMENTAL (Adjustment Factors)                             │
│  ├── SST: Sea surface temperature (NDBC buoys)                 │
│  └── AOD: Aerosol optical depth (CAMS/OpenAQ)                  │
│                                                                  │
│  MODEL FORECASTS (Base Layer)                                   │
│  ├── HRRR: High-Resolution Rapid Refresh (primary US)          │
│  ├── GFS: Global Forecast System (secondary US)                │
│  ├── UKV: UK Met Office model (primary UK)                     │
│  ├── LAMP: Localized Aviation MOS Program                      │
│  └── NBM: National Blend of Models                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. METAR Observations (Primary)

### What It Is

METAR (Meteorological Aerodrome Report) is the official aviation weather observation format. This is the **settlement authority** — the temperature that determines market resolution.

### Data Sources (3-Way Race Protocol)

HELIOS uses a "racing" protocol to maximize reliability:

| Source | URL Pattern | Format | Priority |
|--------|-------------|--------|----------|
| NOAA JSON API | `aviationweather.gov/api/data/metar` | JSON | Primary |
| AWC TDS XML | Aviation Weather Center TDS | XML | Secondary |
| TGFTP Text | TGFTP bulletin feed | Raw text | Fallback |

**Racing Logic:**
1. All three sources fetched in parallel (3-second timeout each)
2. Winner = most recent observation time
3. Tie-breaker = highest temperature
4. Each route's latency is logged for monitoring

### Fields Extracted

| Field | Type | Description |
|-------|------|-------------|
| `station_id` | string | ICAO code (KLGA, KATL, EGLC) |
| `temp_c` | float | Temperature in Celsius |
| `temp_f` | int | **Judge-aligned** (rounded half-up) |
| `temp_f_raw` | float | With decimals for display |
| `dewpoint_c` | float | Dewpoint temperature |
| `humidity_pct` | float | Calculated from Magnus-Tetens |
| `wind_dir_degrees` | int | Wind direction (0-360°) |
| `wind_speed_kt` | int | Wind speed in knots |
| `sky_condition` | string | CLR, FEW, SCT, BKN, OVC |

### Quality Control

**Hard Rules:**
- Temperature bounds: -50°C to +60°C
- Dewpoint must be ≤ temperature

**Flags:**
- `TEMP_OUT_OF_BOUNDS`: Temperature outside physical limits
- `DEWPOINT_GT_TEMP`: Physically impossible (sensor error)

### Update Frequency

~60 minutes (METARs issued at :53 each hour, with SPECIs for significant changes)

### Implementation

`main.py:fetch_metar()`, `core/models.py:OfficialObs`

---

## 2. PWS Cluster (Secondary)

### What It Is

Personal Weather Stations (PWS) are citizen-operated weather stations within ~40km of the target airport. HELIOS aggregates them into a **consensus temperature** to validate METAR and detect anomalies.

### Key Metrics

| Metric | Definition | Example |
|--------|------------|---------|
| **Support** | Number of valid stations after QC | 12 |
| **Drift** | PWS consensus - METAR official | -1.5°F |
| **MAD** | Median Absolute Deviation | 0.8°F |

### Data Sources

**Layer 1: Synoptic Data API (Real PWS)**
- URL: `api.synopticdata.com/v2/stations/latest`
- Requires: `SYNOPTIC_API_TOKEN` environment variable
- Radius: 40km around station
- Max stations: 50
- Filter: Active stations, observations < 90 minutes old

**Layer 2: Open-Meteo Grid (Fallback)**
- Virtual grid points at fixed offsets
- 10 points per station (±0.05°, ±0.10°, ±0.15° lat/lon)
- Used when Synoptic unavailable

### Consensus Algorithm

```
1. HARD RULES QC
   Remove temps outside [-50°C, +60°C]

2. MAD OUTLIER DETECTION
   median = median(all_temps)
   mad = median(|temp - median|)

   For each temp:
     z_score = |temp - median| / mad
     if z_score > 3.0:
       mark as OUTLIER, exclude

3. FINAL CONSENSUS
   consensus = median(valid_temps)
```

### Output Structure

```python
AuxObs(
    station_id="PWS_KLGA",
    temp_c=consensus_celsius,
    temp_f=consensus_fahrenheit,
    is_aggregate=True,
    support=12,              # Stations after QC
    drift=-1.5,              # Celsius difference from METAR
    source="PWS_SYNOPTIC"
)
```

### Interpreting Drift

| Drift Value | Interpretation |
|-------------|----------------|
| -2°F to +2°F | Normal variation |
| > +3°F | PWS cluster warmer (urban heat? sensor drift?) |
| < -3°F | PWS cluster cooler (coastal effect? cold pocket?) |
| Very high drift | Possible METAR error or local anomaly |

### Quality Flags

- `PWS_LOW_SUPPORT`: Fewer than 5 stations
- `PWS_HIGH_SPREAD_MAD_X`: MAD exceeds 2.0°F
- `PWS_OUTLIERS_N`: N stations flagged as outliers

### Implementation

`main.py:fetch_and_publish_pws()`, `core/models.py:AuxObs`

---

## 3. Environmental Features

### 3a. SST (Sea Surface Temperature)

**What It Is:**
Water temperature from NOAA NDBC buoys, used to adjust predictions for coastal stations affected by sea breeze.

**Data Source:**
- URL: `ndbc.noaa.gov/data/realtime2/{buoy_id}.txt`
- Format: Space-delimited text file

**Buoy Assignments:**
| Station | Buoy ID | Location |
|---------|---------|----------|
| KLGA | 44065 | NY Harbor Entrance (15 NM SE) |
| KATL | 41008 | Grays Reef (40 NM SE Savannah) |
| EGLC | — | No buoy mapped |

**Fields Extracted:**
- `water_temp_c`: WTMP field
- `water_temp_f`: Converted
- `wind_dir`, `wind_speed_kt`
- `wave_height_m`

**Adjustment Logic:**
```python
if wind_is_onshore:
    delta = expected_sst - actual_sst
    if delta > 2.0:   # Water colder than expected
        adjustment = -delta × 0.5  # Cooling effect
    elif delta < -2.0:  # Water warmer
        adjustment = -delta × 0.3  # Smaller warming effect
```

**Onshore Wind Detection:**
- KLGA: Winds from 0-90° or 270-360° (NE to E, W to N)
- KATL: N/A (inland)

### 3b. AOD (Aerosol Optical Depth)

**What It Is:**
Measure of smoke, haze, or dust in the atmosphere that can affect solar radiation and temperature.

**Data Sources (with fallback):**

1. **CAMS (Primary)** — Copernicus Atmosphere Monitoring Service
   - API: `cds.climate.copernicus.eu`
   - Variable: `total_aerosol_optical_depth_550nm`
   - Update: Daily (00:00 UTC run)

2. **OpenAQ (Fallback)** — PM2.5 correlation
   - API: `api.openaq.org/v2/latest`
   - Correlation: PM2.5 → AOD approximation

**PM2.5 to AOD Mapping:**
| PM2.5 (μg/m³) | Approximate AOD | Condition |
|---------------|-----------------|-----------|
| ≤ 20 | 0.1 | Clear |
| 21-35 | 0.2 | Light haze |
| 36-55 | 0.4 | Moderate haze |
| > 55 | 0.6 | Heavy smoke |

**Temperature Adjustment:**
```python
if aod_550nm > 0.5:
    adjustment = -2.0°F  # Heavy smoke/haze
elif aod_550nm > 0.3:
    adjustment = -1.0°F  # Moderate haze
else:
    adjustment = 0.0°F
```

### Implementation

`main.py:fetch_buoy_sst()`, `main.py:fetch_aerosol_data()`, `core/models.py:EnvFeature`

---

## 4. Weather Model Forecasts

### 4a. HRRR/GFS (Primary Models)

**HRRR (High-Resolution Rapid Refresh)**
- Resolution: 3km grid
- Update: Hourly
- Horizon: 18 hours
- Use: Primary US model, hourly temperature curve

**GFS (Global Forecast System)**
- Resolution: ~25km grid
- Update: Every 6 hours
- Horizon: 16 days
- Use: Secondary US model, consensus check

**UKV (UK Met Office)**
- Used for: EGLC (London)
- Via: `ukmo_seamless` in Open-Meteo

**Data Source:**
- API: `api.open-meteo.com/v1/forecast`
- Parameters: `temperature_2m`, `soil_moisture`, `shortwave_radiation`, `cloud_cover`

**Consensus Calculation:**
```python
if both_available:
    consensus = (hrrr_max + gfs_max) / 2
    divergence = abs(hrrr_max - gfs_max)
else:
    consensus = available_model_max
```

**Fields Captured:**
- `t_hourly_f`: List of 24 hourly temperatures
- `t_max_base_f`: Daily maximum from curve
- `t_peak_hour`: Hour of maximum (0-23)
- `model_source`: "HRRR", "GFS", etc.

### 4b. LAMP (Localized Aviation MOS)

**What It Is:**
Hourly temperature forecast from NOAA's Model Output Statistics, bias-corrected for specific airports.

**Data Source:**
- Primary: Open-Meteo hourly forecast (via API)
- Horizon: 25 hours ahead

**Confidence Calculation:**
```python
# Based on 7-day rolling accuracy
mae = mean_absolute_error(lamp_predictions, actuals)
confidence = max(0.0, 1.0 - (mae / 6.0))
```

### 4c. NBM (National Blend of Models)

**What It Is:**
Ensemble consensus from 30+ weather models blended together.

**Data Source:**
- Open-Meteo ensemble forecast
- Or NOAA NOMADS text bulletins

**Fields:**
- `max_temp_f`: Daily maximum
- `min_temp_f`: Daily minimum

### Implementation

`main.py:fetch_hrrr()`, `main.py:fetch_lamp()`, `main.py:fetch_nbm()`

---

## 5. Advection Data (Upstream Weather)

### What It Is

Virtual "upstream" observation point 50-60km upwind of target station, used to detect incoming air masses before they arrive.

### Calculation

```python
1. Get current wind direction from METAR
2. Calculate point 50-60km in upwind direction:
   upstream_lat = station_lat + 0.5° × cos(wind_dir + 180°)
   upstream_lon = station_lon + 0.5° × sin(wind_dir + 180°)
3. Fetch weather from that virtual location via Open-Meteo
```

### Use Case

- If upstream is 3°F cooler → expect cooling trend in 1-3 hours
- If upstream is 3°F warmer → expect warming trend

### Implementation

`main.py:fetch_upstream_weather()`

---

## 6. Health Tracking

### Source Health States

| State | Definition | Age Threshold |
|-------|------------|---------------|
| `LIVE` | Data received within 2 minutes | < 2 min |
| `OK` | Data fresh | 2-10 min |
| `STALE` | Data aging, still usable | 10-30 min |
| `DEAD` | Data too old, unreliable | > 30 min |
| `NO_DATA` | Never received | — |

### Tracked Per Source

```python
SourceHealth:
    source_name: str          # "METAR_KLGA", "PWS_KLGA"
    last_seen_utc: datetime   # When data arrived
    update_count: int         # Total updates
    error_count: int          # Failures
    latency_s: float          # ingest_time - obs_time
    status: str               # LIVE/OK/STALE/DEAD
```

### Impact on Predictions

- `STALE` sources add +0.3°F to sigma per source
- `DEAD` sources add +0.3°F to sigma per source
- Multiple stale sources can reduce confidence by 5% each

---

## 7. Data Update Frequencies Summary

| Source | Update Frequency | Typical Latency |
|--------|------------------|-----------------|
| METAR | ~60 min (hourly) | 2-5 min |
| PWS Cluster | On-demand | 1-2 min |
| NDBC Buoy (SST) | 5-10 min | 5-15 min |
| CAMS AOD | Daily | 6-12 hours |
| OpenAQ PM2.5 | Real-time | 1-5 min |
| HRRR | Hourly | 30-60 min |
| GFS | 6-hourly | 2-4 hours |
| LAMP | Hourly | 30-60 min |
| NBM | 6-hourly | 2-4 hours |
| Advection | On-demand | Real-time |

---

## 8. Data Collection Order

In each prediction cycle, data is collected in this order:

```
1. METAR official observation
   └─ 3-way race, QC validation, dedupe

2. HRRR/GFS multi-model forecasts
   └─ Dual-model consensus

3. LAMP hourly MOS
   └─ 7-day confidence calculation

4. NBM ensemble blend
   └─ Daily max/min

5. PWS cluster consensus
   └─ MAD outlier detection, drift calculation

6. SST from NDBC buoy
   └─ Sea breeze adjustment (if applicable)

7. AOD from CAMS/OpenAQ
   └─ Smoke/haze adjustment

8. Upstream advection
   └─ Trend detection
```

**All events published to WorldState immediately** after collection.
