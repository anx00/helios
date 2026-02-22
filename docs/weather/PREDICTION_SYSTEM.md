# HELIOS Prediction System — Technical Reference

This document explains how HELIOS generates temperature predictions step-by-step, with complete numerical examples.

## Prediction Output Structure

Every prediction generates a `NowcastDistribution` with these components:

```python
NowcastDistribution:
    # Central Estimate
    tmax_mean_f: float           # Best estimate (e.g., 48.5°F)
    tmax_sigma_f: float          # Uncertainty (e.g., 1.8°F)

    # Probability Distributions
    p_bucket: List[BucketProbability]  # Per-bracket probabilities
    p_ge_strike: Dict[int, float]      # Cumulative probabilities

    # Time-of-Peak
    t_peak_bins: List[TPeakBin]        # 2-hour bin probabilities
    t_peak_expected_hour: int          # Most likely peak hour

    # Quality Metrics
    confidence: float                   # 0.0-1.0 score
    confidence_factors: List[str]       # Explanation of score

    # Transparency
    explanations: List[NowcastExplanation]  # Step-by-step breakdown
    inputs_used: List[str]                  # Data lineage
```

---

## Step-by-Step Prediction Process

### Step 1: Gather Base Forecast (Layer 1)

**Source:** HRRR hourly temperature curve

```python
base_forecast = BaseForecast(
    t_hourly_f=[22, 23, 24, 26, 29, 32, 35, 37, 38, 39, 40, 41,
                42, 43, 44, 44, 43, 42, 40, 38, 36, 34, 32, 30],
    t_max_base_f=44,           # Maximum from curve
    t_peak_hour=15,            # 3:00 PM local
    model_source="HRRR"
)
```

**Fallback:** If no base forecast, use `last_obs_temp + 5.0°F`

### Step 2: Calculate Bias (Layer 2)

**Current bias state:**
```python
bias_state = BiasState(
    current_bias_f=1.2,        # Model has been running warm
    observation_count=5,
    last_update_utc="2026-01-29 10:00:00"
)
```

**Decay the bias based on time to peak:**
```python
hours_to_peak = peak_hour - current_hour
               = 15 - 11 = 4 hours

tau = 2.0  # decay_hours / 3

decayed_bias = bias × exp(-hours_to_peak / tau)
             = 1.2 × exp(-4 / 2.0)
             = 1.2 × 0.135
             = 0.16°F
```

### Step 3: Calculate Adjusted Tmax

```python
T_adj = T_base_max + decayed_bias
      = 44.0 + 0.16
      = 44.16°F
```

### Step 4: Apply Constraints

**Floor Constraint:**
```python
if T_adj < max_so_far_aligned_f:
    T_adj = max_so_far_aligned_f

# Example: max_so_far = 41°F
# T_adj = max(44.16, 41) = 44.16°F (no change)
```

**Post-Peak Cap (only if current_hour >= peak_hour + 1):**
```python
# If it's 4:00 PM and peak was 3:00 PM:
margin = 0.5 + 1.5 × (hours_to_sunset / peak_to_sunset_span)
       = 0.5 + 1.5 × (1 / 2)  # 1 hour to sunset, 2 hour span
       = 0.5 + 0.75
       = 1.25°F

post_peak_cap = max_so_far + margin
              = 43 + 1.25
              = 44.25°F

T_adj = min(44.16, 44.25) = 44.16°F (no change)
```

**Final Result:**
```python
tmax_mean_f = 44.2°F  # Rounded to 1 decimal
```

### Step 5: Calculate Uncertainty (Sigma)

```python
# Start with base
sigma = 2.0°F

# QC state: OK → no penalty
sigma += 0.0

# METAR age: 15 minutes → +0.3
sigma += 0.3

# Stale sources: 1 source STALE → +0.3
sigma += 0.3

# Current total: 2.6°F

# Time multiplier: 11:00 AM → ×1.0
sigma *= 1.0

# Observation count: 5 → no bonus (need >10)
sigma *= 1.0

# Final
tmax_sigma_f = 2.6°F
```

### Step 6: Generate Bucket Probabilities

**Create buckets around center (cone_size=3):**
```python
center = round(44.2) = 44

buckets = [41-42, 42-43, 43-44, 44-45, 45-46, 46-47, 47-48]
```

**Calculate probability for each bucket:**

Using Logistic CDF with mean=44.2, sigma=2.6:

| Bucket | CDF(high) | CDF(low) | Raw Prob | Normalized |
|--------|-----------|----------|----------|------------|
| 41-42 | 0.1824 | 0.1192 | 0.0632 | 0.0648 |
| 42-43 | 0.2689 | 0.1824 | 0.0865 | 0.0887 |
| 43-44 | 0.3775 | 0.2689 | 0.1086 | 0.1114 |
| **44-45** | 0.5000 | 0.3775 | **0.1225** | **0.1257** |
| 45-46 | 0.6225 | 0.5000 | 0.1225 | 0.1257 |
| 46-47 | 0.7311 | 0.6225 | 0.1086 | 0.1114 |
| 47-48 | 0.8176 | 0.7311 | 0.0865 | 0.0887 |

**Note:** Probabilities normalized so sum = 1.0 (accounting for tail probabilities outside cone)

### Step 7: Calculate Cumulative Probabilities

```python
p_ge_strike = {}
for strike in range(39, 50):
    p_ge_strike[strike] = 1 - logistic_cdf(strike, 44.2, 2.6)

# Result:
{
    39: 0.9808,   # P(T >= 39°F) = 98.08%
    40: 0.9541,
    41: 0.9192,
    42: 0.8176,
    43: 0.7311,
    44: 0.6225,
    45: 0.5000,
    46: 0.3775,
    47: 0.2689,
    48: 0.1824,
    49: 0.1192
}
```

### Step 8: Calculate T-Peak Distribution

**2-hour bins with weights based on HRRR curve:**

| Bin | Max in Bin | Diff from 44°F | Weight | Normalized |
|-----|------------|----------------|--------|------------|
| 00-02 | 23°F | 21 | 0.000 | 0.000 |
| 02-04 | 24°F | 20 | 0.000 | 0.000 |
| 04-06 | 26°F | 18 | 0.000 | 0.000 |
| 06-08 | 32°F | 12 | 0.002 | 0.003 |
| 08-10 | 37°F | 7 | 0.030 | 0.042 |
| 10-12 | 40°F | 4 | 0.135 | 0.188 |
| 12-14 | 43°F | 1 | 0.607 | 0.421 |
| **14-16** | **44°F** | 0 | **1.000** | **0.347** |
| 16-18 | 42°F | 2 | 0.368 | Can't peak (past) |
| ... | ... | ... | ... | ... |

**Expected peak hour:** 15:00 (3:00 PM)

### Step 9: Calculate Confidence

```python
confidence = 1.0  # Start at 100%

# Sigma penalty (2.6 > 2.0): -0.10
confidence -= 0.10

# QC OK: no penalty
confidence -= 0.00

# 1 stale source: -0.05
confidence -= 0.05

# Base forecast valid: no penalty
confidence -= 0.00

# No stable bias bonus (|1.2| > 1.0)
confidence += 0.00

# No post-peak bonus (it's 11 AM)
confidence += 0.00

# Final
confidence = 0.85 (85%)

confidence_factors = [
    "Moderate uncertainty (sigma 2.6°F)",
    "One data source stale"
]
```

---

## Complete Numerical Example

**Scenario:** KLGA on January 29, 2026, at 2:30 PM EST

### Inputs

```
METAR:
  - Latest temp: 42°F at 2:15 PM
  - Max observed today: 43°F at 1:45 PM
  - QC: OK

PWS Cluster:
  - Consensus: 42.5°F
  - Support: 14 stations
  - Drift: +0.5°F

HRRR Forecast:
  - Base max: 45°F
  - Peak hour: 14 (2:00 PM)
  - Current curve position: descending

Bias State:
  - Current bias: -0.8°F (model running warm)
  - Observation count: 8

Health:
  - All sources: OK
```

### Calculation

**Step 1: Bias decay**
```
hours_to_peak = 14 - 14.5 = -0.5 (past peak)
decayed_bias = -0.8 × exp(0.5/2.0) = -0.8 × 1.28 = -1.03°F
```

**Step 2: Initial adjustment**
```
T_adj = 45 + (-1.03) = 43.97°F
```

**Step 3: Floor constraint**
```
max_so_far = 43°F
T_adj = max(43.97, 43) = 43.97°F
```

**Step 4: Post-peak cap (we're past peak)**
```
hours_to_sunset = 17 - 14.5 = 2.5
peak_to_sunset = 17 - 14 = 3
margin = 0.5 + 1.5 × (2.5/3) = 0.5 + 1.25 = 1.75°F
post_peak_cap = 43 + 1.75 = 44.75°F
T_adj = min(43.97, 44.75) = 43.97°F → 44.0°F
```

**Step 5: Sigma calculation**
```
base: 2.0°F
QC OK: +0.0
METAR 15min old: +0.3
No stale sources: +0.0
Post-peak (14:30): ×0.7
8 observations: ×1.0

sigma = (2.0 + 0.3) × 0.7 = 1.61°F
```

**Step 6: Bucket probabilities (mean=44.0, sigma=1.61)**

| Bucket | Probability |
|--------|-------------|
| 41-42 | 0.0312 |
| 42-43 | 0.0892 |
| 43-44 | 0.1876 |
| **44-45** | **0.2840** |
| 45-46 | 0.2432 |
| 46-47 | 0.1156 |
| 47-48 | 0.0392 |

**Step 7: Confidence**
```
base: 1.0
sigma 1.61 (< 2.0): no penalty
QC OK: no penalty
All sources OK: no penalty
8 obs, bias stable: +0.10
Post-peak: +0.10

confidence = 1.0 + 0.10 + 0.10 = 1.0 (clamped)
```

### Final Output

```python
NowcastDistribution(
    tmax_mean_f=44.0,
    tmax_sigma_f=1.61,
    p_bucket=[
        BucketProbability("41-42", 0.0312),
        BucketProbability("42-43", 0.0892),
        BucketProbability("43-44", 0.1876),
        BucketProbability("44-45", 0.2840),  # ← Leading
        BucketProbability("45-46", 0.2432),
        BucketProbability("46-47", 0.1156),
        BucketProbability("47-48", 0.0392),
    ],
    p_ge_strike={
        42: 0.9688, 43: 0.9376, 44: 0.7500, 45: 0.4660,
        46: 0.2228, 47: 0.0836
    },
    t_peak_expected_hour=14,
    confidence=1.0,
    confidence_factors=["Stable bias", "Post-peak observation window"],
    explanations=[
        ("BASE", 45.0, "HRRR forecast max"),
        ("BIAS_DECAY", -1.0, "Bias -0.8°F decayed"),
        ("POST_PEAK_CAP", 0.0, "No cap applied (below cap)"),
        ("FINAL", 44.0, "Judge-aligned prediction")
    ],
    inputs_used=["METAR/KLGA", "PWS_KLGA", "HRRR"]
)
```

---

## Explanation Factors

Each prediction includes a breakdown explaining how the final value was calculated:

| Factor | Description |
|--------|-------------|
| `BASE` | Starting point from HRRR curve |
| `BIAS_DECAY` | Decayed bias adjustment |
| `FLOOR` | Minimum based on observed max |
| `POST_PEAK_CAP` | Maximum based on post-peak physics |
| `FINAL` | Judge-aligned result |

---

## Confidence Interpretation

| Confidence | Interpretation | Typical Scenario |
|------------|----------------|------------------|
| 0.9-1.0 | Very high | Post-peak, good data, stable bias |
| 0.7-0.9 | Good | Normal conditions |
| 0.5-0.7 | Moderate | Some stale data or high uncertainty |
| 0.3-0.5 | Low | Multiple issues |
| < 0.3 | Very low | Significant problems |

---

## Constraint Logic Summary

### Floor Constraint
- **When:** Always active
- **Effect:** Prediction cannot go below observed maximum
- **Why:** Temperature cannot "un-max"

### Post-Peak Cap
- **When:** current_hour >= peak_hour + 1
- **Effect:** Limits upside based on observed max + shrinking margin
- **Why:** After peak, temperature rise becomes increasingly unlikely

### Margin Shrinkage
```
At peak+1h: margin = ~2.0°F (still some upside possible)
At sunset:  margin = 0.5°F  (very little upside)
```

---

## Market Comparison

HELIOS predictions can be compared to Polymarket odds:

```python
# HELIOS says 44-45°F at 28.4%
# Polymarket says 44-45°F at 22% implied probability

difference = 28.4% - 22% = +6.4% (HELIOS more bullish)

if difference > 5%:
    signal = "POTENTIAL_EDGE"
```

See `docs/system/ARCHITECTURE.md` for opportunity detection logic.
