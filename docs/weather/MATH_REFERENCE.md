# HELIOS Mathematics — Technical Reference

This document contains all mathematical formulas used by HELIOS for temperature prediction and probability calculation.

## 1. Temperature Prediction (Tmax Adjustment)

### Core Formula

```
T_adj(t) = T_base_max + bias × decay(hours_to_peak)
```

**Where:**
- `T_adj(t)` = Adjusted maximum temperature prediction
- `T_base_max` = Maximum from HRRR hourly curve (Layer 1)
- `bias` = Current EMA-tracked bias
- `decay(h)` = Exponential decay function

### Exponential Decay

```
decay(h) = exp(-h / τ)

Where:
  h = hours until expected peak
  τ = decay_hours / 3.0 = 6.0 / 3.0 = 2.0 hours
```

**Decay Values by Time to Peak:**
| Hours to Peak | Decay Factor | Effect |
|---------------|--------------|--------|
| 0 | 1.000 | Full bias applied |
| 1 | 0.607 | ~61% of bias |
| 2 | 0.368 | ~37% of bias |
| 3 | 0.223 | ~22% of bias |
| 6 | 0.050 | ~5% of bias |

**Implementation:** `core/nowcast_models.py:67-86`

---

## 2. Bias Correction (EMA Update)

### Formula

```
bias_new = α × (T_observed - T_model) + (1 - α) × bias_old
```

**Where:**
- `α` = Learning rate (alpha)
  - `α = 0.3` when QC = OK
  - `α = 0.1` when QC = UNCERTAIN
- `T_observed` = Judge-aligned METAR temperature
- `T_model` = Base forecast temperature at observation time

### Special Cases

**First observation:**
```
bias = T_observed - T_model  (direct assignment)
```

**Example:**
```
Given:
  T_observed = 34°F (METAR at 10:00)
  T_model = 32°F (HRRR at same hour)
  bias_old = 1.5°F
  α = 0.3 (QC OK)

Then:
  delta = 34 - 32 = 2°F
  bias_new = 0.3 × 2 + 0.7 × 1.5 = 0.6 + 1.05 = 1.65°F
```

**Implementation:** `core/nowcast_models.py:43-65`

---

## 3. Uncertainty Estimation (Sigma Calculation)

### Base Formula

```
σ = σ_base × time_factor × obs_factor + penalties
```

### Component Breakdown

**Base sigma:**
```
σ_base = 2.0°F
```

**QC penalties:**
```
if QC = "UNCERTAIN":  σ += 0.5°F
if QC = "OUTLIER":    σ += 1.0°F
```

**METAR staleness penalties:**
```
if metar_age > 3600s (1 hour):   σ += 1.0°F
elif metar_age > 1800s (30 min): σ += 0.7°F
elif metar_age > 900s (15 min):  σ += 0.3°F
```

**Stale sources penalty:**
```
stale_count = count(source_status ∈ {STALE, DEAD, NO_DATA})
σ += stale_count × 0.3°F
```

**Time-of-day multiplier:**
```
if current_hour >= 14:  σ *= 0.7   (post-peak: more certain)
elif current_hour >= 12: σ *= 0.85  (near peak)
else:                    σ *= 1.0   (morning: full uncertainty)
```

**Observation count benefit:**
```
if observation_count > 10: σ *= 0.9
```

**Floor constraint:**
```
σ = max(0.5°F, σ)
```

### Complete Example

```
Morning scenario (8:00 AM):
  Base:        2.0°F
  QC OK:       +0.0°F
  METAR 20min: +0.3°F
  No stale:    +0.0°F
  Time mult:   ×1.0
  3 obs:       ×1.0
  Result:      2.3°F

Post-peak scenario (3:00 PM):
  Base:        2.0°F
  QC OK:       +0.0°F
  METAR 5min:  +0.0°F
  No stale:    +0.0°F
  Time mult:   ×0.7
  12 obs:      ×0.9
  Result:      2.0 × 0.7 × 0.9 = 1.26°F
```

**Implementation:** `core/nowcast_engine.py:451-510`

---

## 4. Probability Distribution

### Distribution Type

HELIOS uses a **Logistic distribution** by default (configurable to Normal).

### Logistic CDF

```
CDF(x) = 1 / (1 + exp(-(x - μ) / s))

Where:
  μ = tmax_mean_f (mean prediction)
  s = σ × 0.5513  (scale parameter)
  0.5513 ≈ √3/π (converts σ to logistic scale)
```

**Why Logistic?**
- Heavier tails than Normal distribution
- Better handles extreme temperature events
- More robust to outliers

### Normal CDF (Alternative)

```
CDF(x) = 0.5 × (1 + erf((x - μ) / (σ × √2)))

Where:
  erf = error function
```

**Implementation:** `core/nowcast_engine.py:569-579`

---

## 5. Bucket Probability Calculation

### Formula

```
P(bucket) = CDF(high) - CDF(low)
```

**Where:**
- `high` = bucket upper bound (exclusive)
- `low` = bucket lower bound (inclusive)

### Bucket Creation

```
center_bucket = round(tmax_mean_f)
cone_size = 3  (configurable)

buckets = [center - 3, center - 2, center - 1, center,
           center + 1, center + 2, center + 3]
```

**Example with mean=50.5°F, σ=2.0°F:**

| Bucket | Low | High | CDF(high) | CDF(low) | P(bucket) |
|--------|-----|------|-----------|----------|-----------|
| 47-48 | 47 | 48 | 0.0893 | 0.0519 | 0.0374 |
| 48-49 | 48 | 49 | 0.1457 | 0.0893 | 0.0564 |
| 49-50 | 49 | 50 | 0.2296 | 0.1457 | 0.0839 |
| **50-51** | 50 | 51 | 0.3543 | 0.2296 | **0.1247** |
| 51-52 | 51 | 52 | 0.5000 | 0.3543 | 0.1457 |
| 52-53 | 52 | 53 | 0.6457 | 0.5000 | 0.1457 |
| 53-54 | 53 | 54 | 0.7704 | 0.6457 | 0.1247 |

### Normalization

After calculating raw probabilities, normalize so sum = 1.0:

```
total = Σ P(bucket)
P_normalized(bucket) = P(bucket) / total
```

### Market Floor Constraint

Buckets below observed maximum are marked impossible:

```
if bucket.high <= max_so_far_aligned_f:
    bucket.is_impossible = True
    bucket.probability = 0.0
```

**Implementation:** `core/nowcast_engine.py:512-550`

---

## 6. Cumulative Probabilities (P ≥ Strike)

### Formula

```
P(T ≥ strike) = 1 - CDF(strike)
```

### Example

For mean=50°F, σ=2°F:

| Strike | CDF(strike) | P(T ≥ strike) |
|--------|-------------|---------------|
| 46°F | 0.0228 | 0.9772 |
| 48°F | 0.1587 | 0.8413 |
| 50°F | 0.5000 | 0.5000 |
| 52°F | 0.8413 | 0.1587 |
| 54°F | 0.9772 | 0.0228 |

**Implementation:** `core/nowcast_engine.py:581-600`

---

## 7. Post-Peak Cap

### When Applied

Triggers when `current_hour >= peak_hour + 1` (1+ hour after expected peak).

### Formula

```
margin = 0.5 + 1.5 × (hours_to_sunset / peak_to_sunset_span)

post_peak_cap = max_so_far_aligned_f + margin

T_adj = min(T_adj, post_peak_cap)
```

**Where:**
- `sunset_hour = 17` (5:00 PM, configurable)
- `hours_to_sunset = max(0, sunset - current_hour)`
- `peak_to_sunset_span = sunset - peak_hour`

### Margin Values by Time

For peak_hour=14, sunset=17:

| Current Hour | Hours to Sunset | Span | Margin |
|--------------|-----------------|------|--------|
| 15:00 | 2 | 3 | 0.5 + 1.5×(2/3) = 1.5°F |
| 16:00 | 1 | 3 | 0.5 + 1.5×(1/3) = 1.0°F |
| 17:00 | 0 | 3 | 0.5 + 1.5×(0/3) = 0.5°F |

**Intuition:** As sunset approaches, upside potential decreases.

**Implementation:** `core/nowcast_engine.py:417-439`

---

## 8. Confidence Score Calculation

### Base

```
confidence = 1.0  (start at 100%)
```

### Penalties

```
Sigma penalty:
  if σ > 3.0°F: confidence -= 0.30
  elif σ > 2.0°F: confidence -= 0.10

QC penalty:
  if QC = "UNCERTAIN": confidence -= 0.15
  if QC = "OUTLIER": confidence -= 0.30

Staleness penalty:
  confidence -= 0.05 × stale_source_count

Base forecast penalty:
  if base_forecast invalid: confidence -= 0.20
```

### Bonuses

```
Stable bias bonus:
  if obs_count >= 5 AND |bias| < 1.0°F:
    confidence += 0.10

Post-peak bonus:
  if current_hour >= 14:
    confidence += 0.10
```

### Clamp

```
confidence = max(0.1, min(1.0, confidence))
```

**Implementation:** `core/nowcast_engine.py:697-753`

---

## 9. T-Peak Distribution (Time of Maximum)

### Bin Structure

12 two-hour bins: 00-02, 02-04, ..., 22-24

### Weight Calculation

```
For each bin:
  if bin.end <= current_hour:
    weight = 0  (can't peak in past)
  else:
    bin_max = max(HRRR temps in this bin)
    diff = daily_max - bin_max
    weight = exp(-diff / 2.0)
```

### Trend Adjustment

```
if trend_f_per_hour > 1.0 (rising fast):
  bins after current+2: weight *= 1.3

if trend_f_per_hour < -0.5 (falling):
  bins before current+1: weight *= 1.5
```

### Normalize

```
total = Σ weights
probability(bin) = weight / total
```

**Implementation:** `core/nowcast_engine.py:606-682`

---

## 10. Quick Reference Card

| Formula | Expression | Parameters |
|---------|------------|------------|
| **Tmax Adjustment** | T_base + bias × e^(-h/τ) | τ=2.0h |
| **Bias EMA** | α×δ + (1-α)×bias_old | α=0.3 or 0.1 |
| **Logistic CDF** | 1/(1+e^(-(x-μ)/s)) | s=σ×0.5513 |
| **Bucket Prob** | CDF(high) - CDF(low) | cone_size=3 |
| **Post-Peak Margin** | 0.5 + 1.5×(h_sunset/span) | sunset=17h |
| **Base Sigma** | 2.0°F | min=0.5°F |

---

## 11. Configuration Parameters

From `NowcastConfig` class:

```python
# Bias parameters
bias_decay_hours: float = 6.0        # τ = 2.0 hours
bias_alpha_normal: float = 0.3       # QC OK
bias_alpha_uncertain: float = 0.1    # QC uncertain

# Sigma parameters
base_sigma_f: float = 2.0
sigma_qc_penalty_uncertain: float = 0.5
sigma_qc_penalty_outlier: float = 1.0
sigma_stale_source_penalty: float = 0.3

# Distribution parameters
cone_size: int = 3                   # ±3 buckets
distribution_type: str = "logistic"  # or "normal"

# Physical bounds
min_temp_f: float = -40.0
max_temp_f: float = 130.0
```

**Implementation:** `core/nowcast_engine.py:37-63`
