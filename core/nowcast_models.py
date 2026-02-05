"""
Nowcast Engine Models - Phase 3

Core data structures for the Nowcasting Engine v0.
Implements distribution-based temperature prediction with:
- DailyState: Per-market intraday state tracking
- BaseForecast: Stable HRRR-based daily curve
- NowcastDistribution: Output contract with P(bucket), t_peak, confidence
- BiasState: EMA-based bias tracking with temporal decay
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Tuple
from zoneinfo import ZoneInfo
import math

from core.polymarket_labels import format_range_label

# Timezones
NYC = ZoneInfo("America/New_York")
MAD = ZoneInfo("Europe/Madrid")
UTC = ZoneInfo("UTC")


@dataclass
class BiasState:
    """
    Bias tracking with EMA smoothing (Section 3.4).

    bias = EMA(bias, obs - model_now)
    Alpha is dynamic based on QC state.
    """
    current_bias_f: float = 0.0          # Current EMA bias in °F
    last_update_utc: Optional[datetime] = None
    observation_count: int = 0

    # EMA parameters
    alpha_normal: float = 0.3            # Alpha when QC is OK
    alpha_uncertain: float = 0.1         # Alpha when QC is uncertain

    # Decay parameters for propagation
    decay_hours: float = 6.0             # Hours until bias decays to ~0

    def update(self, observed_f: float, model_f: float, qc_ok: bool = True) -> float:
        """
        Update bias with new observation.

        Args:
            observed_f: Actual observed temperature (°F)
            model_f: Model predicted temperature for this time (°F)
            qc_ok: Whether QC passed (affects alpha)

        Returns:
            Updated bias value
        """
        delta = observed_f - model_f
        alpha = self.alpha_normal if qc_ok else self.alpha_uncertain

        if self.observation_count == 0:
            self.current_bias_f = delta
        else:
            self.current_bias_f = alpha * delta + (1 - alpha) * self.current_bias_f

        self.last_update_utc = datetime.now(UTC)
        self.observation_count += 1
        return self.current_bias_f

    def get_decayed_bias(self, hours_ahead: float) -> float:
        """
        Get bias with temporal decay applied.

        T_adj(t) = T_base(t) + bias * decay(t)
        decay(t) falls to 0 over decay_hours.

        Args:
            hours_ahead: Hours from now

        Returns:
            Decayed bias value
        """
        if hours_ahead <= 0:
            return self.current_bias_f

        # Exponential decay: exp(-t/tau) where tau = decay_hours/3
        tau = self.decay_hours / 3.0
        decay_factor = math.exp(-hours_ahead / tau)
        return self.current_bias_f * decay_factor


@dataclass
class TPeakBin:
    """
    Time-of-peak probability bin (Section 3.5).

    Hours are in NYC timezone (America/New_York).
    """
    bin_start_hour: int     # Hour (0-23) start of bin in NYC
    bin_end_hour: int       # Hour (0-23) end of bin in NYC
    probability: float      # P(t_peak in this bin)

    def __post_init__(self):
        # Clamp probability to valid range
        self.probability = max(0.0, min(1.0, self.probability))

    @property
    def label_nyc(self) -> str:
        """NYC time label (primary)."""
        return f"{self.bin_start_hour:02d}:00-{self.bin_end_hour:02d}:00"

    @property
    def label_es(self) -> str:
        """Spanish time label (secondary). NYC + 6 hours typically."""
        # NYC to Madrid offset: +6 hours (EST to CET)
        # Note: This is approximate, DST transitions may differ by a week
        offset = 6
        start_es = (self.bin_start_hour + offset) % 24
        end_es = (self.bin_end_hour + offset) % 24
        return f"{start_es:02d}:00-{end_es:02d}:00"

    @property
    def label(self) -> str:
        """Human-readable bin label: NYC first, then ES in parentheses."""
        return f"{self.label_nyc} NYC ({self.label_es} ES)"

    @property
    def label_short(self) -> str:
        """Short label for histograms - just NYC hours."""
        return f"{self.bin_start_hour:02d}-{self.bin_end_hour:02d}"


@dataclass
class BaseForecast:
    """
    Stable daily forecast from HRRR/ensemble (Section 3.3).

    Updated only when new model run arrives.
    Provides the "slow" layer of the two-layer architecture.
    """
    station_id: str
    model_run_time_utc: datetime         # When the model was run
    ingest_time_utc: datetime            # When we fetched it
    target_date: date                    # Settlement day

    # Hourly temperature curve (24 values, °F)
    t_hourly_f: List[float] = field(default_factory=list)

    # Derived values
    t_max_base_f: float = 0.0            # Max from curve
    t_peak_hour: int = 14                # Hour of max (0-23)

    # Associated features at peak
    radiation_peak: Optional[float] = None
    cloud_cover_peak: Optional[float] = None
    wind_speed_peak: Optional[float] = None
    humidity_peak: Optional[float] = None
    timezone: str = "America/New_York"

    # Validity
    valid_until_utc: Optional[datetime] = None
    model_source: str = "HRRR"           # HRRR, NBM, LAMP, etc.

    def get_temp_at_hour(self, hour: int) -> Optional[float]:
        """Get temperature for a specific hour."""
        if 0 <= hour < len(self.t_hourly_f):
            return self.t_hourly_f[hour]
        return None

    def get_temp_at_time(self, dt: datetime) -> Optional[float]:
        """Get temperature for a specific datetime (interpolated)."""
        # Convert to model timezone for hour extraction
        try:
            tz = ZoneInfo(self.timezone)
        except Exception:
            tz = NYC
        local_dt = dt.astimezone(tz)
        hour = local_dt.hour
        return self.get_temp_at_hour(hour)

    def is_valid(self) -> bool:
        """Check if forecast is still valid."""
        if self.valid_until_utc is None:
            return True
        return datetime.now(UTC) < self.valid_until_utc


@dataclass
class DailyState:
    """
    Per-market daily state (Section 3.2.1).

    Maintains all intra-day tracking for a single market.
    Updated event-driven with each METAR/feature update.
    """
    station_id: str
    target_date: date

    # --- Core tracking ---
    max_so_far_aligned_f: float = -999.0    # Maximum observed (judge-aligned)
    max_so_far_raw_f: float = -999.0        # Maximum observed (raw, decimals)
    max_so_far_time_utc: Optional[datetime] = None

    # Last observation (aligned)
    last_obs_temp_f: float = 0.0            # Raw for trend
    last_obs_aligned_f: float = 0.0         # Judge-aligned
    last_obs_time_utc: Optional[datetime] = None
    last_obs_source: str = ""

    # --- Bias state ---
    bias_state: BiasState = field(default_factory=BiasState)

    # --- Trend state ---
    trend_f_per_hour: float = 0.0           # dT/dt smoothed
    trend_samples: int = 0

    # --- Uncertainty state ---
    sigma_f: float = 2.0                    # Current uncertainty (°F)
    sigma_base: float = 2.0                 # Baseline uncertainty
    sigma_qc_penalty: float = 0.0           # Added uncertainty from QC issues

    # --- t_peak state ---
    t_peak_bins: List[TPeakBin] = field(default_factory=list)

    # --- Market constraints ---
    market_floor_f: Optional[float] = None  # Minimum possible (from observations)
    market_ceiling_f: Optional[float] = None

    # --- Health tracking ---
    sources_health: Dict[str, str] = field(default_factory=dict)  # source -> status
    last_qc_state: str = "OK"               # OK, UNCERTAIN, OUTLIER
    staleness_seconds: float = 0.0

    # --- Timestamps ---
    created_utc: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_update_utc: datetime = field(default_factory=lambda: datetime.now(UTC))

    def update_max(
        self,
        temp_f_aligned: float,
        temp_f_raw: Optional[float],
        obs_time: datetime
    ) -> bool:
        """
        Update max_so_far if new observation is higher.

        Returns:
            True if max was updated
        """
        updated = False
        if temp_f_aligned > self.max_so_far_aligned_f:
            self.max_so_far_aligned_f = temp_f_aligned
            self.max_so_far_time_utc = obs_time
            # Update market floor - can't go below observed max
            self.market_floor_f = temp_f_aligned
            updated = True
        if temp_f_raw is not None and temp_f_raw > self.max_so_far_raw_f:
            self.max_so_far_raw_f = temp_f_raw
            updated = True
        return updated

    def update_observation(
        self,
        temp_f_raw: float,
        temp_f_aligned: float,
        obs_time: datetime,
        source: str
    ):
        """Update with new observation."""
        # Calculate trend if we have previous
        if self.last_obs_time_utc is not None:
            dt_hours = (obs_time - self.last_obs_time_utc).total_seconds() / 3600.0
            if dt_hours > 0.05:  # At least 3 minutes
                new_trend = (temp_f_raw - self.last_obs_temp_f) / dt_hours
                # EMA smoothing on trend
                if self.trend_samples == 0:
                    self.trend_f_per_hour = new_trend
                else:
                    alpha = 0.3
                    self.trend_f_per_hour = alpha * new_trend + (1 - alpha) * self.trend_f_per_hour
                self.trend_samples += 1

        self.last_obs_temp_f = temp_f_raw
        self.last_obs_aligned_f = temp_f_aligned
        self.last_obs_time_utc = obs_time
        self.last_obs_source = source
        self.last_update_utc = datetime.now(UTC)

        # Update max
        self.update_max(temp_f_aligned, temp_f_raw, obs_time)

    def update_uncertainty(self, qc_state: str, source_health: Dict[str, str]):
        """Update uncertainty based on QC and source health."""
        self.last_qc_state = qc_state
        self.sources_health = source_health

        # Base sigma
        self.sigma_f = self.sigma_base

        # Add penalty for QC issues
        if qc_state == "UNCERTAIN":
            self.sigma_qc_penalty = 0.5
        elif qc_state == "OUTLIER":
            self.sigma_qc_penalty = 1.0
        else:
            self.sigma_qc_penalty = 0.0

        # Add penalty for stale sources
        stale_count = sum(1 for s in source_health.values() if s == "STALE")
        self.sigma_f += self.sigma_qc_penalty + (stale_count * 0.3)


@dataclass
class BucketProbability:
    """Probability for a temperature bucket."""
    bucket_low_f: int                       # Lower bound (inclusive)
    bucket_high_f: int                      # Upper bound (exclusive)
    probability: float                      # P(temp in this bucket)
    is_impossible: bool = False             # Below market floor

    @property
    def label(self) -> str:
        """Bucket label in Polymarket format (e.g., '48-49°F')."""
        return format_range_label(self.bucket_low_f, self.bucket_high_f)

    @property
    def midpoint_f(self) -> float:
        """Midpoint of bucket."""
        return (self.bucket_low_f + self.bucket_high_f) / 2.0


@dataclass
class NowcastExplanation:
    """Single explanation factor for the nowcast."""
    factor: str                             # e.g., "bias", "advection", "cloud"
    contribution_f: float                   # How much this factor added/subtracted
    description: str                        # Human-readable explanation
    confidence_impact: float = 0.0          # How much this affects confidence


@dataclass
class NowcastDistribution:
    """
    Output distribution for Phase 3 (Section 3.8).

    The main output contract of the Nowcasting Engine.
    Contains probability distribution over buckets, t_peak bins,
    confidence, and explanations.
    """
    # --- Timestamps ---
    ts_generated_utc: datetime
    ts_nyc: str                             # "2026-01-29 10:30:00"
    ts_es: str                              # "2026-01-29 16:30:00"

    # --- Identifiers ---
    market_id: str                          # Polymarket market ID if known
    station_id: str                         # KLGA, KATL, etc.
    target_date: date

    # --- Central estimate ---
    tmax_mean_f: float                      # Best estimate of Tmax
    tmax_sigma_f: float                     # Uncertainty (std dev)

    # --- Bucket probabilities ---
    p_bucket: List[BucketProbability] = field(default_factory=list)

    # --- Cumulative probabilities ---
    p_ge_strike: Dict[int, float] = field(default_factory=dict)  # strike -> P(T >= strike)

    # --- t_peak distribution ---
    t_peak_bins: List[TPeakBin] = field(default_factory=list)
    t_peak_expected_hour: int = 14          # Most likely hour of peak

    # --- Confidence & quality ---
    confidence: float = 0.5                 # 0-1 confidence score
    confidence_factors: List[str] = field(default_factory=list)

    # --- Explanations ---
    explanations: List[NowcastExplanation] = field(default_factory=list)

    # --- Validity ---
    valid_until_utc: datetime = field(default_factory=lambda: datetime.now(UTC) + timedelta(minutes=5))

    # --- Source tracking ---
    bias_applied_f: float = 0.0
    base_forecast_source: str = "HRRR"
    inputs_used: List[str] = field(default_factory=list)

    def get_bucket_probability(self, bucket_label: str) -> Optional[float]:
        """Get probability for a specific bucket."""
        for b in self.p_bucket:
            if b.label == bucket_label:
                return b.probability
        return None

    def get_top_buckets(self, n: int = 3) -> List[BucketProbability]:
        """Get top N buckets by probability."""
        sorted_buckets = sorted(self.p_bucket, key=lambda b: b.probability, reverse=True)
        return sorted_buckets[:n]

    def to_dict(self) -> Dict:
        """Serialize to dictionary for JSON/SSE."""
        return {
            "ts_generated_utc": self.ts_generated_utc.isoformat(),
            "ts_nyc": self.ts_nyc,
            "ts_es": self.ts_es,
            "market_id": self.market_id,
            "station_id": self.station_id,
            "target_date": self.target_date.isoformat(),
            "tmax_mean_f": round(self.tmax_mean_f, 1),
            "tmax_sigma_f": round(self.tmax_sigma_f, 2),
            "p_bucket": [
                {
                    "label": b.label,
                    "probability": round(b.probability, 4),
                    "is_impossible": b.is_impossible
                }
                for b in self.p_bucket
            ],
            "p_ge_strike": {str(k): round(v, 4) for k, v in self.p_ge_strike.items()},
            "t_peak_bins": [
                {
                    "label": b.label,
                    "label_nyc": b.label_nyc,
                    "label_es": b.label_es,
                    "label_short": b.label_short,
                    "bin_start_hour_nyc": b.bin_start_hour,
                    "bin_end_hour_nyc": b.bin_end_hour,
                    "probability": round(b.probability, 4)
                }
                for b in self.t_peak_bins
            ],
            "t_peak_expected_hour": self.t_peak_expected_hour,
            "confidence": round(self.confidence, 3),
            "confidence_factors": self.confidence_factors,
            "explanations": [
                {
                    "factor": e.factor,
                    "contribution_f": round(e.contribution_f, 2),
                    "description": e.description
                }
                for e in self.explanations
            ],
            "valid_until_utc": self.valid_until_utc.isoformat(),
            "bias_applied_f": round(self.bias_applied_f, 2),
            "base_forecast_source": self.base_forecast_source,
            "inputs_used": self.inputs_used
        }


# --- Utility functions ---

def create_hourly_t_peak_bins() -> List[TPeakBin]:
    """Create 24 hourly bins for t_peak distribution."""
    return [
        TPeakBin(bin_start_hour=h, bin_end_hour=h+1, probability=0.0)
        for h in range(24)
    ]


def create_2hour_t_peak_bins() -> List[TPeakBin]:
    """Create 12 two-hour bins for t_peak distribution."""
    return [
        TPeakBin(bin_start_hour=h, bin_end_hour=h+2, probability=0.0)
        for h in range(0, 24, 2)
    ]


def create_bucket_range(center_f: float, cone_size: int = 2) -> List[BucketProbability]:
    """
    Create bucket probabilities around a center value.

    Polymarket temperature brackets are 2°F wide and aligned to even lows,
    e.g. 30-31°F, 32-33°F, 34-35°F.

    Args:
        center_f: Central temperature estimate
        cone_size: Number of buckets on each side of center

    Returns:
        List of BucketProbability objects
    """
    center_bucket = int(round(center_f))
    # Align to even low so buckets match Polymarket bracket scheme
    if center_bucket % 2 == 1:
        center_bucket -= 1

    buckets = []

    for offset in range(-cone_size, cone_size + 1):
        low = center_bucket + (offset * 2)
        high = low + 1
        buckets.append(BucketProbability(
            bucket_low_f=low,
            bucket_high_f=high,
            probability=0.0
        ))

    return buckets
