"""
Nowcast Engine - Phase 3 (Section 3.2)

Two-layer architecture:
1. Base Forecast Layer (slow, stable - from HRRR/ensemble)
2. Nowcast Adjustment Layer (fast, reacts to observations)

Produces probability distributions over temperature buckets.
"""

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Tuple
from zoneinfo import ZoneInfo

from core.nowcast_models import (
    DailyState, BaseForecast, NowcastDistribution,
    BiasState, TPeakBin, BucketProbability, NowcastExplanation,
    create_2hour_t_peak_bins, create_bucket_range
)
from core.judge import JudgeAlignment
from core.models import OfficialObs, AuxObs, EnvFeature
try:
    from config import STATIONS
except Exception:
    STATIONS = {}

# Timezones
NYC = ZoneInfo("America/New_York")
MAD = ZoneInfo("Europe/Madrid")
UTC = ZoneInfo("UTC")


@dataclass
class NowcastConfig:
    """Configuration for the Nowcast Engine."""
    # Bias decay
    bias_decay_hours: float = 6.0
    bias_alpha_normal: float = 0.3
    bias_alpha_uncertain: float = 0.1

    # Uncertainty
    base_sigma_f: float = 2.0
    sigma_qc_penalty_uncertain: float = 0.5
    sigma_qc_penalty_outlier: float = 1.0
    sigma_stale_source_penalty: float = 0.3

    # Distribution
    cone_size: int = 3  # Buckets on each side of center
    distribution_type: str = "logistic"  # "normal" or "logistic"

    # t_peak
    t_peak_bin_hours: int = 2  # 2-hour bins

    # Update intervals
    periodic_interval_seconds: float = 60.0

    # Physical limits
    min_temp_f: float = -40.0
    max_temp_f: float = 130.0


class NowcastEngine:
    """
    Main Nowcasting Engine.

    Maintains per-market DailyState and produces NowcastDistribution
    on demand or when triggered by events.
    """

    def __init__(self, station_id: str, config: Optional[NowcastConfig] = None):
        """
        Initialize engine for a station.

        Args:
            station_id: ICAO station code (e.g., "KLGA")
            config: Optional configuration override
        """
        self.station_id = station_id.upper()
        self.config = config or NowcastConfig()
        self.station_tz = self._resolve_station_tz()
        self.station_tz_name = getattr(self.station_tz, "key", "America/New_York")

        # State management
        self._daily_states: Dict[date, DailyState] = {}
        self._base_forecast: Optional[BaseForecast] = None
        self._last_distribution: Optional[NowcastDistribution] = None

        # Tracking
        self._last_update_utc: Optional[datetime] = None
        self._update_count: int = 0

    def _resolve_station_tz(self) -> ZoneInfo:
        """Resolve station timezone from config, default to NYC."""
        try:
            station = STATIONS.get(self.station_id)
            if station and station.timezone:
                return ZoneInfo(station.timezone)
        except Exception:
            pass
        return NYC

    def _get_target_date(self, dt_utc: datetime) -> date:
        """
        Get target date for daily state.
        Defaults to NYC settlement day for US stations.
        """
        try:
            if self.station_tz_name != "America/New_York":
                return dt_utc.astimezone(self.station_tz).date()
        except Exception:
            pass
        return JudgeAlignment.settlement_date(dt_utc)

    # =========================================================================
    # State Management
    # =========================================================================

    def get_or_create_daily_state(self, target_date: Optional[date] = None) -> DailyState:
        """Get or create DailyState for a date."""
        if target_date is None:
            target_date = self._get_target_date(datetime.now(UTC))

        if target_date not in self._daily_states:
            self._daily_states[target_date] = DailyState(
                station_id=self.station_id,
                target_date=target_date,
                sigma_base=self.config.base_sigma_f
            )
            # Propagate config to bias state
            self._daily_states[target_date].bias_state.alpha_normal = self.config.bias_alpha_normal
            self._daily_states[target_date].bias_state.alpha_uncertain = self.config.bias_alpha_uncertain
            self._daily_states[target_date].bias_state.decay_hours = self.config.bias_decay_hours

        state = self._daily_states[target_date]
        state.bias_state.alpha_normal = self.config.bias_alpha_normal
        state.bias_state.alpha_uncertain = self.config.bias_alpha_uncertain
        state.bias_state.decay_hours = self.config.bias_decay_hours
        return state

    def get_base_forecast(self) -> Optional[BaseForecast]:
        """Get current base forecast."""
        return self._base_forecast

    # =========================================================================
    # Layer 1: Base Forecast (Slow)
    # =========================================================================

    def update_base_forecast(
        self,
        hourly_temps_f: List[float],
        model_run_time: datetime,
        model_source: str = "HRRR",
        target_date: Optional[date] = None,
        additional_features: Optional[Dict] = None
    ) -> BaseForecast:
        """
        Update the Base Forecast layer.

        Called when new HRRR/ensemble data arrives.
        This is the "slow" layer that provides stability.

        Args:
            hourly_temps_f: 24 hourly temperatures in °F
            model_run_time: When the model was run
            model_source: Model name (HRRR, NBM, etc.)
            target_date: Settlement day (default: today NYC)
            additional_features: Dict with radiation, cloud, wind, humidity at peak

        Returns:
            Updated BaseForecast
        """
        if target_date is None:
            target_date = self._get_target_date(datetime.now(UTC))

        if not hourly_temps_f:
            # Keep existing forecast if new data is empty
            return self._base_forecast

        # Find max and peak hour
        t_max_f = max(hourly_temps_f) if hourly_temps_f else 0.0
        t_peak_hour = hourly_temps_f.index(t_max_f) if hourly_temps_f else 14

        # Extract features at peak hour
        features = additional_features or {}

        self._base_forecast = BaseForecast(
            station_id=self.station_id,
            model_run_time_utc=model_run_time,
            ingest_time_utc=datetime.now(UTC),
            target_date=target_date,
            t_hourly_f=hourly_temps_f,
            t_max_base_f=t_max_f,
            t_peak_hour=t_peak_hour,
            radiation_peak=features.get("radiation"),
            cloud_cover_peak=features.get("cloud_cover"),
            wind_speed_peak=features.get("wind_speed"),
            humidity_peak=features.get("humidity"),
            valid_until_utc=datetime.now(UTC) + timedelta(hours=6),
            model_source=model_source,
            timezone=self.station_tz_name
        )

        return self._base_forecast

    # =========================================================================
    # Layer 2: Nowcast Adjustment (Fast)
    # =========================================================================

    def update_observation(
        self,
        obs: OfficialObs,
        qc_state: str = "OK"
    ) -> NowcastDistribution:
        """
        Update with new METAR observation.

        This triggers the "fast" nowcast adjustment layer.

        Args:
            obs: Official observation
            qc_state: QC result (OK, UNCERTAIN, OUTLIER)

        Returns:
            Updated NowcastDistribution
        """
        state = self.get_or_create_daily_state(target_date=self._get_target_date(obs.obs_time_utc))

        # Update daily state
        obs_temp_raw_f = obs.temp_f_raw if obs.temp_f_raw else obs.temp_f
        obs_temp_aligned_f = obs.temp_f if obs.temp_f is not None else obs_temp_raw_f
        state.update_observation(
            temp_f_raw=obs_temp_raw_f,
            temp_f_aligned=obs_temp_aligned_f,
            obs_time=obs.obs_time_utc,
            source=obs.source
        )
        state.last_qc_state = qc_state
        state.sources_health["METAR"] = "OK"

        # Update bias if we have base forecast
        if self._base_forecast and self._base_forecast.is_valid() and self._base_forecast.target_date == state.target_date:
            model_temp = self._base_forecast.get_temp_at_time(obs.obs_time_utc)
            if model_temp is not None:
                state.bias_state.update(
                    observed_f=obs_temp_raw_f,
                    model_f=model_temp,
                    qc_ok=(qc_state == "OK")
                )

        # Update uncertainty
        state.update_uncertainty(qc_state, state.sources_health)

        # Generate new distribution
        return self.generate_distribution()

    def update_aux_observation(
        self,
        aux: AuxObs
    ) -> None:
        """
        Update with auxiliary observation (PWS consensus).

        Does not trigger full distribution regeneration,
        but updates health tracking.
        """
        state = self.get_or_create_daily_state(target_date=self._get_target_date(aux.obs_time_utc))
        state.sources_health[f"AUX_{aux.station_id}"] = "OK"

    def update_env_feature(
        self,
        feature: EnvFeature
    ) -> None:
        """
        Update with environment feature (SST, AOD, etc.).

        Adjusts uncertainty based on feature status.
        """
        state = self.get_or_create_daily_state(target_date=self._get_target_date(feature.obs_time_utc))

        feature_key = f"{feature.feature_type}_{feature.location_ref}"
        state.sources_health[feature_key] = feature.status

    # =========================================================================
    # Distribution Calculation
    # =========================================================================

    def generate_distribution(self) -> NowcastDistribution:
        """
        Generate the full NowcastDistribution.

        Main output of the engine.
        """
        now_utc = datetime.now(UTC)
        now_nyc = now_utc.astimezone(NYC)
        now_es = now_utc.astimezone(MAD)

        state = self.get_or_create_daily_state()

        # Calculate adjusted Tmax mean with breakdown
        tmax_mean_f, tmax_breakdown = self._calculate_adjusted_tmax(state)

        # Calculate uncertainty (sigma)
        sigma_f = self._calculate_sigma(state)

        # Generate bucket probabilities
        p_bucket = self._calculate_bucket_probabilities(tmax_mean_f, sigma_f, state)

        # Generate cumulative probabilities P(T >= strike)
        p_ge_strike = self._calculate_cumulative_probabilities(tmax_mean_f, sigma_f)

        # Generate t_peak distribution
        t_peak_bins = self._calculate_t_peak_distribution(state)
        t_peak_expected = self._get_expected_peak_hour(t_peak_bins)

        # Calculate confidence
        confidence, factors = self._calculate_confidence(state, sigma_f)

        # Build explanations
        explanations = self._build_explanations(state)

        # Inputs used
        inputs_used = self._get_inputs_used(state)

        distribution = NowcastDistribution(
            ts_generated_utc=now_utc,
            ts_nyc=now_nyc.strftime("%Y-%m-%d %H:%M:%S"),
            ts_es=now_es.strftime("%Y-%m-%d %H:%M:%S"),
            market_id="",  # Set externally if known
            station_id=self.station_id,
            target_date=state.target_date,
            tmax_mean_f=round(tmax_mean_f, 1),
            tmax_sigma_f=round(sigma_f, 2),
            p_bucket=p_bucket,
            p_ge_strike=p_ge_strike,
            t_peak_bins=t_peak_bins,
            t_peak_expected_hour=t_peak_expected,
            confidence=confidence,
            confidence_factors=factors,
            explanations=explanations,
            valid_until_utc=now_utc + timedelta(minutes=5),
            bias_applied_f=round(state.bias_state.current_bias_f, 2),
            base_forecast_source=self._base_forecast.model_source if self._base_forecast else "N/A",
            inputs_used=inputs_used
        )

        self._last_distribution = distribution
        self._last_update_utc = now_utc
        self._update_count += 1

        return distribution

    def _calculate_adjusted_tmax(self, state: DailyState) -> tuple:
        """
        Calculate adjusted Tmax estimate with full breakdown.

        T_adj = T_base_max + bias * decay(hours_to_peak)

        Returns:
            tuple: (final_value, breakdown_dict)
        """
        breakdown = {
            "base_max_f": None,
            "peak_hour": None,
            "current_hour": None,
            "hours_to_peak": None,
            "decayed_bias_f": None,
            "after_bias_f": None,
            "floor_applied": False,
            "floor_value_f": None,
            "post_peak_cap_applied": False,
            "post_peak_cap_f": None,
            "post_peak_margin_f": None,
            "final_f": None,
        }

        # Start with base forecast max
        use_base = False
        if self._base_forecast and self._base_forecast.is_valid():
            if self._base_forecast.target_date == state.target_date:
                use_base = True
        if use_base:
            base_max = self._base_forecast.t_max_base_f
            peak_hour = self._base_forecast.t_peak_hour
        else:
            # Fallback: use last observation + typical afternoon rise
            base_max = state.last_obs_temp_f + 5.0 if state.last_obs_temp_f else 50.0
            peak_hour = 14

        breakdown["base_max_f"] = round(base_max, 1)
        breakdown["peak_hour"] = peak_hour

        # Calculate hours to peak
        now_local = datetime.now(UTC).astimezone(self.station_tz)
        current_hour = now_local.hour + now_local.minute / 60.0
        hours_to_peak = max(0, peak_hour - current_hour)

        breakdown["current_hour"] = round(current_hour, 2)
        breakdown["hours_to_peak"] = round(hours_to_peak, 2)

        # Apply decayed bias
        decayed_bias = state.bias_state.get_decayed_bias(hours_to_peak)
        adjusted = base_max + decayed_bias

        breakdown["decayed_bias_f"] = round(decayed_bias, 2)
        breakdown["after_bias_f"] = round(adjusted, 1)

        # Apply floor constraint: can't go below observed max
        if state.max_so_far_aligned_f > -900:
            if adjusted < state.max_so_far_aligned_f:
                breakdown["floor_applied"] = True
                breakdown["floor_value_f"] = round(state.max_so_far_aligned_f, 1)
            adjusted = max(adjusted, state.max_so_far_aligned_f)

        # =========================================================================
        # v13.2 Post-Peak Cap (FIXED TRIGGER)
        # Only apply cap 1+ hour AFTER the expected peak hour, not at the peak itself.
        # Margin decreases from 2.0°F at peak+1 to 0.5°F by sunset (17:00).
        # Rationale: Need time for actual peak to be reached before capping upside.
        # =========================================================================
        if state.max_so_far_aligned_f > -900 and current_hour >= peak_hour + 1:
            sunset_hour = 17
            hours_to_sunset = max(0, sunset_hour - current_hour)
            peak_to_sunset_span = max(1, sunset_hour - peak_hour)

            if hours_to_sunset > 0:
                margin_f = 0.5 + 1.5 * (hours_to_sunset / peak_to_sunset_span)
            else:
                margin_f = 0.5  # At or past sunset

            post_peak_cap = state.max_so_far_aligned_f + margin_f

            if adjusted > post_peak_cap:
                breakdown["post_peak_cap_applied"] = True
                breakdown["post_peak_cap_f"] = round(post_peak_cap, 1)
                breakdown["post_peak_margin_f"] = round(margin_f, 2)
                adjusted = post_peak_cap

        # Physical clamps
        adjusted = max(self.config.min_temp_f, min(self.config.max_temp_f, adjusted))

        breakdown["final_f"] = round(adjusted, 1)

        # Store breakdown in state for access by explanations
        state._last_tmax_breakdown = breakdown

        return adjusted, breakdown

    def _calculate_sigma(self, state: DailyState) -> float:
        """
        Calculate uncertainty (sigma) dynamically.

        Sigma increases with:
        - QC issues
        - Stale sources
        - High model divergence

        Sigma decreases with:
        - Consistent observations
        - More data points
        """
        sigma = self.config.base_sigma_f

        # QC penalty
        if state.last_qc_state == "UNCERTAIN":
            sigma += self.config.sigma_qc_penalty_uncertain
        elif state.last_qc_state == "OUTLIER":
            sigma += self.config.sigma_qc_penalty_outlier

        # METAR staleness penalty
        metar_age_s = None
        if state.last_obs_time_utc:
            metar_age_s = (datetime.now(UTC) - state.last_obs_time_utc).total_seconds()
            state.staleness_seconds = metar_age_s
            if metar_age_s > 3600:
                sigma += 1.0
                state.sources_health["METAR"] = "DEAD"
            elif metar_age_s > 1800:
                sigma += 0.7
                state.sources_health["METAR"] = "STALE"
            elif metar_age_s > 900:
                sigma += 0.3
                state.sources_health["METAR"] = "STALE"
            else:
                state.sources_health["METAR"] = "OK"
        else:
            sigma += 1.0
            state.sources_health["METAR"] = "NO_DATA"

        # Stale sources penalty
        stale_count = sum(1 for s in state.sources_health.values() if s in ("STALE", "DEAD", "NO_DATA"))
        sigma += stale_count * self.config.sigma_stale_source_penalty

        # Time-based adjustment: sigma decreases as we approach peak
        now_local = datetime.now(UTC).astimezone(self.station_tz)
        current_hour = now_local.hour

        if current_hour >= 14:
            # After typical peak, reduce sigma
            sigma *= 0.7
        elif current_hour >= 12:
            sigma *= 0.85

        # Observation count benefit
        if state.bias_state.observation_count > 10:
            sigma *= 0.9

        return max(0.5, sigma)  # Minimum sigma

    def _calculate_bucket_probabilities(
        self,
        mean_f: float,
        sigma_f: float,
        state: DailyState
    ) -> List[BucketProbability]:
        """
        Calculate probability for each temperature bucket.

        Uses truncated normal or logistic distribution.
        """
        buckets = create_bucket_range(mean_f, self.config.cone_size)

        # Calculate probability mass for each bucket
        total_prob = 0.0
        for bucket in buckets:
            prob = self._integrate_distribution(
                bucket.bucket_low_f,
                bucket.bucket_high_f,
                mean_f,
                sigma_f
            )
            bucket.probability = prob

            # Mark impossible buckets (below observed max)
            if state.max_so_far_aligned_f > -900:
                if bucket.bucket_high_f <= state.max_so_far_aligned_f:
                    bucket.is_impossible = True
                    bucket.probability = 0.0

            total_prob += bucket.probability

        # Normalize probabilities to sum to 1
        if total_prob > 0:
            for bucket in buckets:
                if not bucket.is_impossible:
                    bucket.probability /= total_prob

        return buckets

    def _integrate_distribution(
        self,
        low: float,
        high: float,
        mean: float,
        sigma: float
    ) -> float:
        """
        Integrate probability density between low and high.

        Uses CDF of logistic or normal distribution.
        """
        if self.config.distribution_type == "logistic":
            return self._logistic_cdf(high, mean, sigma) - self._logistic_cdf(low, mean, sigma)
        else:
            return self._normal_cdf(high, mean, sigma) - self._normal_cdf(low, mean, sigma)

    def _logistic_cdf(self, x: float, mean: float, sigma: float) -> float:
        """Logistic CDF (slightly heavier tails than normal)."""
        # Scale parameter s ≈ sigma * sqrt(3) / pi for comparable spread
        s = sigma * 0.5513
        z = (x - mean) / s
        return 1 / (1 + math.exp(-z))

    def _normal_cdf(self, x: float, mean: float, sigma: float) -> float:
        """Standard normal CDF using error function."""
        z = (x - mean) / (sigma * math.sqrt(2))
        return 0.5 * (1 + math.erf(z))

    def _calculate_cumulative_probabilities(
        self,
        mean_f: float,
        sigma_f: float
    ) -> Dict[int, float]:
        """
        Calculate P(T >= strike) for various strikes.
        """
        center = int(round(mean_f))
        strikes = range(center - 5, center + 6)

        p_ge = {}
        for strike in strikes:
            if self.config.distribution_type == "logistic":
                prob = 1 - self._logistic_cdf(strike, mean_f, sigma_f)
            else:
                prob = 1 - self._normal_cdf(strike, mean_f, sigma_f)
            p_ge[strike] = round(prob, 4)

        return p_ge

    # =========================================================================
    # t_peak Distribution (Section 3.5)
    # =========================================================================

    def _calculate_t_peak_distribution(self, state: DailyState) -> List[TPeakBin]:
        """
        Calculate probability distribution for time of peak.

        Based on:
        - Base forecast temperature curve
        - Current temperature trend
        - Advection (if warm air arriving late)
        - Cloud clearing patterns
        """
        bins = create_2hour_t_peak_bins()
        now_local = datetime.now(UTC).astimezone(self.station_tz)
        current_hour = now_local.hour

        # Base weights from temperature curve
        if (
            self._base_forecast
            and self._base_forecast.is_valid()
            and self._base_forecast.target_date == state.target_date
            and self._base_forecast.t_hourly_f
        ):
            temps = self._base_forecast.t_hourly_f
            max_temp = max(temps)

            # Assign weight based on how close each hour is to max
            for bin in bins:
                bin_mid_hour = (bin.bin_start_hour + bin.bin_end_hour) / 2

                # Can't have peak in the past
                if bin.bin_end_hour <= current_hour:
                    bin.probability = 0.0
                    continue

                # Get temps in this bin's range
                bin_temps = []
                for h in range(bin.bin_start_hour, min(bin.bin_end_hour, len(temps))):
                    if h < len(temps):
                        bin_temps.append(temps[h])

                if bin_temps:
                    bin_max = max(bin_temps)
                    # Weight based on proximity to daily max
                    diff = max_temp - bin_max
                    bin.probability = math.exp(-diff / 2.0)  # Exponential falloff
                else:
                    bin.probability = 0.0
        else:
            # Default: typical afternoon peak
            for bin in bins:
                if bin.bin_end_hour <= current_hour:
                    bin.probability = 0.0
                elif 12 <= bin.bin_start_hour <= 16:
                    bin.probability = 0.3
                elif 10 <= bin.bin_start_hour < 12 or 16 < bin.bin_start_hour <= 18:
                    bin.probability = 0.1
                else:
                    bin.probability = 0.05

        # Adjust based on trend
        if state.trend_f_per_hour > 1.0:
            # Temperature rising fast - peak likely later
            for bin in bins:
                if bin.bin_start_hour >= current_hour + 2:
                    bin.probability *= 1.3
        elif state.trend_f_per_hour < -0.5:
            # Temperature falling - peak likely past or soon
            for bin in bins:
                if bin.bin_start_hour <= current_hour + 1:
                    bin.probability *= 1.5

        # Normalize
        total = sum(b.probability for b in bins)
        if total > 0:
            for bin in bins:
                bin.probability = round(bin.probability / total, 4)

        return bins

    def _get_expected_peak_hour(self, bins: List[TPeakBin]) -> int:
        """Get expected peak hour from distribution."""
        if not bins:
            return 14

        # Find bin with highest probability
        max_bin = max(bins, key=lambda b: b.probability)
        return (max_bin.bin_start_hour + max_bin.bin_end_hour) // 2

    # =========================================================================
    # Confidence & Explanations
    # =========================================================================

    def _calculate_confidence(
        self,
        state: DailyState,
        sigma_f: float
    ) -> Tuple[float, List[str]]:
        """
        Calculate overall confidence score (0-1).

        High confidence when:
        - Low sigma
        - QC is OK
        - Sources are fresh
        - Bias is stable
        """
        confidence = 1.0
        factors = []

        # Sigma penalty
        if sigma_f > 3.0:
            confidence -= 0.3
            factors.append("High uncertainty (sigma > 3°F)")
        elif sigma_f > 2.0:
            confidence -= 0.1
            factors.append("Moderate uncertainty")

        # QC penalty
        if state.last_qc_state == "UNCERTAIN":
            confidence -= 0.15
            factors.append("QC uncertainty")
        elif state.last_qc_state == "OUTLIER":
            confidence -= 0.3
            factors.append("QC outlier detected")

        # Staleness penalty
        stale_count = sum(1 for s in state.sources_health.values() if s in ("STALE", "DEAD"))
        if stale_count > 0:
            confidence -= 0.05 * stale_count
            factors.append(f"{stale_count} stale source(s)")

        # Bias stability bonus
        if state.bias_state.observation_count >= 5:
            if abs(state.bias_state.current_bias_f) < 1.0:
                confidence += 0.1
                factors.append("Stable bias")

        # No base forecast penalty
        if not self._base_forecast or not self._base_forecast.is_valid():
            confidence -= 0.2
            factors.append("No base forecast" if not self._base_forecast else "Base forecast stale")

        # Time-of-day adjustment
        now_local = datetime.now(UTC).astimezone(self.station_tz)
        if now_local.hour >= 14:
            confidence += 0.1
            factors.append("Post-peak observation window")

        return max(0.1, min(1.0, confidence)), factors

    def _build_explanations(self, state: DailyState) -> List[NowcastExplanation]:
        """
        Build step-by-step calculation breakdown for the prediction.
        """
        explanations = []

        # Get breakdown from state (set by _calculate_adjusted_tmax)
        breakdown = getattr(state, '_last_tmax_breakdown', None)

        if not breakdown:
            return explanations

        # === STEP-BY-STEP CALCULATION ===
        base_max = breakdown.get("base_max_f")
        bias = breakdown.get("decayed_bias_f", 0)
        after_bias = breakdown.get("after_bias_f")
        peak_hr = breakdown.get("peak_hour")
        current_hr = breakdown.get("current_hour", 0)
        obs_max = state.max_so_far_aligned_f if state.max_so_far_aligned_f > -900 else None
        final = breakdown.get("final_f")

        # Step 1: Base forecast
        if base_max is not None:
            explanations.append(NowcastExplanation(
                factor="1_BASE",
                contribution_f=base_max,
                description=f"Base HRRR forecast: {base_max:.1f}°F"
            ))

        # Step 2: Bias adjustment
        if bias is not None:
            sign = "+" if bias >= 0 else ""
            explanations.append(NowcastExplanation(
                factor="2_BIAS",
                contribution_f=bias,
                description=f"Bias correction: {sign}{bias:.2f}°F → {after_bias:.1f}°F"
            ))

        # Step 3: Peak hour status
        if peak_hr is not None:
            is_passed = current_hr >= peak_hr
            status = "PASSED" if is_passed else f"in {peak_hr - current_hr:.1f}h"
            explanations.append(NowcastExplanation(
                factor="3_PEAK",
                contribution_f=0.0,
                description=f"Peak hour: {peak_hr}:00 {self.station_tz_name} ({status}, now={current_hr:.0f}:00)"
            ))

        # Step 4: Observed maximum (floor constraint)
        if obs_max is not None:
            floor_applied = breakdown.get("floor_applied", False)
            floor_note = " [FLOOR ACTIVE]" if floor_applied else ""
            explanations.append(NowcastExplanation(
                factor="4_OBS_MAX",
                contribution_f=obs_max,
                description=f"Observed max: {obs_max:.0f}°F{floor_note}"
            ))
        else:
            # No observations recorded yet - flag this
            explanations.append(NowcastExplanation(
                factor="4_OBS_MAX",
                contribution_f=0.0,
                description="⚠️ No METAR observations yet - awaiting data"
            ))

        # Step 5: Post-Peak Cap (the key constraint)
        if breakdown.get("post_peak_cap_applied"):
            cap_val = breakdown.get("post_peak_cap_f", 0)
            margin = breakdown.get("post_peak_margin_f", 0)
            original = after_bias if after_bias else base_max
            reduction = original - cap_val if original else 0
            explanations.append(NowcastExplanation(
                factor="5_CAP",
                contribution_f=-reduction,
                description=f"⚠️ POST-PEAK CAP: {obs_max:.0f}°F + {margin:.1f}°F margin = {cap_val:.1f}°F (reduced from {original:.1f}°F)"
            ))
        elif current_hr >= (peak_hr or 15) + 1 and obs_max is not None:
            # 1+ hour past peak but cap not applied (prediction already below cap)
            explanations.append(NowcastExplanation(
                factor="5_CAP",
                contribution_f=0.0,
                description=f"Post-peak (+1h): prediction already ≤ observed max"
            ))
        elif current_hr >= (peak_hr or 15) + 1 and obs_max is None:
            # 1+ hour past peak but no observations - critical data gap!
            explanations.append(NowcastExplanation(
                factor="5_CAP",
                contribution_f=0.0,
                description="⚠️ 1+ HOUR PAST PEAK but no observations! Cap cannot apply"
            ))
        elif current_hr >= (peak_hr or 15) and current_hr < (peak_hr or 15) + 1:
            # At peak hour but cap not yet active (need 1+ hour past)
            explanations.append(NowcastExplanation(
                factor="5_CAP",
                contribution_f=0.0,
                description=f"At peak hour - cap activates in {((peak_hr or 15) + 1 - current_hr):.1f}h"
            ))

        # Step 6: Final result
        if final is not None:
            explanations.append(NowcastExplanation(
                factor="6_FINAL",
                contribution_f=final,
                description=f"═══ FINAL PREDICTION: {final:.1f}°F ═══"
            ))

        # QC impact (if any)
        if state.last_qc_state != "OK":
            explanations.append(NowcastExplanation(
                factor="qc",
                contribution_f=0.0,
                description=f"QC state: {state.last_qc_state} (increased uncertainty)",
                confidence_impact=-0.15
            ))

        return explanations  # Return full breakdown (removed [:3] truncation)

    def _get_inputs_used(self, state: DailyState) -> List[str]:
        """Get list of inputs used for traceability."""
        inputs = []

        if self._base_forecast:
            if self._base_forecast.is_valid():
                inputs.append(f"BaseForecast ({self._base_forecast.model_source})")
            else:
                inputs.append(f"BaseForecast({self._base_forecast.model_source})-STALE")

        if state.last_obs_source:
            inputs.append(f"METAR ({state.last_obs_source})")

        for source, status in state.sources_health.items():
            if status == "OK":
                inputs.append(source)

        inputs.append(f"Bias (n={state.bias_state.observation_count})")

        return inputs

    # =========================================================================
    # Periodic Update (Trigger A)
    # =========================================================================

    def periodic_update(self) -> Optional[NowcastDistribution]:
        """
        Periodic recalculation (every 60s).

        Recalculates outputs without fetching new data.
        """
        if self._last_update_utc:
            elapsed = (datetime.now(UTC) - self._last_update_utc).total_seconds()
            if elapsed < self.config.periodic_interval_seconds:
                return self._last_distribution

        return self.generate_distribution()

    # =========================================================================
    # Serialization
    # =========================================================================

    def get_state_snapshot(self) -> Dict:
        """Get full state snapshot for debugging."""
        state = self.get_or_create_daily_state()

        return {
            "station_id": self.station_id,
            "target_date": state.target_date.isoformat(),
            "max_so_far_f": state.max_so_far_aligned_f,
            "max_so_far_raw_f": state.max_so_far_raw_f,
            "last_obs_temp_f": state.last_obs_temp_f,
            "last_obs_aligned_f": state.last_obs_aligned_f,
            "bias_f": state.bias_state.current_bias_f,
            "bias_observations": state.bias_state.observation_count,
            "trend_f_per_hour": state.trend_f_per_hour,
            "sigma_f": state.sigma_f,
            "qc_state": state.last_qc_state,
            "sources_health": state.sources_health,
            "staleness_seconds": state.staleness_seconds,
            "base_forecast": {
                "source": self._base_forecast.model_source if self._base_forecast else None,
                "t_max_base_f": self._base_forecast.t_max_base_f if self._base_forecast else None,
                "t_peak_hour": self._base_forecast.t_peak_hour if self._base_forecast else None,
                "target_date": self._base_forecast.target_date.isoformat() if self._base_forecast else None,
                "valid_until_utc": self._base_forecast.valid_until_utc.isoformat() if self._base_forecast and self._base_forecast.valid_until_utc else None
            } if self._base_forecast else None,
            "update_count": self._update_count,
            "last_update_utc": self._last_update_utc.isoformat() if self._last_update_utc else None
        }


# =============================================================================
# Factory & Singleton Management
# =============================================================================

_engines: Dict[str, NowcastEngine] = {}


def get_nowcast_engine(station_id: str) -> NowcastEngine:
    """Get or create NowcastEngine for a station."""
    station_id = station_id.upper()
    if station_id not in _engines:
        _engines[station_id] = NowcastEngine(station_id)
    return _engines[station_id]


def reset_engines():
    """Reset all engines (for testing)."""
    global _engines
    _engines = {}
