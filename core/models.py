from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Literal
from zoneinfo import ZoneInfo

# Timezones
NYC = ZoneInfo("America/New_York")
MAD = ZoneInfo("Europe/Madrid")
UTC = ZoneInfo("UTC")

@dataclass
class ObservationEvent:
    """
    Base event for any observation.
    Every event carries dual timestamps (UTC + NYC + Madrid) for both
    obs_time and ingest_time, plus source tracking and latency.
    """
    obs_time_utc: datetime
    ingest_time_utc: datetime = field(default_factory=lambda: datetime.now(UTC))
    source: str = "UNKNOWN"
    raw_value: float = 0.0
    quality_flags: List[str] = field(default_factory=list)
    qc_state: Optional[str] = None  # Filled after QC: OK / UNCERTAIN / OUTLIER / SEVERE

    # --- obs_time conversions ---
    @property
    def obs_time_nyc(self) -> datetime:
        return self.obs_time_utc.astimezone(NYC)

    @property
    def obs_time_madrid(self) -> datetime:
        return self.obs_time_utc.astimezone(MAD)

    # --- ingest_time conversions ---
    @property
    def ingest_time_nyc(self) -> datetime:
        return self.ingest_time_utc.astimezone(NYC)

    @property
    def ingest_time_madrid(self) -> datetime:
        return self.ingest_time_utc.astimezone(MAD)

    @property
    def latency_seconds(self) -> float:
        """source_age = ingest_time_utc - obs_time_utc"""
        return (self.ingest_time_utc - self.obs_time_utc).total_seconds()

@dataclass
class OfficialObs(ObservationEvent):
    """
    Official Observation (METAR) - The 'Judge'.
    Sources: NOAA_JSON, AWC_XML, TGFTP_TXT
    """
    station_id: str = "UNKNOWN"
    temp_c: float = 0.0
    temp_f: float = 0.0  # Judge aligned (rounded to integer for settlement)
    temp_f_raw: float = 0.0  # Raw value with decimals (for display)
    temp_f_low: Optional[float] = None  # Lower bound when no T-group
    temp_f_high: Optional[float] = None  # Upper bound when no T-group
    settlement_f_low: Optional[int] = None  # Lower bound in settlement-space
    settlement_f_high: Optional[int] = None  # Upper bound in settlement-space
    has_t_group: bool = False
    dewpoint_c: Optional[float] = None
    wind_dir: Optional[int] = None
    wind_speed: Optional[int] = None
    sky_condition: str = "CLR"
    report_type: str = "METAR"  # "METAR" or "SPECI"
    is_speci: bool = False
    raw_metar: str = ""

    # Validation flags
    qc_passed: bool = False
    qc_flags: List[str] = field(default_factory=list)

@dataclass
class AuxObs(ObservationEvent):
    """
    Auxiliary Observation (PWS, Upstream).
    Used for context/confirmation, not settlement.
    """
    station_id: str = "UNKNOWN" # PWS ID or Upstream ICAO
    temp_c: float = 0.0
    temp_f: float = 0.0
    humidity: Optional[float] = None
    
    # Cluster stats (if aggregated)
    is_aggregate: bool = False
    support: int = 1 # Number of stations if aggregate
    drift: float = 0.0 # Deviation from official

@dataclass
class EnvFeature(ObservationEvent):
    """
    Physical Environment Variable (SST, AOD, etc.).
    """
    feature_type: Literal["SST", "AOD", "SOIL", "RADIATION"] = "SST"
    location_ref: str = "" # Buoy ID, Grid Cell
    value: float = 0.0
    unit: str = ""
    status: Literal["OK", "STALE", "Estimated"] = "OK"
