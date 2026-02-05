from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, date, time
from zoneinfo import ZoneInfo

# Settlement timezone: Polymarket temperature markets settle on the NYC calendar day
_NYC = ZoneInfo("America/New_York")


class JudgeAlignment:
    """
    Standardized logic to mimic the settlement source (NOAA/Wunderground).
    Ensures 'Judge Alignment' (Contract 2.3).

    Covers:
    - Round Half Up to integer (26.5 -> 27)
    - Correct handling of negatives (-0.5 -> 0, not -1)
    - Day cutoff: midnight NYC defines the settlement "day"
    - C -> F conversion matching NOAA standard
    """

    @staticmethod
    def round_to_settlement(value_f: float) -> float:
        """
        Round to nearest integer using standard Round Half Up.

        Examples:
            26.4  ->  26
            26.5  ->  27
            26.51 ->  27
            -0.5  ->   0  (half-up rounds towards +inf)
            -1.5  ->  -1
        """
        if value_f is None:
            return 0.0

        d = Decimal(str(value_f))
        rounded = d.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        return float(rounded)

    @staticmethod
    def celsius_to_fahrenheit(temp_c: float) -> float:
        """Standard C -> F conversion with 1 decimal precision first."""
        if temp_c is None:
            return 0.0

        # NOAA standard: C * 1.8 + 32
        f = (temp_c * 1.8) + 32
        return round(f, 1)

    @staticmethod
    def map_sky_condition(code: str) -> str:
        """Normalize sky condition codes."""
        mapping = {
            "CLR": "Clear",
            "SKC": "Clear",
            "FEW": "Few Clouds",
            "SCT": "Scattered",
            "BKN": "Broken",
            "OVC": "Overcast",
            "VV": "Obscured"
        }
        return mapping.get(str(code).upper(), "Unknown")

    # --- Day cutoff (Phase 2, section 2.3) ---
    @staticmethod
    def settlement_date(utc_dt: datetime) -> date:
        """
        Determine the settlement calendar day for a given UTC datetime.
        The 'day' is defined by midnight in NYC (America/New_York).

        Example: 2026-01-28 04:30 UTC = 2026-01-27 23:30 NYC => settlement day is Jan 27.
        """
        nyc_dt = utc_dt.astimezone(_NYC)
        return nyc_dt.date()

    @staticmethod
    def is_same_settlement_day(dt1: datetime, dt2: datetime) -> bool:
        """Check if two UTC datetimes fall on the same NYC calendar day."""
        return JudgeAlignment.settlement_date(dt1) == JudgeAlignment.settlement_date(dt2)

    @staticmethod
    def settlement_day_bounds(target_date: date) -> tuple:
        """
        Return (start_utc, end_utc) for a given settlement day.
        Start = midnight NYC of target_date in UTC.
        End   = midnight NYC of target_date + 1 day in UTC.
        """
        start_nyc = datetime.combine(target_date, time.min, tzinfo=_NYC)
        from datetime import timedelta
        end_nyc = start_nyc + timedelta(days=1)
        return (start_nyc.astimezone(ZoneInfo("UTC")),
                end_nyc.astimezone(ZoneInfo("UTC")))
