import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from core.pws_learning import PWSLearningStore


class _Reading:
    def __init__(
        self,
        label: str,
        temp_c: float,
        source: str,
        obs_time_utc: datetime,
        valid: bool = True,
    ):
        self.label = label
        self.temp_c = temp_c
        self.source = source
        self.obs_time_utc = obs_time_utc
        self.valid = valid


class TestPwsLearning(unittest.TestCase):
    def test_source_priors_applied_for_new_station(self):
        with TemporaryDirectory() as td:
            store = PWSLearningStore(path=Path(td) / "weights.json")
            syn = store.get_weight("KLGA", "SYN_A", "SYNOPTIC")
            open_meteo = store.get_weight("KLGA", "OM_A", "OPEN_METEO")
            self.assertGreater(syn, open_meteo)

    def test_lead_signal_does_not_count_as_now_accuracy(self):
        with TemporaryDirectory() as td:
            store = PWSLearningStore(path=Path(td) / "weights.json")

            pws_t = datetime(2026, 2, 17, 16, 0, tzinfo=timezone.utc)
            metar_t = datetime(2026, 2, 17, 16, 51, tzinfo=timezone.utc)

            reading = _Reading(
                label="P1",
                temp_c=20.0,
                source="SYNOPTIC",
                obs_time_utc=pws_t,
            )
            store.update_with_official(
                market_station_id="KLGA",
                official_temp_c=20.0,
                readings=[reading],
                obs_time_utc=metar_t,
                official_obs_time_utc=metar_t,
            )

            profile = store.get_station_profile("KLGA", "P1", "SYNOPTIC")
            self.assertEqual(profile.get("now_samples"), 0)
            self.assertEqual(profile.get("lead_samples"), 1)

    def test_now_accuracy_drives_weight_ordering(self):
        with TemporaryDirectory() as td:
            store = PWSLearningStore(path=Path(td) / "weights.json")
            base_t = datetime(2026, 2, 17, 15, 50, tzinfo=timezone.utc)

            for i in range(30):
                t = base_t + timedelta(hours=i)
                good = _Reading("GOOD", 20.1, "SYNOPTIC", t)
                bad = _Reading("BAD", 24.0, "SYNOPTIC", t)
                store.update_with_official(
                    market_station_id="KLGA",
                    official_temp_c=20.0,
                    readings=[good, bad],
                    obs_time_utc=t,
                    official_obs_time_utc=t,
                )

            w_good = store.get_weight("KLGA", "GOOD", "SYNOPTIC")
            w_bad = store.get_weight("KLGA", "BAD", "SYNOPTIC")
            self.assertGreater(w_good, w_bad)

    def test_market_specific_profiles(self):
        with TemporaryDirectory() as td:
            store = PWSLearningStore(path=Path(td) / "weights.json")
            base_t = datetime(2026, 2, 17, 12, 0, tzinfo=timezone.utc)

            for i in range(20):
                t = base_t + timedelta(hours=i)
                store.update_with_official(
                    market_station_id="KLGA",
                    official_temp_c=20.0,
                    readings=[_Reading("PX1", 20.0, "WUNDERGROUND", t)],
                    obs_time_utc=t,
                    official_obs_time_utc=t,
                )
                store.update_with_official(
                    market_station_id="KATL",
                    official_temp_c=20.0,
                    readings=[_Reading("PX1", 24.0, "WUNDERGROUND", t)],
                    obs_time_utc=t,
                    official_obs_time_utc=t,
                )

            w_klga = store.get_weight("KLGA", "PX1", "WUNDERGROUND")
            w_katl = store.get_weight("KATL", "PX1", "WUNDERGROUND")
            self.assertGreater(w_klga, w_katl)

    def test_pending_snapshot_can_be_scored_later_as_lead(self):
        with TemporaryDirectory() as td:
            store = PWSLearningStore(path=Path(td) / "weights.json")
            pws_t = datetime(2026, 2, 17, 10, 0, tzinfo=timezone.utc)
            metar_t = datetime(2026, 2, 17, 10, 55, tzinfo=timezone.utc)

            store.ingest_pending(
                market_station_id="KLGA",
                readings=[_Reading("LATE_LABEL", 18.0, "MADIS_APRSWXNET", pws_t)],
                obs_time_utc=pws_t,
            )
            store.update_with_official(
                market_station_id="KLGA",
                official_temp_c=18.0,
                readings=[],
                obs_time_utc=metar_t,
                official_obs_time_utc=metar_t,
            )

            profile = store.get_station_profile("KLGA", "LATE_LABEL", "MADIS_APRSWXNET")
            self.assertEqual(profile.get("lead_samples"), 1)
            self.assertEqual(profile.get("now_samples"), 0)


if __name__ == "__main__":
    unittest.main()
