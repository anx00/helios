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
            metar_t = datetime(2026, 2, 17, 16, 25, tzinfo=timezone.utc)

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
            self.assertEqual(profile.get("lead_hits"), 1)
            self.assertEqual(profile.get("lead_score"), 1)
            self.assertFalse(profile.get("rank_eligible"))

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
            metar_t = datetime(2026, 2, 17, 10, 20, tzinfo=timezone.utc)

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

    def test_rank_eligibility_requires_now_and_lead_samples(self):
        with TemporaryDirectory() as td:
            store = PWSLearningStore(path=Path(td) / "weights.json")

            pws_t0 = datetime(2026, 2, 17, 10, 0, tzinfo=timezone.utc)
            metar_t0 = datetime(2026, 2, 17, 10, 25, tzinfo=timezone.utc)
            store.update_with_official(
                market_station_id="KLGA",
                official_temp_c=20.0,
                readings=[_Reading("RANK_PWS", 20.0, "SYNOPTIC", pws_t0)],
                obs_time_utc=metar_t0,
                official_obs_time_utc=metar_t0,
            )

            pws_t1 = datetime(2026, 2, 17, 12, 0, tzinfo=timezone.utc)
            metar_t1 = datetime(2026, 2, 17, 12, 0, tzinfo=timezone.utc)
            store.update_with_official(
                market_station_id="KLGA",
                official_temp_c=20.0,
                readings=[_Reading("RANK_PWS", 20.0, "SYNOPTIC", pws_t1)],
                obs_time_utc=metar_t1,
                official_obs_time_utc=metar_t1,
            )

            profile = store.get_station_profile("KLGA", "RANK_PWS", "SYNOPTIC")
            self.assertEqual(profile.get("lead_samples"), 1)
            self.assertEqual(profile.get("now_samples"), 1)
            self.assertFalse(profile.get("rank_eligible"))

            pws_t2 = datetime(2026, 2, 17, 13, 0, tzinfo=timezone.utc)
            metar_t2 = datetime(2026, 2, 17, 13, 0, tzinfo=timezone.utc)
            store.update_with_official(
                market_station_id="KLGA",
                official_temp_c=20.0,
                readings=[_Reading("RANK_PWS", 20.0, "SYNOPTIC", pws_t2)],
                obs_time_utc=metar_t2,
                official_obs_time_utc=metar_t2,
            )

            profile = store.get_station_profile("KLGA", "RANK_PWS", "SYNOPTIC")
            self.assertGreaterEqual(profile.get("lead_samples") or 0, 1)
            self.assertGreaterEqual(profile.get("now_samples") or 0, 2)
            self.assertTrue(profile.get("rank_eligible"))

    def test_lead_counts_once_per_station_per_official_obs(self):
        with TemporaryDirectory() as td:
            store = PWSLearningStore(path=Path(td) / "weights.json")
            base = datetime(2026, 2, 17, 10, 0, tzinfo=timezone.utc)
            station = "L1"

            # Three pending snapshots for same station in the 5-30 min lead window.
            rows = [
                _Reading(station, 18.0, "SYNOPTIC", base),
                _Reading(station, 18.0, "SYNOPTIC", base + timedelta(minutes=10)),
                _Reading(station, 18.0, "SYNOPTIC", base + timedelta(minutes=20)),
            ]
            store.ingest_pending("KLGA", rows, obs_time_utc=base + timedelta(minutes=20))

            official_t = datetime(2026, 2, 17, 10, 30, tzinfo=timezone.utc)
            store.update_with_official(
                market_station_id="KLGA",
                official_temp_c=18.0,
                readings=[],
                obs_time_utc=official_t,
                official_obs_time_utc=official_t,
            )

            profile = store.get_station_profile("KLGA", station, "SYNOPTIC")
            self.assertEqual(profile.get("lead_samples"), 1)
            self.assertGreaterEqual(int(profile.get("last_lead_candidates_in_window") or 0), 3)

    def test_lead_score_counts_cumulative_hits(self):
        with TemporaryDirectory() as td:
            store = PWSLearningStore(path=Path(td) / "weights.json")
            pws_t0 = datetime(2026, 2, 17, 10, 0, tzinfo=timezone.utc)
            metar_t0 = datetime(2026, 2, 17, 10, 25, tzinfo=timezone.utc)
            pws_t1 = datetime(2026, 2, 17, 11, 0, tzinfo=timezone.utc)
            metar_t1 = datetime(2026, 2, 17, 11, 25, tzinfo=timezone.utc)

            store.update_with_official(
                market_station_id="KLGA",
                official_temp_c=20.0,
                readings=[_Reading("LEAD_CNT", 20.0, "SYNOPTIC", pws_t0)],
                obs_time_utc=metar_t0,
                official_obs_time_utc=metar_t0,
            )
            store.update_with_official(
                market_station_id="KLGA",
                official_temp_c=20.0,
                readings=[_Reading("LEAD_CNT", 27.0, "SYNOPTIC", pws_t1)],
                obs_time_utc=metar_t1,
                official_obs_time_utc=metar_t1,
            )

            profile = store.get_station_profile("KLGA", "LEAD_CNT", "SYNOPTIC")
            self.assertEqual(profile.get("lead_samples"), 2)
            self.assertEqual(profile.get("lead_hits"), 1)
            self.assertEqual(profile.get("lead_score"), 1)

    def test_lead_window_extends_to_45_minutes_and_tracks_bucket_stats(self):
        with TemporaryDirectory() as td:
            store = PWSLearningStore(path=Path(td) / "weights.json")
            pws_t = datetime(2026, 2, 17, 10, 0, tzinfo=timezone.utc)
            metar_t = datetime(2026, 2, 17, 10, 40, tzinfo=timezone.utc)

            store.update_with_official(
                market_station_id="KLGA",
                official_temp_c=20.0,
                readings=[_Reading("LEAD_45", 19.5, "SYNOPTIC", pws_t)],
                obs_time_utc=metar_t,
                official_obs_time_utc=metar_t,
            )

            profile = store.get_station_profile("KLGA", "LEAD_45", "SYNOPTIC")
            self.assertEqual(profile.get("lead_samples"), 1)
            self.assertEqual(profile.get("lead_hits"), 1)
            self.assertAlmostEqual(profile.get("lead_avg_minutes"), 40.0, places=2)
            buckets = profile.get("lead_bucket_summaries") or {}
            self.assertIn("30-45", buckets)
            self.assertEqual((buckets["30-45"] or {}).get("count"), 1)
            self.assertGreaterEqual(float(profile.get("next_metar_prob_within_1_0_c") or 0.0), 1.0)

    def test_audit_rows_are_persisted_per_market_day(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            store = PWSLearningStore(path=root / "weights.json")
            pws_t = datetime(2026, 2, 17, 10, 0, tzinfo=timezone.utc)
            metar_t = datetime(2026, 2, 17, 10, 20, tzinfo=timezone.utc)

            store.update_with_official(
                market_station_id="KLGA",
                official_temp_c=18.0,
                readings=[_Reading("AUDIT_PWS", 18.0, "MADIS_APRSWXNET", pws_t)],
                obs_time_utc=metar_t,
                official_obs_time_utc=metar_t,
            )

            dates = store.list_audit_dates("KLGA")
            self.assertEqual(dates, ["2026-02-17"])
            rows = store.load_audit_rows("KLGA", "2026-02-17")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].get("kind"), "LEAD")
            self.assertEqual(rows[0].get("station_id"), "AUDIT_PWS")

    def test_source_prior_fades_with_strong_historical_evidence(self):
        with TemporaryDirectory() as td:
            store = PWSLearningStore(path=Path(td) / "weights.json")
            base_t = datetime(2026, 2, 17, 15, 0, tzinfo=timezone.utc)

            # Same perfect behavior for both stations; after enough evidence,
            # source prior should have little impact on active weight.
            for i in range(64):
                t = base_t + timedelta(hours=i)
                syn = _Reading("SYN_EQ", 20.0, "SYNOPTIC", t)
                wu = _Reading("WU_EQ", 20.0, "WUNDERGROUND", t)
                store.update_with_official(
                    market_station_id="KLGA",
                    official_temp_c=20.0,
                    readings=[syn, wu],
                    obs_time_utc=t,
                    official_obs_time_utc=t,
                )

            w_syn = store.get_weight("KLGA", "SYN_EQ", "SYNOPTIC")
            w_wu = store.get_weight("KLGA", "WU_EQ", "WUNDERGROUND")
            self.assertGreater(w_syn, 0)
            self.assertGreater(w_wu, 0)
            ratio = w_syn / w_wu if w_wu > 0 else 10.0
            self.assertLess(ratio, 1.08)

    def test_invalid_samples_do_not_train_scores(self):
        with TemporaryDirectory() as td:
            store = PWSLearningStore(path=Path(td) / "weights.json")
            t = datetime(2026, 2, 21, 12, 0, tzinfo=timezone.utc)

            bad = _Reading("BAD_INV", 28.0, "WUNDERGROUND", t, valid=False)
            store.update_with_official(
                market_station_id="LTAC",
                official_temp_c=10.0,
                readings=[bad],
                obs_time_utc=t,
                official_obs_time_utc=t,
            )

            profile = store.get_station_profile("LTAC", "BAD_INV", "WUNDERGROUND")
            self.assertEqual(profile.get("now_samples"), 0)
            self.assertEqual(profile.get("lead_samples"), 0)


if __name__ == "__main__":
    unittest.main()
