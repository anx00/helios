import os
import sys
import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import AuxObs, OfficialObs
from core.nowcast_engine import NowcastConfig, NowcastEngine


UTC = ZoneInfo("UTC")


def _temp_c(temp_f: float) -> float:
    return (temp_f - 32.0) * 5.0 / 9.0


def _build_engine(station_id: str = "KLGA") -> NowcastEngine:
    cfg = NowcastConfig(
        pws_enabled=True,
        pws_min_metar_age_seconds=10 * 60,
        pws_max_age_seconds=30 * 60,
        pws_base_blend=0.45,
        pws_adjustment_cap_f=2.0,
    )
    return NowcastEngine(station_id=station_id, config=cfg)


def _install_base_forecast(engine: NowcastEngine, now_utc: datetime) -> None:
    # Stable base with peak at 23:00 (avoids post-peak cap edge-cases in tests).
    hourly = [56.0 for _ in range(24)]
    hourly[23] = 70.0
    target_date = engine._get_target_date(now_utc)
    engine.update_base_forecast(
        hourly_temps_f=hourly,
        model_run_time=now_utc - timedelta(hours=1),
        model_source="TEST",
        target_date=target_date,
    )


def _seed_metar(engine: NowcastEngine, now_utc: datetime, age_minutes: int, temp_f: float) -> None:
    obs_time = now_utc - timedelta(minutes=age_minutes)
    obs = OfficialObs(
        obs_time_utc=obs_time,
        ingest_time_utc=now_utc,
        source="TEST_METAR",
        station_id=engine.station_id,
        temp_c=_temp_c(temp_f),
        temp_f=temp_f,
        temp_f_raw=temp_f,
    )
    engine.update_observation(obs, qc_state="OK")


def _push_pws(
    engine: NowcastEngine,
    now_utc: datetime,
    temp_f: float,
    support: int,
    source: str,
    drift_c: float = 0.0,
    age_minutes: int = 1,
) -> None:
    aux = AuxObs(
        obs_time_utc=now_utc - timedelta(minutes=age_minutes),
        ingest_time_utc=now_utc,
        source=source,
        station_id=f"PWS_{engine.station_id}",
        temp_c=_temp_c(temp_f),
        temp_f=temp_f,
        is_aggregate=True,
        support=support,
        drift=drift_c,
    )
    engine.update_aux_observation(aux)


class TestNowcastPwsSoftAnchor(unittest.TestCase):
    def test_pws_increases_tmax_when_metar_is_stale(self):
        now_utc = datetime.now(UTC)
        engine = _build_engine("KLGA")
        _install_base_forecast(engine, now_utc)
        _seed_metar(engine, now_utc, age_minutes=45, temp_f=55.0)

        base = engine.generate_distribution()

        _push_pws(
            engine,
            now_utc,
            temp_f=58.0,
            support=20,
            source="PWS_SYNOPTIC+MADIS_MULTI",
            drift_c=1.6,
            age_minutes=1,
        )
        with_pws = engine.generate_distribution()

        self.assertGreater(with_pws.tmax_mean_f, base.tmax_mean_f)
        self.assertLessEqual(with_pws.tmax_sigma_f, base.tmax_sigma_f + 0.01)
        self.assertTrue(any("PWS soft-anchor" in s for s in with_pws.inputs_used))

    def test_pws_does_not_override_fresh_metar(self):
        now_utc = datetime.now(UTC)
        engine = _build_engine("KLGA")
        _install_base_forecast(engine, now_utc)
        _seed_metar(engine, now_utc, age_minutes=2, temp_f=55.0)

        base = engine.generate_distribution()

        _push_pws(
            engine,
            now_utc,
            temp_f=61.0,
            support=20,
            source="PWS_SYNOPTIC+MADIS_MULTI",
            drift_c=3.3,
            age_minutes=1,
        )
        with_pws = engine.generate_distribution()

        self.assertAlmostEqual(with_pws.tmax_mean_f, base.tmax_mean_f, places=2)

    def test_open_meteo_only_has_lower_influence_than_synoptic_mix(self):
        now_utc = datetime.now(UTC)

        eng_open = _build_engine("KLGA")
        _install_base_forecast(eng_open, now_utc)
        _seed_metar(eng_open, now_utc, age_minutes=45, temp_f=55.0)
        base_open = eng_open.generate_distribution()
        _push_pws(
            eng_open,
            now_utc,
            temp_f=60.0,
            support=20,
            source="PWS_OPEN_METEO",
            drift_c=2.8,
            age_minutes=1,
        )
        open_dist = eng_open.generate_distribution()

        eng_syn = _build_engine("KLGA")
        _install_base_forecast(eng_syn, now_utc)
        _seed_metar(eng_syn, now_utc, age_minutes=45, temp_f=55.0)
        base_syn = eng_syn.generate_distribution()
        _push_pws(
            eng_syn,
            now_utc,
            temp_f=60.0,
            support=20,
            source="PWS_SYNOPTIC+MADIS_MULTI",
            drift_c=1.6,
            age_minutes=1,
        )
        syn_dist = eng_syn.generate_distribution()

        open_shift = open_dist.tmax_mean_f - base_open.tmax_mean_f
        syn_shift = syn_dist.tmax_mean_f - base_syn.tmax_mean_f
        self.assertGreater(syn_shift, open_shift)

    def test_pws_switch_off_disables_adjustment(self):
        now_utc = datetime.now(UTC)
        cfg = NowcastConfig(
            pws_enabled=False,
            pws_min_metar_age_seconds=10 * 60,
            pws_max_age_seconds=30 * 60,
            pws_base_blend=0.45,
            pws_adjustment_cap_f=2.0,
        )
        engine = NowcastEngine(station_id="KLGA", config=cfg)
        _install_base_forecast(engine, now_utc)
        _seed_metar(engine, now_utc, age_minutes=45, temp_f=55.0)

        base = engine.generate_distribution()
        _push_pws(
            engine,
            now_utc,
            temp_f=60.0,
            support=20,
            source="PWS_SYNOPTIC+MADIS_MULTI",
            drift_c=1.2,
            age_minutes=1,
        )
        with_pws = engine.generate_distribution()

        self.assertAlmostEqual(with_pws.tmax_mean_f, base.tmax_mean_f, places=2)


if __name__ == "__main__":
    unittest.main()
