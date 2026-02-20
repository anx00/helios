import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collector.pws_fetcher import (
    PWSReading,
    _build_madis_query_params,
    _effective_min_support,
    _is_real_pws_source,
    _load_wunderground_candidates,
    _normalize_open_meteo_payload,
    _parse_madis_xml_records,
    _parse_open_meteo_obs_time,
    _select_open_meteo_readings,
    _source_weight_multiplier,
    _weighted_mad,
    _weighted_median,
)


class TestMadisPwsParser(unittest.TestCase):
    def test_parse_madis_xml_filters_qc_age_and_dedup(self):
        now = datetime.now(timezone.utc)
        t_recent_1 = (now - timedelta(minutes=12)).strftime("%Y-%m-%dT%H:%M")
        t_recent_2 = (now - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M")
        t_stale = (now - timedelta(minutes=150)).strftime("%Y-%m-%dT%H:%M")

        xml_text = f"""<?xml version="1.0" ?>
<mesonet>
  <record var="V-T" shef_id="A1" provider="APRSWXNET" lat="40.78" lon="-73.87" ObTime="{t_recent_1}" data_value="279.15" QCD="V" />
  <record var="V-T" shef_id="A1" provider="APRSWXNET" lat="40.78" lon="-73.87" ObTime="{t_recent_2}" data_value="280.15" QCD="V" />
  <record var="V-T" shef_id="B1" provider="MesoWest" lat="40.79" lon="-73.88" ObTime="{t_recent_1}" data_value="281.15" QCD="X" />
  <record var="V-T" shef_id="C1" provider="APRSWXNET" lat="40.80" lon="-73.86" ObTime="{t_stale}" data_value="282.15" QCD="V" />
  <record var="V-T" shef_id="D1" provider="APRSWXNET" lat="40.81" lon="-73.85" ObTime="{t_recent_1}" data_value="100.00" QCD="V" />
  <record var="V-T" shef_id="E1" provider="APRSWXNET" lat="40.77" lon="-73.90" ObTime="{t_recent_1}" data_value="285.15" QCD="S" />
</mesonet>
"""

        readings = _parse_madis_xml_records(
            xml_text=xml_text,
            center_lat=40.7769,
            center_lon=-73.8740,
            max_age_minutes=90,
            accepted_qcd={"C", "S", "V"},
        )

        self.assertEqual(len(readings), 2)

        by_label = {r.label: r for r in readings}
        self.assertIn("A1", by_label)
        self.assertIn("E1", by_label)

        # Dedup should keep latest A1 record (280.15K -> 7.0C)
        self.assertAlmostEqual(by_label["A1"].temp_c, 7.0, places=2)
        self.assertEqual(by_label["A1"].source, "MADIS_APRSWXNET")

        # QCD=S accepted and converted correctly (285.15K -> 12.0C)
        self.assertAlmostEqual(by_label["E1"].temp_c, 12.0, places=2)
        self.assertGreaterEqual(by_label["E1"].distance_km, 0.0)

    def test_build_query_params_includes_expected_fields(self):
        params = _build_madis_query_params(
            station_id="KLGA",
            center_lat=40.7769,
            center_lon=-73.8740,
            radius_km=25,
            providers=["APRSWXNET", "MesoWest"],
            max_age_minutes=90,
        )

        self.assertIn(("nvars", "T"), params)
        self.assertIn(("qctype", "0"), params)
        self.assertIn(("qcsel", "2"), params)
        self.assertIn(("pvd", "APRSWXNET"), params)
        self.assertIn(("pvd", "MesoWest"), params)

    def test_normalize_open_meteo_payload(self):
        dict_payload = {"current": {"temperature_2m": 7.5}}
        dict_out = _normalize_open_meteo_payload(dict_payload)
        self.assertEqual(len(dict_out), 1)
        self.assertEqual(dict_out[0]["current"]["temperature_2m"], 7.5)

        list_payload = [
            {"current": {"temperature_2m": 5.0}},
            "bad",
            None,
            {"current": {"temperature_2m": 6.0}},
        ]
        list_out = _normalize_open_meteo_payload(list_payload)
        self.assertEqual(len(list_out), 2)
        self.assertEqual(list_out[0]["current"]["temperature_2m"], 5.0)
        self.assertEqual(list_out[1]["current"]["temperature_2m"], 6.0)

    def test_parse_open_meteo_obs_time(self):
        zulu = _parse_open_meteo_obs_time("2026-02-09T19:00Z")
        self.assertIsNotNone(zulu)
        self.assertIsNotNone(zulu.tzinfo)
        self.assertEqual(zulu.utcoffset(), timezone.utc.utcoffset(None))

        naive = _parse_open_meteo_obs_time("2026-02-09T19:00")
        self.assertIsNotNone(naive)
        self.assertIsNotNone(naive.tzinfo)
        self.assertEqual(naive.utcoffset(), timezone.utc.utcoffset(None))

        self.assertIsNone(_parse_open_meteo_obs_time("invalid"))

    def test_weighted_median_prefers_higher_weight_station(self):
        values = [10.0, 20.0, 30.0]
        weights = [0.1, 0.2, 5.0]
        self.assertEqual(_weighted_median(values, weights), 30.0)

    def test_weighted_mad_reflects_weighted_center(self):
        values = [10.0, 20.0, 30.0]
        weights = [0.1, 0.2, 5.0]
        center = _weighted_median(values, weights)
        mad = _weighted_mad(values, weights, center)
        self.assertEqual(center, 30.0)
        self.assertEqual(mad, 0.0)

    def test_wunderground_defaults_include_non_us_stations(self):
        eglc = _load_wunderground_candidates("EGLC", max_stations=20)
        ltac = _load_wunderground_candidates("LTAC", max_stations=20)

        self.assertGreater(len(eglc), 0)
        self.assertGreater(len(ltac), 0)

        eglc_ids = {str(r.get("station_id")) for r in eglc}
        ltac_ids = {str(r.get("station_id")) for r in ltac}
        self.assertIn("ILONDO288", eglc_ids)
        self.assertIn("IANKAR46", ltac_ids)

    def test_effective_min_support_relaxes_for_sparse_priority_markets(self):
        self.assertEqual(_effective_min_support("EGLC", 3, real_count=0), 3)
        self.assertEqual(_effective_min_support("EGLC", 3, real_count=1), 1)
        self.assertEqual(_effective_min_support("EGLC", 3, real_count=2), 2)
        self.assertEqual(_effective_min_support("LTAC", 4, real_count=2), 2)
        self.assertEqual(_effective_min_support("KLGA", 3, real_count=1), 3)

    def test_select_open_meteo_readings_caps_when_real_sources_exist(self):
        sample = [
            PWSReading(
                lat=0.0,
                lon=0.0,
                temp_c=10.0 + idx,
                label=f"OM_{idx}",
                source="OPEN_METEO",
                distance_km=float(idx),
                age_minutes=float(idx),
            )
            for idx in range(6)
        ]
        eglc_with_real = _select_open_meteo_readings("EGLC", sample, real_count=3)
        self.assertEqual(len(eglc_with_real), 2)
        self.assertEqual([r.label for r in eglc_with_real], ["OM_0", "OM_1"])

        eglc_no_real = _select_open_meteo_readings("EGLC", sample, real_count=0)
        self.assertEqual(len(eglc_no_real), 6)

        klga_dense_real = _select_open_meteo_readings("KLGA", sample, real_count=6)
        self.assertEqual(len(klga_dense_real), 3)

    def test_source_weight_multiplier_prioritizes_real_pws_for_eglc(self):
        w_wu = _source_weight_multiplier("EGLC", "WUNDERGROUND", real_count=3)
        w_madis = _source_weight_multiplier("EGLC", "MADIS_APRSWXNET", real_count=3)
        w_om = _source_weight_multiplier("EGLC", "OPEN_METEO", real_count=3)

        self.assertGreater(w_wu, w_madis)
        self.assertGreater(w_madis, w_om)
        self.assertLess(w_om, 1.0)

    def test_real_source_classifier(self):
        self.assertTrue(_is_real_pws_source("SYNOPTIC"))
        self.assertTrue(_is_real_pws_source("WUNDERGROUND"))
        self.assertTrue(_is_real_pws_source("MADIS_APRSWXNET"))
        self.assertFalse(_is_real_pws_source("OPEN_METEO"))


if __name__ == "__main__":
    unittest.main()
