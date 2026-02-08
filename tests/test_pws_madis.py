import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collector.pws_fetcher import _build_madis_query_params, _parse_madis_xml_records


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


if __name__ == "__main__":
    unittest.main()
