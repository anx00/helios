from pathlib import Path


TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "templates" / "pws.html"


def test_pws_template_wires_city_export_button():
    html = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert 'id="pws-export-city-btn"' in html
    assert "Export full city JSON" in html
    assert "function exportPwsCityJson()" in html
    assert "/api/pws/export/city?" in html
    assert "getPwsExportFilename(data, stationId)" in html
