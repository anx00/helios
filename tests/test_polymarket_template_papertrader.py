from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLYMARKET_TEMPLATE = ROOT / "templates" / "polymarket.html"
BASE_TEMPLATE = ROOT / "templates" / "base.html"


def test_polymarket_template_removes_active_papertrader_ui():
    html = POLYMARKET_TEMPLATE.read_text(encoding="utf-8")

    assert "renderPaperTrader(data.papertrader);" not in html
    assert '<div id="pt-panel"' not in html
    assert "> Paper Trading<" not in html
    assert "<i data-lucide=\"crosshair\"></i> Trading Signal" in html


def test_polymarket_template_contains_klga_comparison_panel():
    html = POLYMARKET_TEMPLATE.read_text(encoding="utf-8")

    assert 'id="pm-comparison-section"' in html
    assert 'id="pm-comparison-summary"' in html
    assert "/api/experimental/live-station/" in html
    assert "KLGA Comparison" in html
    assert "isComparisonStation" in html
    assert "Stable comparison for the current KLGA context." in html


def test_base_template_surfaces_polymarket_in_sidebar():
    html = BASE_TEMPLATE.read_text(encoding="utf-8")

    assert 'href="/polymarket"' in html
    assert "<span>Polymarket</span>" in html
    assert "active_page == 'trading'" in html
