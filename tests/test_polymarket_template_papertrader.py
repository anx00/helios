from pathlib import Path


TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "templates" / "polymarket.html"


def test_papertrader_template_does_not_hardcode_bankroll_to_100():
    html = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert "initial_bankroll_usd: 100," not in html
    assert "Reset all 4 paper bots to $100" not in html
    assert "Reset wipes every lane back to $100" not in html
    assert '<input id="pt-bankroll"' in html
    assert "initial_bankroll_usd: ptDraft.initial_bankroll_usd" in html


def test_papertrader_template_surfaces_price_anomaly_as_activity():
    html = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert "Actividad reciente" in html
    assert "event === 'price_anomaly'" in html
    assert "Ask vivo" in html
    assert "Compra bloqueada" in html
    assert "_paperReasonText(ev)" in html
