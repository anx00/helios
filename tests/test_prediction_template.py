from pathlib import Path


def test_prediction_template_exposes_probability_lab_sections():
    template = Path("templates/prediction.html").read_text(encoding="utf-8")

    assert "Prediction Lab" in template
    assert "Target-day forward curves" in template
    assert "Official + PWS same-day layer" in template
    assert "Historical ranking of city stations" in template
    assert "/api/prediction/" in template
    assert "What we have, what is missing, what still needs study" in template
