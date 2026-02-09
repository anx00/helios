import json

from market.discovery import _extract_market_token_ids


def test_extract_market_token_ids_includes_yes_and_no():
    markets = [
        {
            "groupItemTitle": "30-31°F",
            "clobTokenIds": ["yes_a", "no_a"],
        },
        {
            "groupItemTitle": "32-33°F",
            "clobTokenIds": ["yes_b", "no_b"],
        },
    ]

    parsed = _extract_market_token_ids(markets)
    assert parsed == [
        ("yes_a", "30-31°F", "yes"),
        ("no_a", "30-31°F", "no"),
        ("yes_b", "32-33°F", "yes"),
        ("no_b", "32-33°F", "no"),
    ]


def test_extract_market_token_ids_handles_json_and_dedupes():
    markets = [
        {
            "groupItemTitle": "28-29°F",
            "clobTokenIds": json.dumps(["yes_c", "no_c"]),
        },
        {
            "groupItemTitle": "28-29°F duplicate",
            "clobTokenIds": ["yes_c", "no_c"],
        },
    ]

    parsed = _extract_market_token_ids(markets)
    assert parsed == [("yes_c", "28-29°F", "yes"), ("no_c", "28-29°F", "no")]
