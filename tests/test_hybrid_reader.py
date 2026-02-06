import json


def _write_events(base_dir, date_str, channel, events):
    channel_dir = base_dir / f"date={date_str}" / f"ch={channel}"
    channel_dir.mkdir(parents=True, exist_ok=True)
    file_path = channel_dir / "events.ndjson"
    with open(file_path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


def test_list_available_dates_filters_ndjson_by_station(tmp_path):
    from core.compactor import HybridReader

    ndjson_base = tmp_path / "recordings"
    parquet_base = tmp_path / "parquet"

    _write_events(
        ndjson_base,
        "2026-02-05",
        "world",
        [{"station_id": "KLGA", "ts_ingest_utc": "2026-02-05T10:00:00+00:00"}],
    )
    _write_events(
        ndjson_base,
        "2026-02-06",
        "world",
        [{"station_id": "KJFK", "ts_ingest_utc": "2026-02-06T10:00:00+00:00"}],
    )
    # Empty file should not make the date selectable.
    _write_events(ndjson_base, "2026-02-07", "world", [])

    reader = HybridReader(str(ndjson_base), str(parquet_base))

    assert reader.list_available_dates("KLGA") == ["2026-02-05"]
    assert reader.list_available_dates("KJFK") == ["2026-02-06"]
    assert reader.list_available_dates() == ["2026-02-06", "2026-02-05"]


def test_list_channels_for_date_filters_ndjson_by_station(tmp_path):
    from core.compactor import HybridReader

    ndjson_base = tmp_path / "recordings"
    parquet_base = tmp_path / "parquet"

    _write_events(
        ndjson_base,
        "2026-02-06",
        "world",
        [{"station_id": "KJFK", "ts_ingest_utc": "2026-02-06T10:00:00+00:00"}],
    )
    _write_events(
        ndjson_base,
        "2026-02-06",
        "nowcast",
        [{"station_id": "KLGA", "ts_ingest_utc": "2026-02-06T10:01:00+00:00"}],
    )

    reader = HybridReader(str(ndjson_base), str(parquet_base))

    assert set(reader.list_channels_for_date("2026-02-06")) == {"world", "nowcast"}
    assert reader.list_channels_for_date("2026-02-06", "KLGA") == ["nowcast"]
    assert reader.list_channels_for_date("2026-02-06", "KJFK") == ["world"]


def test_read_channel_ndjson_filters_station_and_keeps_global_events(tmp_path):
    from core.compactor import HybridReader

    ndjson_base = tmp_path / "recordings"
    parquet_base = tmp_path / "parquet"

    _write_events(
        ndjson_base,
        "2026-02-06",
        "world",
        [
            {"station_id": "KJFK", "ts_ingest_utc": "2026-02-06T10:00:00+00:00"},
            {"ts_ingest_utc": "2026-02-06T10:00:30+00:00"},  # global/system event
            {"station_id": "KLGA", "ts_ingest_utc": "2026-02-06T10:01:00+00:00"},
        ],
    )

    reader = HybridReader(str(ndjson_base), str(parquet_base))

    events_klga = reader.read_channel("2026-02-06", "world", "KLGA")
    assert len(events_klga) == 2
    assert all(
        e.get("station_id") in (None, "KLGA")
        for e in events_klga
    )

