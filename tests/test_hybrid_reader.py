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


def test_replay_light_prefers_parquet_and_slims_l2_payload(tmp_path):
    from core.compactor import Compactor, HybridReader

    ndjson_base = tmp_path / "recordings"
    parquet_base = tmp_path / "parquet"
    original_events = [
        {
            "station_id": "KLGA",
            "ts_ingest_utc": "2026-02-06T10:00:00+00:00",
            "data": {
                "40-41F": {
                    "mid": 0.55,
                    "best_bid": 0.52,
                    "best_ask": 0.58,
                    "bids": [[0.52, 10]],
                    "asks": [[0.58, 12]],
                }
            },
        }
    ]
    _write_events(ndjson_base, "2026-02-06", "l2_snap", original_events)

    compactor = Compactor(str(ndjson_base), str(parquet_base), batch_size=10, max_dates_per_run=10)
    stats = compactor.compact_channel("2026-02-06", "l2_snap")
    assert stats["events_read"] == 1

    # Mutate NDJSON after compaction: replay_light should still use the compacted parquet copy.
    _write_events(
        ndjson_base,
        "2026-02-06",
        "l2_snap",
        [
            {
                "station_id": "KLGA",
                "ts_ingest_utc": "2026-02-06T10:00:00+00:00",
                "data": {
                    "40-41F": {
                        "mid": 0.99,
                        "best_bid": 0.98,
                        "best_ask": 1.00,
                        "bids": [[0.98, 1]],
                        "asks": [[1.00, 1]],
                    }
                },
            }
        ],
    )

    reader = HybridReader(str(ndjson_base), str(parquet_base))
    events = reader.read_channel("2026-02-06", "l2_snap", "KLGA", read_policy="replay_light")

    assert len(events) == 1
    book = events[0]["data"]["40-41F"]
    assert book["mid"] == 0.55
    assert "bids" not in book
    assert "asks" not in book


def test_replay_light_falls_back_to_ndjson_when_parquet_missing(tmp_path):
    from core.compactor import HybridReader

    ndjson_base = tmp_path / "recordings"
    parquet_base = tmp_path / "parquet"
    _write_events(
        ndjson_base,
        "2026-02-06",
        "l2_snap",
        [
            {
                "station_id": "KLGA",
                "ts_ingest_utc": "2026-02-06T10:00:00+00:00",
                "data": {
                    "40-41F": {
                        "mid": 0.61,
                        "best_bid": 0.60,
                        "best_ask": 0.62,
                        "bids": [[0.60, 10]],
                        "asks": [[0.62, 12]],
                    }
                },
            }
        ],
    )

    reader = HybridReader(str(ndjson_base), str(parquet_base))
    events = reader.read_channel("2026-02-06", "l2_snap", "KLGA", read_policy="replay_light")

    assert len(events) == 1
    book = events[0]["data"]["40-41F"]
    assert book["mid"] == 0.61
    assert "bids" not in book
    assert "asks" not in book
