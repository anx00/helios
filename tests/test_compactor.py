import json
from pathlib import Path
import shutil
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from core.compactor import Compactor, ParquetReader


def _write_ndjson(base_dir: Path, *, date_str: str, channel: str, events):
    channel_dir = base_dir / f"date={date_str}" / f"ch={channel}"
    channel_dir.mkdir(parents=True, exist_ok=True)
    event_file = channel_dir / "events.ndjson"
    with event_file.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")


def test_compactor_streams_large_channel_into_multiple_parts(tmp_path):
    ndjson_base = tmp_path / "recordings"
    parquet_base = tmp_path / "parquet"
    events = [
        {
            "ts_ingest_utc": f"2026-02-01T00:00:0{i}Z",
            "station_id": "KLGA",
            "data": {
                "temp_f": 40 + i,
                "wind_dir": 90 if i % 2 == 0 else "VRB",
            },
        }
        for i in range(5)
    ]
    _write_ndjson(ndjson_base, date_str="2026-02-01", channel="world", events=events)

    compactor = Compactor(
        str(ndjson_base),
        str(parquet_base),
        batch_size=2,
        max_dates_per_run=10,
    )
    stats = compactor.compact_channel("2026-02-01", "world")

    assert stats["events_read"] == 5
    part_files = sorted(
        (parquet_base / "station=KLGA" / "date=2026-02-01" / "ch=world").glob("*.parquet")
    )
    assert len(part_files) == 3

    reader = ParquetReader(str(parquet_base))
    restored = reader.read_channel("2026-02-01", "world", station_id="KLGA")

    assert len(restored) == 5
    restored_dirs = [event["data"]["wind_dir"] for event in restored]
    assert restored_dirs.count(90) == 3
    assert restored_dirs.count("VRB") == 2


def test_compactor_reprocesses_partial_date_before_skipping(tmp_path):
    ndjson_base = tmp_path / "recordings"
    parquet_base = tmp_path / "parquet"

    _write_ndjson(
        ndjson_base,
        date_str="2026-02-01",
        channel="world",
        events=[{"ts_ingest_utc": "2026-02-01T00:00:00Z", "data": {"src": "METAR"}}],
    )
    _write_ndjson(
        ndjson_base,
        date_str="2026-02-01",
        channel="nowcast",
        events=[{"ts_ingest_utc": "2026-02-01T00:05:00Z", "data": {"winner": "40-41F"}}],
    )
    _write_ndjson(
        ndjson_base,
        date_str="2026-02-02",
        channel="world",
        events=[{"ts_ingest_utc": "2026-02-02T00:00:00Z", "data": {"src": "METAR"}}],
    )

    # Seed a partial compaction for 2026-02-01 so compact_all_pending must not skip it.
    seed = Compactor(str(ndjson_base), str(parquet_base), batch_size=1, max_dates_per_run=10)
    seed.compact_channel("2026-02-01", "world")

    compactor = Compactor(str(ndjson_base), str(parquet_base), batch_size=1, max_dates_per_run=1)
    result = compactor.compact_all_pending()

    assert result["compacted_dates"] == ["2026-02-01"]
    assert (parquet_base / "date=2026-02-01" / "ch=world" / "part-0000.parquet").exists()
    assert (parquet_base / "date=2026-02-01" / "ch=nowcast" / "part-0000.parquet").exists()
    assert not (parquet_base / "date=2026-02-02" / "ch=world" / "part-0000.parquet").exists()


def test_compactor_cleanup_old_ndjson_keeps_protected_channels(tmp_path):
    ndjson_base = tmp_path / "recordings"
    parquet_base = tmp_path / "parquet"
    _write_ndjson(
        ndjson_base,
        date_str="2000-01-01",
        channel="pws",
        events=[{"ts_ingest_utc": "2000-01-01T00:00:00Z", "data": {"temp_c": 10.0}}],
    )
    _write_ndjson(
        ndjson_base,
        date_str="2000-01-01",
        channel="world",
        events=[{"ts_ingest_utc": "2000-01-01T00:00:00Z", "data": {"src": "METAR"}}],
    )

    compactor = Compactor(
        str(ndjson_base),
        str(parquet_base),
        retention_days=14,
        keep_forever_channels={"pws"},
        max_dates_per_run=10,
    )
    deleted = compactor.cleanup_old_ndjson()

    assert deleted == 1
    assert (ndjson_base / "date=2000-01-01" / "ch=pws" / "events.ndjson").exists()
    assert not (ndjson_base / "date=2000-01-01" / "ch=world").exists()


def test_compactor_repairs_missing_station_output_before_marking_date_compacted(tmp_path):
    ndjson_base = tmp_path / "recordings"
    parquet_base = tmp_path / "parquet"
    _write_ndjson(
        ndjson_base,
        date_str="2026-02-03",
        channel="world",
        events=[
            {"station_id": "KLGA", "ts_ingest_utc": "2026-02-03T00:00:00Z", "data": {"src": "METAR"}},
            {"station_id": "KATL", "ts_ingest_utc": "2026-02-03T00:05:00Z", "data": {"src": "METAR"}},
        ],
    )

    seed = Compactor(str(ndjson_base), str(parquet_base), batch_size=10, max_dates_per_run=10)
    seed.compact_channel("2026-02-03", "world")

    missing_station_dir = parquet_base / "station=KATL" / "date=2026-02-03" / "ch=world"
    shutil.rmtree(missing_station_dir)

    compactor = Compactor(str(ndjson_base), str(parquet_base), batch_size=10, max_dates_per_run=1)
    assert compactor._is_channel_compacted("2026-02-03", "world") is False
    assert compactor._is_date_compacted("2026-02-03") is False

    result = compactor.compact_all_pending()

    assert result["compacted_dates"] == ["2026-02-03"]
    assert (parquet_base / "station=KLGA" / "date=2026-02-03" / "ch=world" / "part-0000.parquet").exists()
    assert (parquet_base / "station=KATL" / "date=2026-02-03" / "ch=world" / "part-0000.parquet").exists()
