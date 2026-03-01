import importlib.util
from datetime import datetime, timezone
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_housekeeping_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "helios_housekeeping.py"
    spec = importlib.util.spec_from_file_location("helios_housekeeping_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_channel_file(base_dir: Path, *, date_str: str, channel: str, file_name: str):
    channel_dir = base_dir / f"date={date_str}" / f"ch={channel}"
    channel_dir.mkdir(parents=True, exist_ok=True)
    (channel_dir / file_name).write_text("{}", encoding="utf-8")
    return channel_dir


def test_collect_candidates_supports_channel_specific_retention(tmp_path, monkeypatch):
    hk = _load_housekeeping_module()

    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            current = cls(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)
            if tz is None:
                return current.replace(tzinfo=None)
            return current.astimezone(tz)

    repo_root = tmp_path / "repo"
    recordings_root = repo_root / "data" / "recordings"
    parquet_root = repo_root / "data" / "parquet" / "station=KLGA"

    l2_recordings = _write_channel_file(
        recordings_root,
        date_str="2026-02-18",
        channel="l2_snap",
        file_name="events.ndjson",
    )
    _write_channel_file(
        recordings_root,
        date_str="2026-02-18",
        channel="world",
        file_name="events.ndjson",
    )
    l2_parquet_recordings_guard = _write_channel_file(
        parquet_root,
        date_str="2026-02-18",
        channel="l2_snap",
        file_name="part-0000.parquet",
    )
    l2_parquet = _write_channel_file(
        parquet_root,
        date_str="2026-02-15",
        channel="l2_snap",
        file_name="part-0000.parquet",
    )
    _write_channel_file(
        parquet_root,
        date_str="2026-02-15",
        channel="world",
        file_name="part-0000.parquet",
    )

    monkeypatch.setattr(hk, "datetime", _FrozenDateTime)
    monkeypatch.setenv("HELIOS_RECORDINGS_RETENTION_DAYS", "30")
    monkeypatch.setenv("HELIOS_PARQUET_RETENTION_DAYS", "60")
    monkeypatch.setenv("HELIOS_RECORDINGS_CHANNEL_RETENTION_DAYS", "l2_snap:1")
    monkeypatch.setenv("HELIOS_PARQUET_CHANNEL_RETENTION_DAYS", "l2_snap:3")

    candidates = hk._collect_candidates(repo_root, retention_days=90)
    paths = {c.path for c in candidates}

    assert l2_recordings in paths
    assert l2_parquet in paths
    assert l2_parquet_recordings_guard not in paths
    assert (recordings_root / "date=2026-02-18" / "ch=world") not in paths
    assert (parquet_root / "date=2026-02-15" / "ch=world") not in paths


def test_collect_candidates_keeps_recordings_without_parquet_snapshot(tmp_path, monkeypatch):
    hk = _load_housekeeping_module()

    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            current = cls(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)
            if tz is None:
                return current.replace(tzinfo=None)
            return current.astimezone(tz)

    repo_root = tmp_path / "repo"
    recordings_root = repo_root / "data" / "recordings"

    date_dir = recordings_root / "date=2026-02-10"
    (date_dir / "ch=world").mkdir(parents=True, exist_ok=True)
    (date_dir / "ch=world" / "events.ndjson").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(hk, "datetime", _FrozenDateTime)
    monkeypatch.setenv("HELIOS_RECORDINGS_RETENTION_DAYS", "3")
    monkeypatch.setenv("HELIOS_HOUSEKEEPING_REQUIRE_PARQUET_FOR_RECORDINGS_DELETE", "1")

    candidates = hk._collect_candidates(repo_root, retention_days=3)

    assert all(c.path != date_dir for c in candidates)
