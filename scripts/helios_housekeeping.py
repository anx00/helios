#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional


UTC = timezone.utc
DATE_DIR_RE = re.compile(r"^date=(\d{4}-\d{2}-\d{2})$")
WORLD_TAPE_RE = re.compile(r"^world_tape_(\d{4}-\d{2}-\d{2})\.ndjson$")
GENERIC_LOG_EXTS = {".log", ".out", ".err", ".gz"}


@dataclass
class DeletionCandidate:
    path: Path
    kind: str  # "file" | "dir"
    reason: str
    size_bytes: int


@dataclass
class CleanupStats:
    deleted_files: int = 0
    deleted_dirs: int = 0
    failed: int = 0
    bytes_freed: int = 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean up old HELIOS logs and stored data with retention by age/date."
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="HELIOS repo root (defaults to parent of this script).",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=int(os.getenv("HELIOS_RETENTION_DAYS", "7")),
        help="Delete data/logs older than this many days (default: 7).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without deleting anything.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each deletion candidate.",
    )
    return parser.parse_args()


def _repo_root_from_args(args: argparse.Namespace) -> Path:
    if args.repo_root:
        return Path(args.repo_root).resolve()
    return Path(__file__).resolve().parents[1]


def _safe_path_under(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except FileNotFoundError:
                continue
    except FileNotFoundError:
        return 0
    return total


def _file_size_bytes(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _parse_date_from_dir_name(name: str) -> Optional[datetime.date]:
    match = DATE_DIR_RE.match(name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d").date()
    except ValueError:
        return None


def _parse_date_from_world_tape_name(name: str) -> Optional[datetime.date]:
    match = WORLD_TAPE_RE.match(name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d").date()
    except ValueError:
        return None


def _iter_recordings_date_dirs(recordings_root: Path) -> Iterable[Path]:
    if not recordings_root.exists():
        return []
    return [p for p in recordings_root.iterdir() if p.is_dir() and DATE_DIR_RE.match(p.name)]


def _iter_parquet_date_dirs(parquet_root: Path) -> Iterable[Path]:
    if not parquet_root.exists():
        return []
    seen = set()
    out: List[Path] = []
    for p in parquet_root.rglob("date=*"):
        try:
            if not p.is_dir():
                continue
        except FileNotFoundError:
            continue
        rp = p.resolve()
        if rp in seen:
            continue
        if DATE_DIR_RE.match(p.name):
            seen.add(rp)
            out.append(p)
    # Delete deeper paths first to avoid double work on parent traversals
    out.sort(key=lambda x: len(x.parts), reverse=True)
    return out


def _iter_old_log_files(logs_root: Path, cutoff_dt: datetime, cutoff_date) -> Iterable[DeletionCandidate]:
    if not logs_root.exists():
        return []

    candidates: List[DeletionCandidate] = []
    for p in logs_root.iterdir():
        if not p.is_file():
            continue

        # Daily world tape logs are already date-partitioned by filename. Delete by embedded date.
        tape_date = _parse_date_from_world_tape_name(p.name)
        if tape_date is not None:
            if tape_date < cutoff_date:
                candidates.append(
                    DeletionCandidate(
                        path=p,
                        kind="file",
                        reason=f"world_tape_date<{cutoff_date.isoformat()}",
                        size_bytes=_file_size_bytes(p),
                    )
                )
            continue

        suffix = p.suffix.lower()
        if suffix not in GENERIC_LOG_EXTS:
            continue

        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=UTC)
        except FileNotFoundError:
            continue

        if mtime < cutoff_dt:
            candidates.append(
                DeletionCandidate(
                    path=p,
                    kind="file",
                    reason=f"log_mtime<{cutoff_dt.isoformat()}",
                    size_bytes=_file_size_bytes(p),
                )
            )
    return candidates


def _collect_candidates(repo_root: Path, retention_days: int) -> List[DeletionCandidate]:
    now_utc = datetime.now(UTC)
    cutoff_dt = now_utc - timedelta(days=max(1, retention_days))
    cutoff_date = cutoff_dt.date()

    logs_root = repo_root / "logs"
    recordings_root = repo_root / "data" / "recordings"
    parquet_root = repo_root / "data" / "parquet"

    candidates: List[DeletionCandidate] = []

    # 1) File logs in logs/ older than retention.
    candidates.extend(_iter_old_log_files(logs_root, cutoff_dt, cutoff_date))

    # 2) NDJSON recordings partitioned by date.
    for p in _iter_recordings_date_dirs(recordings_root):
        file_date = _parse_date_from_dir_name(p.name)
        if file_date and file_date < cutoff_date:
            candidates.append(
                DeletionCandidate(
                    path=p,
                    kind="dir",
                    reason=f"recordings_date<{cutoff_date.isoformat()}",
                    size_bytes=_dir_size_bytes(p),
                )
            )

    # 3) Parquet partitions (date dirs may be nested under station=...).
    for p in _iter_parquet_date_dirs(parquet_root):
        file_date = _parse_date_from_dir_name(p.name)
        if file_date and file_date < cutoff_date:
            candidates.append(
                DeletionCandidate(
                    path=p,
                    kind="dir",
                    reason=f"parquet_date<{cutoff_date.isoformat()}",
                    size_bytes=_dir_size_bytes(p),
                )
            )

    # Deduplicate by resolved path (dir deletions can subsume nested file matches in exotic layouts).
    unique: dict[Path, DeletionCandidate] = {}
    for c in candidates:
        try:
            key = c.path.resolve()
        except FileNotFoundError:
            continue
        prev = unique.get(key)
        if prev is None or (c.kind == "dir" and prev.kind == "file"):
            unique[key] = c

    # Prefer deleting directories before files if same subtree is selected.
    ordered = sorted(
        unique.values(),
        key=lambda c: (0 if c.kind == "dir" else 1, -len(c.path.parts), str(c.path)),
    )
    return ordered


def _delete_candidate(root: Path, candidate: DeletionCandidate, dry_run: bool) -> bool:
    if not _safe_path_under(root, candidate.path):
        print(f"[SKIP] Unsafe path outside repo: {candidate.path}")
        return False
    if not candidate.path.exists():
        return True

    if dry_run:
        return True

    if candidate.kind == "dir":
        shutil.rmtree(candidate.path, ignore_errors=False)
    else:
        candidate.path.unlink(missing_ok=True)
    return True


def _prune_empty_dirs(root: Path, dry_run: bool, verbose: bool) -> int:
    removed = 0
    for base in [root / "data" / "parquet", root / "data" / "recordings"]:
        if not base.exists():
            continue
        # Bottom-up traversal.
        for p in sorted((x for x in base.rglob("*") if x.is_dir()), key=lambda x: len(x.parts), reverse=True):
            try:
                if any(p.iterdir()):
                    continue
            except FileNotFoundError:
                continue
            if not _safe_path_under(root, p):
                continue
            if verbose:
                print(f"[EMPTY] {p}")
            if not dry_run:
                try:
                    p.rmdir()
                    removed += 1
                except OSError:
                    continue
            else:
                removed += 1
    return removed


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root_from_args(args)
    if not repo_root.exists():
        print(f"[ERROR] Repo root not found: {repo_root}", file=sys.stderr)
        return 2

    candidates = _collect_candidates(repo_root, args.retention_days)
    stats = CleanupStats()

    mode = "DRY-RUN" if args.dry_run else "DELETE"
    print(
        f"[{mode}] HELIOS housekeeping in {repo_root} | retention_days={args.retention_days} | "
        f"candidates={len(candidates)}"
    )

    for c in candidates:
        if args.verbose or args.dry_run:
            size_mb = c.size_bytes / (1024 * 1024)
            print(f"  - {c.kind.upper():3} {c.path} | {size_mb:.2f} MiB | {c.reason}")

        try:
            ok = _delete_candidate(repo_root, c, args.dry_run)
            if not ok:
                stats.failed += 1
                continue
            if c.kind == "dir":
                stats.deleted_dirs += 1
            else:
                stats.deleted_files += 1
            stats.bytes_freed += c.size_bytes
        except Exception as exc:
            stats.failed += 1
            print(f"[ERROR] Failed to delete {c.path}: {exc}", file=sys.stderr)

    empty_pruned = _prune_empty_dirs(repo_root, dry_run=args.dry_run, verbose=args.verbose)

    print(
        f"[{mode}] Summary: deleted_files={stats.deleted_files} deleted_dirs={stats.deleted_dirs} "
        f"empty_dirs_pruned={empty_pruned} failed={stats.failed} "
        f"estimated_freed={stats.bytes_freed / (1024 * 1024):.2f} MiB"
    )
    return 1 if stats.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
