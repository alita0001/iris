#!/usr/bin/env python3
"""Reversibly remove one non-formal collection run from active raw assets.

The collector appends meta/state rows to shared JSONL files and writes one raw
trajectory per physical run.  A provenance-invalid or failed run therefore
cannot be quarantined by moving only its manifest.  This migration snapshots
the complete pre-migration shared files, atomically removes only rows whose
``run_id`` matches, and moves run-owned artifacts beneath a quarantine root.

Dry-run is the default.  ``--execute`` performs the migration; ``--rollback``
restores byte-identical backups only when no post-migration file has changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "iris.collection-run-quarantine.v1"


class CollectionRunQuarantineError(ValueError):
    """The requested migration is ambiguous or no longer rollback-safe."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise CollectionRunQuarantineError(
            f"artifact escapes data root: {path}") from exc


def _jsonl_partition(path: Path, run_id: str) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    removed: list[str] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(
            keepends=True), 1):
        if not line.strip():
            kept.append(line)
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CollectionRunQuarantineError(
                f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if str(row.get("run_id") or "") == run_id:
            removed.append(line)
        else:
            kept.append(line)
    return kept, removed


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _discover(root: Path, run_id: str) -> dict[str, Any]:
    data_root = root / "data"
    manifest = data_root / "manifests" / "collection_runs" / f"{run_id}.json"
    if not manifest.is_file():
        raise CollectionRunQuarantineError(
            f"active collection manifest is absent: {manifest}")
    try:
        manifest_row = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CollectionRunQuarantineError(
            f"invalid collection manifest: {exc}") from exc
    if manifest_row.get("run_id") != run_id:
        raise CollectionRunQuarantineError("collection manifest run_id mismatch")
    if manifest_row.get("status") not in {"COMPLETE", "FAILED"}:
        raise CollectionRunQuarantineError(
            "only terminal COMPLETE/FAILED runs can be quarantined")

    shared: list[dict[str, Any]] = []
    candidates = [data_root / "raw" / "trajectories_meta.jsonl"]
    candidates.extend(sorted((data_root / "raw" / "state_bank").glob("*.jsonl")))
    for path in candidates:
        if not path.is_file():
            continue
        kept, removed = _jsonl_partition(path, run_id)
        if removed:
            shared.append({
                "path": path,
                "kept": kept,
                "removed": removed,
                "before_sha256": _sha256(path),
            })

    owned: set[Path] = set()
    for item in shared:
        if item["path"].name != "trajectories_meta.jsonl":
            continue
        for line in item["removed"]:
            row = json.loads(line)
            for field in ("raw_artifact", "policy_attempt_artifact"):
                value = str(row.get(field) or "")
                if value:
                    candidate = (data_root / value).resolve()
                    _relative(candidate, data_root)
                    if candidate.is_file():
                        owned.add(candidate)
    for directory in (data_root / "raw" / "trajectories",
                      data_root / "raw" / "policy_attempts"):
        if directory.exists():
            owned.update(path.resolve() for path in directory.glob(
                f"*__run_{run_id}.jsonl") if path.is_file())

    return {
        "data_root": data_root,
        "manifest": manifest.resolve(),
        "manifest_row": manifest_row,
        "shared": shared,
        "owned": sorted(owned),
    }


def quarantine_collection_run(root: Path, run_id: str, reason: str, *,
                              execute: bool = False) -> dict[str, Any]:
    root = Path(root).resolve()
    if not run_id or any(char not in
                         "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
                         for char in run_id):
        raise CollectionRunQuarantineError("run_id must be path-safe")
    if not str(reason or "").strip():
        raise CollectionRunQuarantineError("a non-empty reason is required")
    discovered = _discover(root, run_id)
    data_root: Path = discovered["data_root"]
    qdir = data_root / "raw" / "quarantine" / "collection_runs" / run_id
    if qdir.exists():
        raise CollectionRunQuarantineError(
            f"quarantine destination already exists: {qdir}")

    preview = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "reason": str(reason).strip(),
        "execute": bool(execute),
        "manifest_status": discovered["manifest_row"].get("status"),
        "shared_files": [
            {"path": _relative(item["path"], root),
             "removed_rows": len(item["removed"]),
             "before_sha256": item["before_sha256"]}
            for item in discovered["shared"]
        ],
        "owned_artifacts": [
            _relative(path, root) for path in discovered["owned"]],
        "counts_as_formal": False,
    }
    if not execute:
        return preview

    qdir.mkdir(parents=True, exist_ok=False)
    changes: list[dict[str, Any]] = []
    moves: list[dict[str, Any]] = []
    try:
        for item in discovered["shared"]:
            source: Path = item["path"]
            relative = source.relative_to(data_root)
            backup = qdir / "backups" / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, backup)
            if _sha256(backup) != item["before_sha256"]:
                raise CollectionRunQuarantineError(
                    f"backup hash mismatch for {source}")
            _atomic_text(source, "".join(item["kept"]))
            changes.append({
                "path": _relative(source, root),
                "backup_path": _relative(backup, root),
                "before_sha256": item["before_sha256"],
                "after_sha256": _sha256(source),
                "removed_rows": len(item["removed"]),
            })

        move_sources = [discovered["manifest"], *discovered["owned"]]
        for source in move_sources:
            if not source.is_file():
                raise CollectionRunQuarantineError(
                    f"run-owned artifact disappeared: {source}")
            relative = source.relative_to(data_root)
            target = qdir / "artifacts" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = _sha256(source)
            os.replace(source, target)
            if _sha256(target) != digest:
                raise CollectionRunQuarantineError(
                    f"moved artifact hash mismatch for {source}")
            moves.append({
                "source_path": _relative(source, root),
                "quarantine_path": _relative(target, root),
                "sha256": digest,
            })

        report = {
            **preview,
            "execute": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "shared_file_changes": changes,
            "moved_artifacts": moves,
            "rollback_guard": (
                "Rollback is allowed only while every active shared file still "
                "matches after_sha256 and every quarantined artifact matches sha256."),
        }
        report_path = qdir / "MIGRATION_REPORT.json"
        _atomic_text(report_path, json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        report["report_path"] = _relative(report_path, root)
        return report
    except Exception:
        # Never attempt an implicit rollback: a partial migration remains
        # inspectable under qdir and requires the explicit hash-guarded path.
        raise


def rollback_collection_run(root: Path, run_id: str) -> dict[str, Any]:
    root = Path(root).resolve()
    qdir = root / "data" / "raw" / "quarantine" / "collection_runs" / run_id
    report_path = qdir / "MIGRATION_REPORT.json"
    if not report_path.is_file():
        raise CollectionRunQuarantineError(
            f"quarantine migration report is absent: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != SCHEMA_VERSION or \
            report.get("run_id") != run_id:
        raise CollectionRunQuarantineError("quarantine report identity mismatch")

    for change in report.get("shared_file_changes") or []:
        active = root / change["path"]
        if not active.is_file() or _sha256(active) != change["after_sha256"]:
            raise CollectionRunQuarantineError(
                f"rollback refused: active shared file changed: {active}")
    for move in report.get("moved_artifacts") or []:
        source = root / move["source_path"]
        target = root / move["quarantine_path"]
        if source.exists() or not target.is_file() or \
                _sha256(target) != move["sha256"]:
            raise CollectionRunQuarantineError(
                f"rollback refused: moved artifact state changed: {source}")

    for change in report.get("shared_file_changes") or []:
        active = root / change["path"]
        backup = root / change["backup_path"]
        if _sha256(backup) != change["before_sha256"]:
            raise CollectionRunQuarantineError(
                f"rollback backup hash mismatch: {backup}")
        _atomic_text(active, backup.read_text(encoding="utf-8"))
    for move in report.get("moved_artifacts") or []:
        source = root / move["source_path"]
        target = root / move["quarantine_path"]
        source.parent.mkdir(parents=True, exist_ok=True)
        os.replace(target, source)
    marker = qdir / "ROLLED_BACK.json"
    _atomic_text(marker, json.dumps({
        "schema_version": "iris.collection-run-quarantine-rollback.v1",
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, indent=2, sort_keys=True) + "\n")
    return {"run_id": run_id, "rolled_back": True,
            "marker": _relative(marker, root)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-id", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--rollback", action="store_true")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()
    if args.rollback:
        result = rollback_collection_run(Path(args.root), args.run_id)
    else:
        result = quarantine_collection_run(
            Path(args.root), args.run_id, args.reason, execute=args.execute)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
