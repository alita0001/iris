#!/usr/bin/env python3
"""Fingerprint every dirty worktree asset needed to reproduce an audit run.

The manifest never embeds file contents.  It records the base commit plus the
status, byte size and SHA-256 of each modified/untracked file.  The output file
itself is excluded to avoid a self-referential hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = "iris.worktree_manifest.v1"


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True,
    ).stdout


def _status_entries(root: Path) -> list[tuple[str, str]]:
    payload = _git(
        root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    fields = payload.split(b"\0")
    entries: list[tuple[str, str]] = []
    index = 0
    while index < len(fields):
        field = fields[index]
        index += 1
        if not field:
            continue
        if len(field) < 4 or field[2:3] != b" ":
            raise RuntimeError("unexpected git status --porcelain record")
        status = field[:2].decode("ascii")
        path = field[3:].decode("utf-8", errors="surrogateescape")
        entries.append((status, path))
        if "R" in status or "C" in status:
            # In -z mode a rename/copy record is followed by its source path.
            if index >= len(fields) or not fields[index]:
                raise RuntimeError("truncated git rename/copy status record")
            index += 1
    return entries


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def audit(root: Path, output: Path) -> dict:
    root = root.resolve()
    output = output.resolve()
    try:
        output_rel = output.relative_to(root).as_posix()
    except ValueError:
        output_rel = ""
    rows = []
    for status, relative in _status_entries(root):
        if relative == output_rel:
            continue
        path = root / relative
        row: dict[str, object] = {
            "path": relative,
            "status": status,
            "exists": path.exists() or path.is_symlink(),
        }
        if path.is_file() and not path.is_symlink():
            row["sha256"], row["size_bytes"] = _hash_file(path)
        elif path.is_symlink():
            target = os.readlink(path)
            row["symlink_target"] = target
            row["sha256"] = hashlib.sha256(
                target.encode("utf-8", errors="surrogateescape")).hexdigest()
            row["size_bytes"] = len(target.encode(
                "utf-8", errors="surrogateescape"))
        rows.append(row)
    rows.sort(key=lambda item: str(item["path"]))
    canonical = json.dumps(
        rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    head = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(root),
        "base_head": head,
        "dirty": bool(rows),
        "entry_count": len(rows),
        "worktree_sha256": hashlib.sha256(canonical).hexdigest(),
        "excluded_outputs": [output_rel] if output_rel else [],
        "entries": rows,
    }


def _atomic_write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2,
                      sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path,
        default=Path("docs/audit/IRIS-worktree-manifest.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    report = audit(root, output)
    _atomic_write(output, report)
    print(json.dumps({key: report[key] for key in (
        "schema_version", "base_head", "dirty", "entry_count",
        "worktree_sha256")}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
