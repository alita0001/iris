#!/usr/bin/env python3
"""Index ambiguous legacy trajectory lineage without mutating raw assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8")
            if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            if isinstance(value, list):
                for row in value:
                    handle.write(json.dumps(row, ensure_ascii=False,
                                            sort_keys=True) + "\n")
            else:
                json.dump(value, handle, ensure_ascii=False, sort_keys=True,
                          indent=2)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_legacy_lineage_quarantine(data_root: Path) -> dict:
    root = Path(data_root)
    meta_path = root / "raw" / "trajectories_meta.jsonl"
    state_paths = sorted((root / "raw" / "state_bank").glob("*.jsonl"))
    raw_paths = sorted((root / "raw" / "trajectories").glob("*.jsonl"))
    metas = _rows(meta_path)
    counts = Counter(str(row.get("trajectory_id") or "") for row in metas)
    occurrences: Counter[str] = Counter()
    indexed = []
    n_transactional = 0
    for line_no, row in enumerate(metas, 1):
        trajectory_id = str(row.get("trajectory_id") or "")
        occurrences[trajectory_id] += 1
        reasons = []
        if not trajectory_id:
            reasons.append("missing_trajectory_id")
        if trajectory_id and counts[trajectory_id] > 1:
            reasons.append("duplicate_trajectory_id")
        if not str(row.get("run_id") or ""):
            reasons.append("missing_immutable_run_id")
        if not str(row.get("environment_origin") or ""):
            reasons.append("missing_environment_origin")
        # A new transactional row with a unique immutable trajectory id,
        # explicit run id and environment provenance is not legacy merely
        # because it shares the append-only inventory file. Leave it outside
        # quarantine; the formal lineage audit verifies its run closure when a
        # released sample actually references it.
        if not reasons:
            n_transactional += 1
            continue
        indexed.append({
            "schema_version": "iris.legacy_lineage_quarantine.v1",
            "source": str(meta_path), "source_line": line_no,
            "trajectory_id": trajectory_id,
            "occurrence": occurrences[trajectory_id],
            "formal_eligible": False,
            "reason_codes": reasons,
            "row_sha256": hashlib.sha256(json.dumps(
                row, ensure_ascii=False, sort_keys=True,
                separators=(",", ":")).encode("utf-8")).hexdigest(),
        })
    sources = [meta_path, *raw_paths, *state_paths]
    report = {
        "schema_version": "iris.legacy_lineage_quarantine.report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "index_only_non_destructive",
        "n_meta_rows": len(metas),
        "n_legacy_rows": len(indexed),
        "n_transactional_rows_excluded": n_transactional,
        "n_unique_trajectory_ids": len([key for key in counts if key]),
        "n_duplicate_trajectory_ids": sum(
            1 for key, count in counts.items() if key and count > 1),
        "n_rows_with_duplicate_id": sum(
            count for key, count in counts.items() if key and count > 1),
        "formal_eligible": 0,
        "source_assets": [{
            "path": str(path), "sha256": _sha(path),
            "bytes": path.stat().st_size if path.exists() else 0,
        } for path in sources],
        "rollback": "delete only data/raw/quarantine/legacy_lineage_*; sources unchanged",
        "note": (f"{len(indexed)} historical/ambiguous rows are quarantine "
                 "observations, not independent formal runs; "
                 f"{n_transactional} transactional rows remain outside quarantine."),
    }
    out = root / "raw" / "quarantine"
    _atomic_json(out / "legacy_lineage_rows.jsonl", indexed)
    _atomic_json(out / "legacy_lineage_manifest.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(json.dumps(build_legacy_lineage_quarantine(args.data_root),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
