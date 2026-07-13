"""Non-destructive migration inventory for pre point-level grounding assets."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import save_probe_points

MIGRATION_VERSION = "legacy-grounding-to-point-v1"


@dataclass(frozen=True)
class MigrationPaths:
    source: Path
    legacy_manifest: Path
    points: Path
    point_manifest: Path
    quarantine: Path
    smoke_index: Path
    report: Path


def default_paths(data_root: Path) -> MigrationPaths:
    grounded = data_root / "grounded"
    quarantine_dir = grounded / "quarantine"
    return MigrationPaths(
        source=grounded / "reversibility.jsonl",
        legacy_manifest=grounded / "MANIFEST.jsonl",
        points=grounded / "probe_points.jsonl",
        point_manifest=grounded / "POINT_MANIFEST.jsonl",
        quarantine=quarantine_dir / "legacy_rows.jsonl",
        smoke_index=quarantine_dir / "class_probe_smoke_index.jsonl",
        report=quarantine_dir / "migration-report.json",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


def _text_jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)


def _atomic_create_or_verify(path: Path, text: str) -> None:
    """Create a generated artifact, or verify an identical prior run."""
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(
                f"refusing to overwrite non-identical migration artifact {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def migrate_legacy_grounding(data_root: Path) -> dict[str, Any]:
    """Inventory legacy rows without converting any of them to formal points.

    Rows lacking even a legacy ``probe_id`` are quarantined.  Rows represented
    by the old manifest are indexed as class-level smoke probes.  Neither set
    has ``state_id``/``action_instance_id``/transition hashes, so formal point
    output is deliberately empty.
    """
    paths = default_paths(data_root)
    if not paths.source.exists():
        raise FileNotFoundError(paths.source)
    before_hash = _sha256(paths.source)
    body = _read_jsonl(paths.source)
    manifest = _read_jsonl(paths.legacy_manifest)
    manifest_ids = {r.get("probe_id") for r in manifest if r.get("probe_id")}

    quarantine: list[dict[str, Any]] = []
    smoke: list[dict[str, Any]] = []
    for line_no, row in enumerate(body, 1):
        probe_id = row.get("probe_id", "")
        common_reasons = [
            "missing_state_id", "missing_action_instance_id",
            "missing_point_transition_hashes",
        ]
        if not probe_id:
            reasons = ["missing_probe_id", *common_reasons]
            if row.get("label") == "IRREVERSIBLE":
                reasons.append(
                    "legacy_irreversible_without_budget_solver_or_point_identity")
            quarantine.append({
                "migration_version": MIGRATION_VERSION,
                "source": str(paths.source),
                "source_line": line_no,
                "classification": "legacy_quarantine",
                "reason_codes": reasons,
                "original": row,
            })
        else:
            smoke.append({
                "migration_version": MIGRATION_VERSION,
                "source": str(paths.source),
                "source_line": line_no,
                "probe_id": probe_id,
                "probe_name": row.get("probe_name", ""),
                "action_type": row.get("action_type", ""),
                "classification": "class_probe_smoke_only",
                "legacy_manifest_present": probe_id in manifest_ids,
                "formal_eligibility": "EXCLUDED",
                "reason_codes": common_reasons,
            })

    body_probe_ids = {r.get("probe_id") for r in body if r.get("probe_id")}
    report: dict[str, Any] = {
        "migration_version": MIGRATION_VERSION,
        "source": str(paths.source),
        "source_sha256_before": before_hash,
        "source_rows": len(body),
        "legacy_manifest_rows": len(manifest),
        "legacy_quarantine_rows": len(quarantine),
        "class_probe_smoke_rows": len(smoke),
        "formal_points_created": 0,
        "formal_policy": (
            "No legacy row has point-level state/action/transition provenance; "
            "none was upgraded or used as formal supervision."),
        "legacy_body_only_probe_ids": sorted(body_probe_ids - manifest_ids),
        "legacy_manifest_only_probe_ids": sorted(manifest_ids - body_probe_ids),
        "legacy_place_order_policy": (
            "Rows without point-level provenance are EXCLUDED/UNKNOWN for formal "
            "training; IRREVERSIBLE is not carried into the canonical ontology."),
        "rollback": [
            str(paths.quarantine), str(paths.smoke_index), str(paths.report),
            str(paths.points), str(paths.point_manifest),
        ],
    }

    _atomic_create_or_verify(paths.quarantine, _text_jsonl(quarantine))
    _atomic_create_or_verify(paths.smoke_index, _text_jsonl(smoke))
    # Materialize an honest empty formal body/manifest.  The schema writer
    # guarantees their 1:1 integrity and refuses to absorb legacy rows.
    save_probe_points([], paths.points, paths.point_manifest, append=True)
    after_hash = _sha256(paths.source)
    if after_hash != before_hash:
        raise RuntimeError("legacy source changed during non-destructive migration")
    report["source_sha256_after"] = after_hash
    _atomic_create_or_verify(
        paths.report,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return report
