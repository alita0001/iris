"""Reversible, index-only quarantine for pre-v3 training artifacts.

Source JSONL files are never moved or rewritten.  The manifest records each
source line, its byte hash and every formal-export exclusion reason.  Removing
the generated quarantine directory is therefore the complete rollback.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SOURCES = (
    "sft/revact_sft.jsonl",
    "sft/revact_sft_multiturn.jsonl",
    "sft/revact_sft_distilled.jsonl",
    "dpo/revact_dpo.jsonl",
    "dpo/revact_dpo_multiturn.jsonl",
)


def _read_meta(data_root: Path) -> tuple[dict[str, list[dict]], str]:
    path = data_root / "raw" / "trajectories_meta.jsonl"
    by_id: dict[str, list[dict]] = defaultdict(list)
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                by_id[str(row.get("trajectory_id") or "")].append(row)
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""
    return dict(by_id), digest


def exclusion_reasons(row: dict[str, Any], *, family: str,
                      trajectory_rows: list[dict]) -> list[str]:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    reasons: list[str] = []
    if meta.get("formal_dataset") is not True:
        reasons.append("formal_dataset_not_true")
    trajectory_id = str(meta.get("trajectory_id") or "")
    origin = str(meta.get("environment_origin") or "").lower()
    if meta.get("is_mock") is True or origin == "mock" or trajectory_id.startswith("mock."):
        reasons.append("mock_environment")
    for field in ("probe_point_id", "state_id", "action_instance_id"):
        if not meta.get(field):
            reasons.append(f"missing_{field}")
    if meta.get("history_source") != "trajectory":
        reasons.append("non_trajectory_history")
    if meta.get("collector_success") is not True:
        reasons.append("collector_success_not_proven")
    if trajectory_id:
        if len(trajectory_rows) > 1:
            reasons.append("ambiguous_duplicate_trajectory_lineage")
        elif trajectory_rows and trajectory_rows[0].get("success") is not True:
            reasons.append("failed_collector_trajectory")
        elif not trajectory_rows and not trajectory_id.startswith("mock."):
            reasons.append("trajectory_meta_missing")
    if family == "sft":
        if meta.get("prediction_source") != "probe_transition":
            reasons.append("prediction_not_probe_transition")
        if meta.get("undo_source") != "probe_point_id":
            reasons.append("undo_not_probe_point")
    else:
        if meta.get("negative_source") not in {"legal_candidate", "on_policy"}:
            reasons.append("non_deployment_negative_source")
    return sorted(set(reasons))


def build_quarantine_index(data_root: Path, *,
                           sources: Iterable[str] = DEFAULT_SOURCES) -> dict:
    data_root = Path(data_root)
    train_root = data_root / "train"
    trajectory_map, trajectory_meta_sha256 = _read_meta(data_root)
    entries: list[dict] = []
    source_files: list[dict] = []
    reason_counts: Counter[str] = Counter()

    for relative in sources:
        path = train_root / relative
        family = "dpo" if relative.startswith("dpo/") else "sft"
        if not path.exists():
            source_files.append({"path": relative, "exists": False, "sha256": ""})
            continue
        source_files.append({
            "path": relative,
            "exists": True,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
        for line_no, raw in enumerate(path.open(encoding="utf-8"), 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
            trajectory_id = str(meta.get("trajectory_id") or "")
            reasons = exclusion_reasons(
                row, family=family,
                trajectory_rows=trajectory_map.get(trajectory_id, []),
            )
            reason_counts.update(reasons)
            entries.append({
                "schema_version": "iris.quarantine.v1",
                "source_file": relative,
                "source_line": line_no,
                "record_id": row.get("sample_id") or row.get("pair_id") or
                             f"{relative}:{line_no}",
                "record_sha256": hashlib.sha256(raw.rstrip("\n").encode("utf-8")).hexdigest(),
                "family": family,
                "trajectory_id": trajectory_id or None,
                "reasons": reasons,
                "formal_eligible": not reasons,
            })

    return {
        "schema_version": "iris.quarantine_report.v1",
        "source_files": source_files,
        "trajectory_meta_sha256": trajectory_meta_sha256,
        "n_indexed": len(entries),
        "n_formal_eligible": sum(item["formal_eligible"] for item in entries),
        "n_quarantined": sum(not item["formal_eligible"] for item in entries),
        "reason_counts": dict(sorted(reason_counts.items())),
        "entries": entries,
        "migration_mode": "index_only_source_files_unchanged",
        "rollback": "delete data/train/quarantine; source JSONL files were not changed",
    }


def write_quarantine_index(data_root: Path) -> dict:
    report = build_quarantine_index(data_root)
    out = Path(data_root) / "train" / "quarantine"
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "legacy_training_manifest.jsonl"
    manifest.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in report.pop("entries")), encoding="utf-8")
    report_path = out / "migration-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                      sort_keys=True) + "\n", encoding="utf-8")
    return {**report, "manifest": str(manifest), "report": str(report_path)}
