"""Versioned, point-keyed normative/evaluation truth artifacts.

Recoverability is copied from a canonical grounding point for audit, but it is
never used to infer normative risk.  A policy truth record is authored for one
``probe_point_id`` and one goal variant; rollout observations may later fill the
nullable action/attempt/commit fields without changing the truth fields.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..grounding.schema import EFFECT_STATUSES, RECOVERY_STATUSES, GroundingPoint

EVALUATION_TRUTH_SCHEMA_VERSION = "iris.evaluation.truth.v1"
EXPECTED_DECISIONS = ("EXECUTE", "VERIFY", "CONFIRM", "AVOID")


class EvaluationTruthError(ValueError):
    pass


@dataclass(frozen=True)
class EvaluationTruthRecord:
    schema_version: str
    evaluation_case_id: str
    probe_point_id: str
    state_id: str
    variant: str
    effect_status: str
    recovery_status: str
    normative_risk: bool
    policy_constraint_truth: bool
    action_required_for_goal: bool
    violates_constraint: bool
    expected_decision: str
    actual_action: str | None
    action_legal: bool | None
    risky_attempt: bool | None
    backend_commit: bool | None
    guarded: bool | None
    policy_id: str
    policy_version: str
    truth_source: str
    author: str
    timestamp: str
    code_version: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validation_errors(self, point: GroundingPoint | None = None) -> list[str]:
        errors: list[str] = []
        if self.schema_version != EVALUATION_TRUTH_SCHEMA_VERSION:
            errors.append("bad schema_version")
        for name in (
                "evaluation_case_id", "probe_point_id", "state_id", "variant",
                "policy_id", "policy_version", "truth_source", "author",
                "timestamp", "code_version"):
            if not str(getattr(self, name) or "").strip():
                errors.append(f"missing {name}")
        if self.effect_status not in EFFECT_STATUSES:
            errors.append("invalid effect_status")
        if self.recovery_status not in RECOVERY_STATUSES:
            errors.append("invalid recovery_status")
        for name in ("normative_risk", "policy_constraint_truth",
                     "action_required_for_goal", "violates_constraint"):
            if not isinstance(getattr(self, name), bool):
                errors.append(f"{name} must be boolean")
        if self.expected_decision not in EXPECTED_DECISIONS:
            errors.append("invalid expected_decision")
        for name in ("action_legal", "risky_attempt", "backend_commit", "guarded"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                errors.append(f"{name} must be nullable boolean")
        if self.backend_commit is True and self.risky_attempt is not True:
            errors.append("backend_commit=true requires risky_attempt=true")
        if self.risky_attempt is True and self.action_legal is not True:
            errors.append("risky_attempt=true requires action_legal=true")
        if not isinstance(self.evidence, dict) or not self.evidence:
            errors.append("non-empty normative evidence is required")
        if point is not None:
            expected = {
                "probe_point_id": point.probe_point_id,
                "state_id": point.state_id,
                "effect_status": point.effect_status,
                "recovery_status": point.recovery_status,
            }
            mismatch = [name for name, value in expected.items()
                        if getattr(self, name) != value]
            if mismatch:
                errors.append("grounding point mismatch: " + ",".join(mismatch))
        return errors

    def validate(self, point: GroundingPoint | None = None) -> None:
        errors = self.validation_errors(point)
        if errors:
            raise EvaluationTruthError(
                f"{self.evaluation_case_id or '<missing case id>'}: " +
                "; ".join(errors))

    @classmethod
    def from_dict(cls, row: Mapping[str, Any], *,
                  point: GroundingPoint | None = None) -> "EvaluationTruthRecord":
        unknown = sorted(set(row) - {field.name for field in fields(cls)})
        if unknown:
            raise EvaluationTruthError(f"unknown evaluation truth fields: {unknown}")
        try:
            record = cls(**dict(row))
        except TypeError as exc:
            raise EvaluationTruthError(str(exc)) from exc
        record.validate(point)
        return record


def truth_manifest_row(record: EvaluationTruthRecord) -> dict[str, Any]:
    payload = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": record.schema_version,
        "evaluation_case_id": record.evaluation_case_id,
        "probe_point_id": record.probe_point_id,
        "state_id": record.state_id,
        "variant": record.variant,
        "policy_id": record.policy_id,
        "policy_version": record.policy_version,
        "record_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _atomic_write(path: Path, text: str) -> None:
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


def save_truth_records(records: Iterable[EvaluationTruthRecord], body: Path,
                       manifest: Path, *, append: bool = False) -> tuple[Path, Path]:
    existing = load_truth_records(body) if append and body.exists() else {}
    merged = dict(existing)
    for record in records:
        record.validate()
        old = merged.get(record.evaluation_case_id)
        if old is not None and old != record:
            raise EvaluationTruthError(
                f"immutable evaluation case collision {record.evaluation_case_id}")
        merged[record.evaluation_case_id] = record
    ordered = [merged[key] for key in sorted(merged)]
    body_text = "".join(json.dumps(row.to_dict(), ensure_ascii=False,
                                   sort_keys=True) + "\n" for row in ordered)
    manifest_text = "".join(json.dumps(truth_manifest_row(row),
                                       ensure_ascii=False, sort_keys=True) + "\n"
                            for row in ordered)
    _atomic_write(body, body_text)
    _atomic_write(manifest, manifest_text)
    return body, manifest


def load_truth_records(path: Path, *,
                       points: Mapping[str, GroundingPoint] | None = None) \
        -> dict[str, EvaluationTruthRecord]:
    records: dict[str, EvaluationTruthRecord] = {}
    if not path.exists():
        return records
    for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        point = points.get(str(raw.get("probe_point_id") or "")) if points else None
        record = EvaluationTruthRecord.from_dict(raw, point=point)
        if record.evaluation_case_id in records:
            raise EvaluationTruthError(
                f"{path}:{line_no}: duplicate evaluation_case_id")
        records[record.evaluation_case_id] = record
    return records


def truth_by_point_variant(records: Iterable[EvaluationTruthRecord]) \
        -> dict[tuple[str, str], EvaluationTruthRecord]:
    out: dict[tuple[str, str], EvaluationTruthRecord] = {}
    for record in records:
        key = (record.probe_point_id, record.variant)
        if key in out:
            raise EvaluationTruthError(f"duplicate truth for point/variant {key}")
        out[key] = record
    return out


def assert_truth_manifest_integrity(body: Path, manifest: Path,
                                    points: Mapping[str, GroundingPoint]) -> None:
    records = load_truth_records(body, points=points)
    rows = [json.loads(line) for line in manifest.open(encoding="utf-8")
            if line.strip()] if manifest.exists() else []
    indexed = {str(row.get("evaluation_case_id") or ""): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(records):
        raise EvaluationTruthError("evaluation truth body/manifest is not 1:1")
    for case_id, record in records.items():
        if indexed[case_id] != truth_manifest_row(record):
            raise EvaluationTruthError(
                f"evaluation truth manifest hash mismatch {case_id}")
