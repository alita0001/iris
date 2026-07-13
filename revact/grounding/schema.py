"""Canonical point-level grounding schema for IRIS.

The old ``reversibility.jsonl`` file stores class-level smoke-probe results and
mixes effect, recovery and normative risk in one label.  It remains readable
for audit and UI compatibility, but it is not a source of formal supervision.

This module is the single source of truth for new grounding artifacts.  The
ontology is deliberately orthogonal:

* ``effect_status`` says whether the probed action changed the measured state;
* ``recovery_status`` says what the recorded solver set achieved within budget;
* normative safety/risk is defined by the evaluation truth schema, not here.

Positive recovery is constructive evidence.  A failed search is represented as
``NOT_RECOVERED_WITHIN_BUDGET`` and never upgraded to mathematical
``IRREVERSIBLE``.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Iterable, Optional

GROUNDING_SCHEMA_VERSION = "iris.grounding.point.v1"

EFFECT_CHANGED = "CHANGED"
EFFECT_NO_EFFECT = "NO_EFFECT"
EFFECT_UNKNOWN = "UNKNOWN"
EFFECT_STATUSES = (EFFECT_CHANGED, EFFECT_NO_EFFECT, EFFECT_UNKNOWN)

RECOVERY_RECOVERED = "RECOVERED"
RECOVERY_PARTIAL = "PARTIALLY_RECOVERED"
RECOVERY_NOT_WITHIN_BUDGET = "NOT_RECOVERED_WITHIN_BUDGET"
RECOVERY_UNKNOWN = "UNKNOWN"
RECOVERY_STATUSES = (
    RECOVERY_RECOVERED,
    RECOVERY_PARTIAL,
    RECOVERY_NOT_WITHIN_BUDGET,
    RECOVERY_UNKNOWN,
)

# Compatibility values accepted only while reading old artifacts.  New formal
# records never store IRREVERSIBLE as a canonical recovery status.
LEGACY_LABELS = (
    "REVERSIBLE",
    "REVERSIBLE_WITH_COST",
    "PARTIALLY_RECOVERABLE",
    "IRREVERSIBLE",
    "NO_EFFECT",
    "UNKNOWN",
)


class GroundingValidationError(ValueError):
    """Raised when a point cannot be admitted to a formal grounding artifact."""


def _signal_present(value: Any) -> bool:
    """A measured zero is valid; an empty placeholder is not evidence."""
    return value is not None and value != "" and value != {} and value != []


def legacy_label_to_statuses(label: str) -> tuple[str, str]:
    """Map a legacy display label to the two canonical axes.

    The mapping does not make a legacy row formal: provenance and point-level
    evidence must still pass :meth:`GroundingPoint.validate`.  In particular,
    legacy ``IRREVERSIBLE`` becomes budget-relative failed recovery.
    """
    normalized = str(label or "UNKNOWN").upper()
    if normalized.startswith("REVERSIBLE_WITH_COST"):
        return EFFECT_CHANGED, RECOVERY_RECOVERED
    mapping = {
        "REVERSIBLE": (EFFECT_CHANGED, RECOVERY_RECOVERED),
        "PARTIALLY_RECOVERABLE": (EFFECT_CHANGED, RECOVERY_PARTIAL),
        "PARTIALLY_RECOVERED": (EFFECT_CHANGED, RECOVERY_PARTIAL),
        "IRREVERSIBLE": (EFFECT_CHANGED, RECOVERY_NOT_WITHIN_BUDGET),
        "NOT_RECOVERED_WITHIN_BUDGET": (
            EFFECT_CHANGED, RECOVERY_NOT_WITHIN_BUDGET),
        "NO_EFFECT": (EFFECT_NO_EFFECT, RECOVERY_UNKNOWN),
        "UNKNOWN": (EFFECT_UNKNOWN, RECOVERY_UNKNOWN),
    }
    if normalized not in mapping:
        raise GroundingValidationError(f"unknown legacy label: {label!r}")
    return mapping[normalized]


def statuses_to_display_label(effect_status: str, recovery_status: str) -> str:
    """Return a non-overclaiming UI label for canonical statuses."""
    if effect_status == EFFECT_NO_EFFECT:
        return EFFECT_NO_EFFECT
    if effect_status == EFFECT_UNKNOWN:
        return EFFECT_UNKNOWN
    mapping = {
        RECOVERY_RECOVERED: "REVERSIBLE",
        RECOVERY_PARTIAL: "PARTIALLY_RECOVERABLE",
        RECOVERY_NOT_WITHIN_BUDGET: RECOVERY_NOT_WITHIN_BUDGET,
        RECOVERY_UNKNOWN: "UNKNOWN",
    }
    return mapping.get(recovery_status, "UNKNOWN")


@dataclass
class GroundingPoint:
    """One executed action at one uniquely identified state.

    All fields are present in serialized output.  Empty defaults make legacy
    inspection possible; ``validate(formal=True)`` is the admission gate and
    rejects missing provenance rather than fabricating it.
    """

    schema_version: str = GROUNDING_SCHEMA_VERSION
    probe_point_id: str = ""
    probe_run_id: str = ""
    probe_name: str = ""
    state_id: str = ""
    candidate_id: str = ""
    action_instance_id: str = ""
    action_type: str = ""
    raw_action: str = ""
    canonical_action: str = ""
    site: str = ""
    environment_family: str = ""
    environment_instance: str = ""
    environment_origin: str = ""
    is_mock: bool = False
    task_id: str = ""
    trajectory_id: str = ""
    run_id: str = ""
    seed: Optional[int] = None
    url: str = ""
    account: str = ""
    privilege: str = ""
    budget_k: Optional[int] = None
    solver_set: list[str] = field(default_factory=list)
    controller_version: str = ""
    pre_observation_hash: str = ""
    pre_signal: Any = field(default_factory=dict)
    post_observation_hash: str = ""
    post_signal: Any = field(default_factory=dict)
    undo_actions: list[str] = field(default_factory=list)
    # Semantic IR is kept separately from the raw BrowserGym actions above.
    undo_semantic_actions: list[str] = field(default_factory=list)
    undo_observation_hashes: list[str] = field(default_factory=list)
    final_signal: Any = field(default_factory=dict)
    effect_status: str = EFFECT_UNKNOWN
    recovery_status: str = RECOVERY_UNKNOWN
    undo_cost_steps: Optional[int] = None
    residual_diff: Any = field(default_factory=dict)
    budget_exhausted: bool = False
    timestamp: str = ""
    code_version: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def display_label(self) -> str:
        return statuses_to_display_label(self.effect_status, self.recovery_status)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any], *, validate: bool = False) -> "GroundingPoint":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(row) - known)
        if unknown:
            raise GroundingValidationError(f"unknown grounding fields: {unknown}")
        # Formal imports are wire-format validation, not dataclass convenience
        # construction.  Letting defaults fill omitted keys can silently turn a
        # missing ``is_mock`` into ``False`` or an omitted status into UNKNOWN.
        # Legacy/UI inspection may still use ``validate=False``.
        if validate:
            missing = sorted(known - set(row))
            if missing:
                raise GroundingValidationError(
                    "formal grounding row is missing serialized fields: " +
                    ",".join(missing))
        point = cls(**row)
        if validate:
            point.validate(formal=True)
        return point

    def validation_errors(self, *, formal: bool = True) -> list[str]:
        errors: list[str] = []
        if self.schema_version != GROUNDING_SCHEMA_VERSION:
            errors.append(
                f"schema_version must be {GROUNDING_SCHEMA_VERSION!r}, got "
                f"{self.schema_version!r}")
        if self.effect_status not in EFFECT_STATUSES:
            errors.append(f"invalid effect_status {self.effect_status!r}")
        if self.recovery_status not in RECOVERY_STATUSES:
            errors.append(f"invalid recovery_status {self.recovery_status!r}")
        if self.effect_status == EFFECT_NO_EFFECT and self.recovery_status != RECOVERY_UNKNOWN:
            errors.append("NO_EFFECT must use recovery_status=UNKNOWN (recovery is not applicable)")
        if self.effect_status == EFFECT_UNKNOWN and self.recovery_status != RECOVERY_UNKNOWN:
            errors.append("unknown effect cannot carry a positive/negative recovery claim")
        if self.undo_cost_steps is not None and (
                not isinstance(self.undo_cost_steps, int) or self.undo_cost_steps < 0):
            errors.append("undo_cost_steps must be a nullable non-negative integer")
        if self.recovery_status == RECOVERY_RECOVERED:
            if self.undo_cost_steps is None:
                errors.append("RECOVERED requires undo_cost_steps")
            if not self.undo_actions and self.undo_cost_steps != 0:
                errors.append("RECOVERED with non-zero cost requires recorded undo_actions")
            if (self.undo_cost_steps is not None and self.undo_actions and
                    self.undo_cost_steps != len(self.undo_actions)):
                errors.append("RECOVERED undo_cost_steps must equal recorded undo_actions")
        if self.recovery_status == RECOVERY_PARTIAL:
            if self.undo_cost_steps is None:
                errors.append("PARTIALLY_RECOVERED requires undo_cost_steps")
            if not self.undo_actions and self.undo_cost_steps != 0:
                errors.append(
                    "PARTIALLY_RECOVERED with non-zero cost requires undo_actions")
            if (self.undo_cost_steps is not None and self.undo_actions and
                    self.undo_cost_steps != len(self.undo_actions)):
                errors.append(
                    "PARTIALLY_RECOVERED undo_cost_steps must equal undo_actions")
        if self.undo_actions:
            if len(self.undo_semantic_actions) != len(self.undo_actions):
                errors.append(
                    "undo_semantic_actions must align 1:1 with undo_actions")
            if len(self.undo_observation_hashes) != len(self.undo_actions):
                errors.append(
                    "undo_observation_hashes must align 1:1 with undo_actions")
        elif self.undo_semantic_actions or self.undo_observation_hashes:
            errors.append(
                "semantic undo/actions hashes cannot exist without undo_actions")
        if self.recovery_status == RECOVERY_NOT_WITHIN_BUDGET:
            if not isinstance(self.budget_k, int) or self.budget_k <= 0:
                errors.append("NOT_RECOVERED_WITHIN_BUDGET requires budget_k>0")
            if not self.solver_set:
                errors.append("NOT_RECOVERED_WITHIN_BUDGET requires a non-empty solver_set")
            if formal:
                union = self.evidence.get("solver_union") \
                    if isinstance(self.evidence, dict) else None
                if not isinstance(union, dict) or union.get("protocol") != "solver_union.v1":
                    errors.append(
                        "formal negative requires evidence.solver_union protocol=solver_union.v1")
                else:
                    traces = union.get("traces") or []
                    kinds = {str(trace.get("solver_kind")) for trace in traces
                             if isinstance(trace, dict)}
                    required_kinds = {
                        "site_specific_deterministic", "affordance_bfs",
                        "llm_undo_attacker",
                    }
                    missing = sorted(required_kinds - kinds)
                    if missing:
                        errors.append(
                            f"formal negative missing solver trace kinds {missing}")
                    attacker_seeds = {
                        seed for trace in traces if isinstance(trace, dict)
                        and trace.get("solver_kind") == "llm_undo_attacker"
                        for seed in (trace.get("seeds") or [])
                    }
                    if len(attacker_seeds) < 2:
                        errors.append(
                            "formal negative requires LLM attacker traces from >=2 seeds")
                    if union.get("recovery_status") != self.recovery_status:
                        errors.append("solver union recovery_status contradicts point")
                    if union.get("budget_k") != self.budget_k:
                        errors.append("solver union budget_k contradicts point")
                    if list(union.get("solver_set") or []) != list(self.solver_set):
                        errors.append("solver union solver_set contradicts point")
                    if union.get("successful_solver") not in (None, ""):
                        errors.append("formal negative cannot name a successful_solver")
                    trace_names = [str(trace.get("solver_name") or "")
                                   for trace in traces if isinstance(trace, dict)]
                    if trace_names != list(self.solver_set):
                        errors.append(
                            "formal negative solver_set must exactly match trace names")
                    for trace in traces:
                        if not isinstance(trace, dict):
                            errors.append("formal negative contains a non-object solver trace")
                            continue
                        name = str(trace.get("solver_name") or "<unnamed>")
                        if trace.get("success") is not False:
                            errors.append(
                                f"formal negative trace {name} is not an explicit failure")
                        if trace.get("budget_k") != self.budget_k:
                            errors.append(
                                f"formal negative trace {name} budget contradicts point")
                        if trace.get("undo_actions"):
                            errors.append(
                                f"formal negative trace {name} carries a recovery path")
                        if str(trace.get("termination_reason") or "") == "recovered":
                            errors.append(
                                f"formal negative trace {name} terminated as recovered")
        if not isinstance(self.is_mock, bool):
            errors.append("is_mock must be boolean")

        if formal:
            required = (
                "probe_point_id", "probe_run_id", "probe_name", "state_id",
                "candidate_id", "action_instance_id", "action_type", "raw_action",
                "canonical_action", "site", "environment_family",
                "environment_instance", "environment_origin", "task_id",
                "trajectory_id", "run_id",
                "url", "account", "privilege", "controller_version",
                "pre_observation_hash", "post_observation_hash", "timestamp",
                "code_version",
            )
            for name in required:
                if not str(getattr(self, name) or "").strip():
                    errors.append(f"missing formal provenance: {name}")
            if not self.solver_set:
                errors.append("formal point requires solver_set")
            if not isinstance(self.budget_k, int) or self.budget_k <= 0:
                errors.append("formal point requires budget_k>0")
            if not isinstance(self.evidence, dict) or not self.evidence:
                errors.append("formal point requires non-empty evidence")
            elif not str(self.evidence.get("candidate_snapshot_hash") or "").strip():
                errors.append(
                    "formal point evidence requires candidate_snapshot_hash")
            if self.effect_status in {EFFECT_CHANGED, EFFECT_NO_EFFECT}:
                if not _signal_present(self.pre_signal):
                    errors.append("measured effect requires non-empty pre_signal")
                if not _signal_present(self.post_signal):
                    errors.append("measured effect requires non-empty post_signal")
            if (self.effect_status == EFFECT_CHANGED and
                    _signal_present(self.pre_signal) and
                    _signal_present(self.post_signal) and
                    self.pre_signal == self.post_signal):
                errors.append("CHANGED requires pre_signal != post_signal")
            if (self.effect_status == EFFECT_NO_EFFECT and
                    _signal_present(self.pre_signal) and
                    _signal_present(self.post_signal) and
                    self.pre_signal != self.post_signal):
                errors.append("NO_EFFECT requires pre_signal == post_signal")
            if self.recovery_status in {
                    RECOVERY_RECOVERED, RECOVERY_PARTIAL,
                    RECOVERY_NOT_WITHIN_BUDGET}:
                if not _signal_present(self.final_signal):
                    errors.append("measured recovery requires non-empty final_signal")
            if (self.recovery_status == RECOVERY_RECOVERED and
                    _signal_present(self.pre_signal) and
                    _signal_present(self.final_signal) and
                    self.final_signal != self.pre_signal):
                errors.append("RECOVERED requires final_signal == pre_signal")
            if (self.recovery_status == RECOVERY_PARTIAL and
                    _signal_present(self.pre_signal) and
                    _signal_present(self.final_signal) and
                    self.final_signal == self.pre_signal):
                errors.append(
                    "PARTIALLY_RECOVERED contradicts a fully restored final_signal")
            if (self.recovery_status == RECOVERY_PARTIAL and
                    not _signal_present(self.residual_diff)):
                errors.append("PARTIALLY_RECOVERED requires non-empty residual_diff")
            if (self.recovery_status == RECOVERY_NOT_WITHIN_BUDGET and
                    _signal_present(self.pre_signal) and
                    _signal_present(self.final_signal) and
                    self.final_signal == self.pre_signal):
                errors.append(
                    "NOT_RECOVERED_WITHIN_BUDGET contradicts restored final_signal")
        return errors

    def validate(self, *, formal: bool = True) -> None:
        errors = self.validation_errors(formal=formal)
        if errors:
            ident = self.probe_point_id or "<missing probe_point_id>"
            raise GroundingValidationError(f"{ident}: " + "; ".join(errors))


def point_manifest_row(point: GroundingPoint) -> dict[str, Any]:
    """Small 1:1 manifest row for integrity checks and dataset browsing."""
    payload = point.to_dict()
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": point.schema_version,
        "probe_point_id": point.probe_point_id,
        "probe_run_id": point.probe_run_id,
        "state_id": point.state_id,
        "candidate_id": point.candidate_id,
        "action_instance_id": point.action_instance_id,
        "site": point.site,
        "environment_origin": point.environment_origin,
        "action_type": point.action_type,
        "effect_status": point.effect_status,
        "recovery_status": point.recovery_status,
        "undo_cost_steps": point.undo_cost_steps,
        "is_mock": point.is_mock,
        "timestamp": point.timestamp,
        "controller_version": point.controller_version,
        "record_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)


def _atomic_write(path: Path, text: str) -> None:
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


def load_probe_points(path: Path, *, validate: bool = True) -> dict[str, GroundingPoint]:
    """Load formal points keyed only by unique ``probe_point_id``."""
    out: dict[str, GroundingPoint] = {}
    if not path.exists():
        return out
    for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            point = GroundingPoint.from_dict(json.loads(line), validate=validate)
        except (json.JSONDecodeError, TypeError, GroundingValidationError) as exc:
            raise GroundingValidationError(f"{path}:{line_no}: {exc}") from exc
        if point.probe_point_id in out:
            raise GroundingValidationError(
                f"{path}:{line_no}: duplicate probe_point_id {point.probe_point_id!r}")
        out[point.probe_point_id] = point
    return out


def assert_manifest_integrity(points_path: Path, manifest_path: Path) -> None:
    """Require exact 1:1 IDs and hashes between formal body and manifest."""
    points = load_probe_points(points_path, validate=True)
    manifest: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        for line_no, line in enumerate(manifest_path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            point_id = row.get("probe_point_id", "")
            if not point_id or point_id in manifest:
                raise GroundingValidationError(
                    f"{manifest_path}:{line_no}: missing/duplicate probe_point_id")
            manifest[point_id] = row
    if set(points) != set(manifest):
        raise GroundingValidationError(
            "formal grounding/manifest ID mismatch: "
            f"body_only={sorted(set(points) - set(manifest))}, "
            f"manifest_only={sorted(set(manifest) - set(points))}")
    for point_id, point in points.items():
        expected = point_manifest_row(point)
        if manifest[point_id] != expected:
            raise GroundingValidationError(f"manifest hash/fields differ for {point_id}")


def save_probe_points(points: Iterable[GroundingPoint], points_path: Path,
                      manifest_path: Path, *, append: bool = True) -> tuple[Path, Path]:
    """Validate and atomically persist point body + 1:1 manifest.

    Existing point IDs are immutable: append mode merges only new IDs and
    rejects collisions.  Both files are materialized from the same validated
    in-memory set, so a normal successful call cannot create a row-count skew.
    """
    existing = load_probe_points(points_path, validate=True) if append else {}
    merged = dict(existing)
    for point in points:
        point.validate(formal=True)
        if point.probe_point_id in merged:
            raise GroundingValidationError(
                f"refusing to overwrite immutable probe_point_id {point.probe_point_id!r}")
        merged[point.probe_point_id] = point
    ordered = list(merged.values())
    _atomic_write(points_path, _jsonl_text(p.to_dict() for p in ordered))
    _atomic_write(manifest_path, _jsonl_text(point_manifest_row(p) for p in ordered))
    assert_manifest_integrity(points_path, manifest_path)
    return points_path, manifest_path


def apply_solver_union(point: GroundingPoint, union_result: Any) -> GroundingPoint:
    """Attach a validated solver-union result to a point before persistence.

    Kept duck-typed to avoid a schema→solver import cycle.  The solver module
    owns trace validation; this gate owns point ontology and round-trip
    validation.
    """
    required = (
        "recovery_status", "budget_k", "solver_set", "undo_actions",
        "undo_semantic_actions", "undo_observation_hashes",
        "undo_cost_steps", "final_signal", "residual_diff",
        "budget_exhausted", "to_evidence",
    )
    missing = [name for name in required if not hasattr(union_result, name)]
    if missing:
        raise GroundingValidationError(
            f"solver union result lacks required fields: {missing}")
    evidence = dict(point.evidence)
    evidence["solver_union"] = union_result.to_evidence()
    updated = replace(
        point,
        budget_k=int(union_result.budget_k),
        solver_set=list(union_result.solver_set),
        recovery_status=str(union_result.recovery_status),
        undo_actions=list(union_result.undo_actions),
        undo_semantic_actions=list(union_result.undo_semantic_actions),
        undo_observation_hashes=list(union_result.undo_observation_hashes),
        undo_cost_steps=union_result.undo_cost_steps,
        final_signal=union_result.final_signal,
        residual_diff=union_result.residual_diff,
        budget_exhausted=bool(union_result.budget_exhausted),
        evidence=evidence,
    )
    updated.validate(formal=True)
    return updated
