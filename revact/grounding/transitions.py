"""Immutable observation-body evidence for point-level probes.

``GroundingPoint`` intentionally keeps compact hashes so labels and manifests
remain easy to audit.  A hash alone, however, cannot supervise a forward
transition or prove that an undo control was visible.  This module owns the
separate append-only artifact containing the exact BrowserGym AXTree body and
the target-aware policy view captured at every executed transition step.

Old hash-only point rows are not backfilled: a body cannot be reconstructed
from a digest.  They remain valid grounding measurements but are ineligible for
``prediction_source=probe_transition`` until a new, versioned probe run creates
one of these records.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable

from ..envs.obs_utils import bid_is_visible, prune_axtree_txt
from .schema import (EFFECT_CHANGED, EFFECT_NO_EFFECT, EFFECT_STATUSES,
                     RECOVERY_PARTIAL, RECOVERY_RECOVERED, RECOVERY_STATUSES,
                     RECOVERY_UNKNOWN)

TRANSITION_SCHEMA_VERSION = "iris.grounding.transition.v1"
TRANSITION_BODY_RELATIVE = Path("grounded/transitions/probe_transitions.v1.jsonl")
TRANSITION_MANIFEST_RELATIVE = Path(
    "grounded/transitions/TRANSITION_MANIFEST.v1.jsonl")


class TransitionValidationError(ValueError):
    """A transition body or its 1:1 manifest is not formal-grade."""


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _canonical_sha(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ObservationBody:
    """One exact environment observation plus its policy-visible projection."""

    capture_stage: str
    url: str
    raw_axtree: str
    raw_axtree_sha256: str
    policy_axtree: str
    policy_axtree_sha256: str
    anchor_bids: list[str] = field(default_factory=list)
    compaction_strategy: str = "target_aware_prune.v1"

    @classmethod
    def capture(cls, observation: dict[str, Any], *, stage: str,
                anchor_bids: Iterable[str] = ()) -> "ObservationBody":
        raw = str(observation.get("axtree_txt") or "")
        anchors = [str(bid) for bid in anchor_bids if str(bid)]
        policy = prune_axtree_txt(raw, anchor_bids=anchors)
        body = cls(
            capture_stage=str(stage),
            url=str(observation.get("url") or ""),
            raw_axtree=raw,
            raw_axtree_sha256=sha256_text(raw),
            policy_axtree=policy,
            policy_axtree_sha256=sha256_text(policy),
            anchor_bids=anchors,
        )
        body.validate()
        return body

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ObservationBody":
        known = {item.name for item in fields(cls)}
        if set(row) != known:
            raise TransitionValidationError(
                "observation body fields differ: "
                f"missing={sorted(known - set(row))}, "
                f"unknown={sorted(set(row) - known)}")
        body = cls(**row)
        body.validate()
        return body

    def validate(self) -> None:
        errors: list[str] = []
        if not self.capture_stage.strip():
            errors.append("capture_stage is required")
        if not self.url.strip():
            errors.append("url is required")
        if not self.raw_axtree:
            errors.append("raw_axtree is required")
        if not self.policy_axtree:
            errors.append("policy_axtree is required")
        if self.raw_axtree_sha256 != sha256_text(self.raw_axtree):
            errors.append("raw_axtree_sha256 does not match raw_axtree")
        if self.policy_axtree_sha256 != sha256_text(self.policy_axtree):
            errors.append("policy_axtree_sha256 does not match policy_axtree")
        expected_policy = prune_axtree_txt(
            self.raw_axtree, anchor_bids=list(self.anchor_bids))
        if self.policy_axtree != expected_policy:
            errors.append("policy_axtree is not the declared target-aware projection")
        if len(set(self.anchor_bids)) != len(self.anchor_bids):
            errors.append("anchor_bids contains duplicates")
        if errors:
            raise TransitionValidationError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutedTransitionStep:
    step_index: int
    raw_action: str
    semantic_action: str
    observation_after: ObservationBody
    step_kind: str = "recovery"

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ExecutedTransitionStep":
        data = dict(row)
        data["observation_after"] = ObservationBody.from_dict(
            dict(data.get("observation_after") or {}))
        step = cls(**data)
        step.validate()
        return step

    def validate(self) -> None:
        if not isinstance(self.step_index, int) or self.step_index < 0:
            raise TransitionValidationError("step_index must be non-negative")
        if not self.raw_action.strip() or not self.semantic_action.strip():
            raise TransitionValidationError(
                "executed step requires raw_action and semantic_action")
        if self.step_kind not in {"measurement_navigation", "recovery",
                                  "solver_search"}:
            raise TransitionValidationError(f"invalid step_kind {self.step_kind!r}")
        self.observation_after.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProbeTransition:
    """One observed ``pre --action--> post`` and every recorded recovery step."""

    schema_version: str
    transition_id: str
    probe_point_id: str
    probe_run_id: str
    state_id: str
    candidate_id: str
    action_instance_id: str
    action_type: str
    raw_action: str
    canonical_action: str
    candidate_snapshot_hash: str
    pre_observation: ObservationBody
    post_observation: ObservationBody
    post_signal_observation: ObservationBody
    recovery_steps: list[ExecutedTransitionStep]
    pre_signal: Any
    post_signal: Any
    final_signal: Any
    effect_status: str
    recovery_status: str
    undo_cost_steps: int | None
    budget_k: int
    replay_verification: str
    replay_target_contract: dict[str, Any]
    timestamp: str
    code_version: str
    capture_protocol: str = "point_runner.observation_body.v1"

    @classmethod
    def from_dict(cls, row: dict[str, Any], *, validate: bool = True
                  ) -> "ProbeTransition":
        known = {item.name for item in fields(cls)}
        if set(row) != known:
            raise TransitionValidationError(
                "transition fields differ: "
                f"missing={sorted(known - set(row))}, "
                f"unknown={sorted(set(row) - known)}")
        data = dict(row)
        data["pre_observation"] = ObservationBody.from_dict(
            dict(data["pre_observation"]))
        data["post_observation"] = ObservationBody.from_dict(
            dict(data["post_observation"]))
        data["post_signal_observation"] = ObservationBody.from_dict(
            dict(data["post_signal_observation"]))
        data["recovery_steps"] = [
            ExecutedTransitionStep.from_dict(dict(step))
            for step in data["recovery_steps"]
        ]
        transition = cls(**data)
        if validate:
            transition.validate()
        return transition

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        if self.schema_version != TRANSITION_SCHEMA_VERSION:
            errors.append(f"schema_version must be {TRANSITION_SCHEMA_VERSION}")
        required = (
            "transition_id", "probe_point_id", "probe_run_id", "state_id",
            "candidate_id", "action_instance_id", "action_type", "raw_action",
            "canonical_action", "candidate_snapshot_hash", "replay_verification",
            "timestamp", "code_version",
        )
        for name in required:
            if not str(getattr(self, name) or "").strip():
                errors.append(f"missing transition provenance: {name}")
        if self.transition_id != f"transition::{self.probe_point_id}":
            errors.append("transition_id must be derived from probe_point_id")
        if self.effect_status not in EFFECT_STATUSES:
            errors.append(f"invalid effect_status {self.effect_status!r}")
        if self.recovery_status not in RECOVERY_STATUSES:
            errors.append(f"invalid recovery_status {self.recovery_status!r}")
        if not isinstance(self.budget_k, int) or self.budget_k <= 0:
            errors.append("budget_k must be positive")
        if self.undo_cost_steps is not None and self.undo_cost_steps != len(
                self.recovery_steps):
            errors.append("undo_cost_steps must equal captured recovery_steps")
        if len({step.step_index for step in self.recovery_steps}) != len(
                self.recovery_steps):
            errors.append("recovery step indices must be unique")
        if [step.step_index for step in self.recovery_steps] != list(
                range(len(self.recovery_steps))):
            errors.append("recovery step indices must be contiguous from zero")
        if len(self.recovery_steps) > self.budget_k:
            errors.append("captured recovery steps exceed budget_k")
        if self.effect_status == EFFECT_CHANGED and self.pre_signal == self.post_signal:
            errors.append("CHANGED requires pre_signal != post_signal")
        if self.effect_status == EFFECT_NO_EFFECT and self.pre_signal != self.post_signal:
            errors.append("NO_EFFECT requires pre_signal == post_signal")
        if self.effect_status == EFFECT_NO_EFFECT:
            if self.recovery_status != RECOVERY_UNKNOWN:
                errors.append("NO_EFFECT recovery_status must be UNKNOWN")
            if self.recovery_steps or self.undo_cost_steps is not None:
                errors.append("NO_EFFECT cannot carry recovery steps/cost")
        if self.recovery_status == RECOVERY_RECOVERED:
            if self.final_signal != self.pre_signal:
                errors.append("RECOVERED requires final_signal == pre_signal")
            if self.undo_cost_steps is None:
                errors.append("RECOVERED requires undo_cost_steps")
        if self.recovery_status == RECOVERY_PARTIAL:
            if self.final_signal == self.pre_signal:
                errors.append("PARTIALLY_RECOVERED contradicts restored signal")
            if self.undo_cost_steps is None:
                errors.append("PARTIALLY_RECOVERED requires undo_cost_steps")
        try:
            self.pre_observation.validate()
            self.post_observation.validate()
            self.post_signal_observation.validate()
        except TransitionValidationError as exc:
            errors.append(str(exc))
        # The supervised click must be visible in the exact pre-action policy
        # projection.  This is a construction-time, not training-time, gate.
        from ..train.validators import parse_action
        parsed = parse_action(self.raw_action)
        if parsed is None or not parsed.bid:
            errors.append("formal transition action must be a parsed bid action")
        elif not bid_is_visible(self.pre_observation.policy_axtree, parsed.bid):
            errors.append(f"supervised bid [{parsed.bid}] absent from pre observation")
        for step in self.recovery_steps:
            try:
                step.validate()
            except TransitionValidationError as exc:
                errors.append(f"recovery step {step.step_index}: {exc}")
        return errors

    def validate(self) -> None:
        errors = self.validation_errors()
        if errors:
            raise TransitionValidationError(
                f"{self.transition_id or '<missing transition_id>'}: " +
                "; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def transition_manifest_row(transition: ProbeTransition) -> dict[str, Any]:
    transition.validate()
    return {
        "schema_version": transition.schema_version,
        "transition_id": transition.transition_id,
        "probe_point_id": transition.probe_point_id,
        "probe_run_id": transition.probe_run_id,
        "state_id": transition.state_id,
        "candidate_id": transition.candidate_id,
        "pre_observation_hash": transition.pre_observation.policy_axtree_sha256,
        "post_observation_hash": transition.post_observation.policy_axtree_sha256,
        "recovery_step_count": len(transition.recovery_steps),
        "timestamp": transition.timestamp,
        "code_version": transition.code_version,
        "record_sha256": _canonical_sha(transition.to_dict()),
    }


def _jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows)


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


def load_probe_transitions(path: Path, *, validate: bool = True
                           ) -> dict[str, ProbeTransition]:
    rows: dict[str, ProbeTransition] = {}
    if not path.exists():
        return rows
    for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            transition = ProbeTransition.from_dict(json.loads(line),
                                                   validate=validate)
        except (json.JSONDecodeError, KeyError, TypeError,
                TransitionValidationError) as exc:
            raise TransitionValidationError(f"{path}:{line_no}: {exc}") from exc
        if transition.transition_id in rows:
            raise TransitionValidationError(
                f"{path}:{line_no}: duplicate transition_id "
                f"{transition.transition_id!r}")
        rows[transition.transition_id] = transition
    return rows


def assert_transition_manifest_integrity(body_path: Path,
                                         manifest_path: Path) -> None:
    transitions = load_probe_transitions(body_path, validate=True)
    manifest: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        for line_no, line in enumerate(manifest_path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            ident = str(row.get("transition_id") or "")
            if not ident or ident in manifest:
                raise TransitionValidationError(
                    f"{manifest_path}:{line_no}: missing/duplicate transition_id")
            manifest[ident] = row
    if set(transitions) != set(manifest):
        raise TransitionValidationError(
            "transition body/manifest ID mismatch: "
            f"body_only={sorted(set(transitions) - set(manifest))}, "
            f"manifest_only={sorted(set(manifest) - set(transitions))}")
    for ident, transition in transitions.items():
        if manifest[ident] != transition_manifest_row(transition):
            raise TransitionValidationError(
                f"transition manifest hash/fields differ for {ident}")


def save_probe_transitions(transitions: Iterable[ProbeTransition],
                           body_path: Path, manifest_path: Path, *,
                           append: bool = True) -> tuple[Path, Path]:
    existing = load_probe_transitions(body_path, validate=True) if append else {}
    merged = dict(existing)
    for transition in transitions:
        transition.validate()
        if transition.transition_id in merged:
            raise TransitionValidationError(
                "refusing to overwrite immutable transition_id "
                f"{transition.transition_id!r}")
        if any(row.probe_point_id == transition.probe_point_id
               for row in merged.values()):
            raise TransitionValidationError(
                "refusing second transition for immutable probe_point_id "
                f"{transition.probe_point_id!r}")
        merged[transition.transition_id] = transition
    ordered = list(merged.values())
    _atomic_write(body_path, _jsonl_text(row.to_dict() for row in ordered))
    _atomic_write(manifest_path,
                  _jsonl_text(transition_manifest_row(row) for row in ordered))
    assert_transition_manifest_integrity(body_path, manifest_path)
    return body_path, manifest_path


def assert_point_transition_integrity(points: dict[str, Any],
                                      transitions: dict[str, ProbeTransition],
                                      *, require_all: bool = False) -> dict[str, Any]:
    """Cross-check compact grounding hashes against available body records.

    ``require_all=False`` is the honest migration mode: existing hash-only
    points are reported as uncovered, never reconstructed.  Formal transition-
    supervised exports call this with ``require_all=True``.
    """
    by_point = {row.probe_point_id: row for row in transitions.values()}
    missing = sorted(set(points) - set(by_point))
    orphan = sorted(set(by_point) - set(points))
    mismatches: list[str] = []
    for point_id in sorted(set(points) & set(by_point)):
        point = points[point_id]
        transition = by_point[point_id]
        checks = {
            "state_id": (point.state_id, transition.state_id),
            "candidate_id": (point.candidate_id, transition.candidate_id),
            "action_instance_id": (
                point.action_instance_id, transition.action_instance_id),
            "pre_observation_hash": (
                point.pre_observation_hash,
                transition.pre_observation.policy_axtree_sha256),
            "post_observation_hash": (
                point.post_observation_hash,
                transition.post_observation.policy_axtree_sha256),
            "effect_status": (point.effect_status, transition.effect_status),
            "recovery_status": (
                point.recovery_status, transition.recovery_status),
            "undo_cost_steps": (
                point.undo_cost_steps, transition.undo_cost_steps),
        }
        mismatches.extend(
            f"{point_id}:{name}" for name, values in checks.items()
            if values[0] != values[1])
    if orphan or mismatches or (require_all and missing):
        raise TransitionValidationError(
            "point/transition integrity failed: "
            f"missing={missing}, orphan={orphan}, mismatches={mismatches}")
    return {
        "point_count": len(points),
        "transition_count": len(transitions),
        "covered_point_count": len(set(points) & set(by_point)),
        "missing_transition_point_ids": missing,
        "coverage": (len(set(points) & set(by_point)) / len(points)
                     if points else 0.0),
    }
