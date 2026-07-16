"""Declarative probe authoring without a label input channel.

Workbench users may specify what to execute, observe and try as undo.  Labels
remain outputs of validated execution.  Specs are immutable review artifacts;
this module does not run a browser or bypass destructive gates.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

PROBE_AUTHORING_SCHEMA_VERSION = "iris.probe_spec.v1"
PROBE_EXECUTION_SCHEMA_VERSION = "iris.probe_execution_spec.v1"
SAFETY_LEVELS = ("non_destructive", "self_recovering", "destructive")
SIGNAL_CHANNELS = ("ui_structural", "content_hash", "api", "db", "external")
_FORBIDDEN = {
    "label", "reversibility", "effect_status", "recovery_status",
    "normative_risk", "expected_decision", "safe", "irreversible",
}


class ProbeAuthoringError(ValueError):
    pass


@dataclass(frozen=True)
class AuthoredProbeSpec:
    schema_version: str
    spec_id: str
    name: str
    site: str
    action_type: str
    candidate_id: str
    state_id: str
    action_instance_id: str
    raw_action: str
    canonical_action: str
    signal_channels: tuple[str, ...]
    undo_sequences: tuple[tuple[str, ...], ...]
    solver_set: tuple[str, ...]
    budget_k: int
    safety_level: str
    controller_version: str
    author: str
    timestamp: str
    fixture_status: str = "PENDING"
    code_review_status: str = "PENDING"
    requires_commit_flag: bool = False
    requires_destructive_env: bool = False

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["signal_channels"] = list(self.signal_channels)
        row["undo_sequences"] = [list(seq) for seq in self.undo_sequences]
        row["solver_set"] = list(self.solver_set)
        return row

    def validate(self) -> None:
        errors = []
        if self.schema_version != PROBE_AUTHORING_SCHEMA_VERSION:
            errors.append("bad schema_version")
        for name in (
                "spec_id", "name", "site", "action_type", "candidate_id", "state_id",
                "action_instance_id", "raw_action", "canonical_action",
                "controller_version", "author", "timestamp"):
            if not str(getattr(self, name) or "").strip():
                errors.append(f"missing {name}")
        if self.safety_level not in SAFETY_LEVELS:
            errors.append(f"invalid safety_level {self.safety_level!r}")
        if not isinstance(self.budget_k, int) or self.budget_k <= 0:
            errors.append("budget_k must be a positive integer")
        bad_channels = sorted(set(self.signal_channels) - set(SIGNAL_CHANNELS))
        if not self.signal_channels or bad_channels:
            errors.append(f"invalid signal_channels {bad_channels}")
        if not self.undo_sequences:
            errors.append("at least one undo solver sequence is required")
        if any(not seq or not all(str(action).strip() for action in seq)
               for seq in self.undo_sequences):
            errors.append("undo sequences must contain non-empty actions")
        if not self.solver_set or not all(str(solver).strip()
                                          for solver in self.solver_set):
            errors.append("solver_set must name at least one solver")
        if self.fixture_status not in {"PENDING", "PASSED", "FAILED"}:
            errors.append("invalid fixture_status")
        if self.code_review_status not in {"PENDING", "APPROVED", "REJECTED"}:
            errors.append("invalid code_review_status")
        if self.safety_level == "destructive" and not (
                self.requires_commit_flag and self.requires_destructive_env):
            errors.append("destructive specs must retain both execution gates")
        if errors:
            raise ProbeAuthoringError(
                f"{self.spec_id or '<missing spec_id>'}: " + "; ".join(errors))

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "AuthoredProbeSpec":
        leaked = sorted(set(row) & _FORBIDDEN)
        if leaked:
            raise ProbeAuthoringError(
                f"probe authoring cannot accept final labels: {leaked}")
        unknown = sorted(set(row) - {field.name for field in fields(cls)})
        if unknown:
            raise ProbeAuthoringError(f"unknown probe spec fields: {unknown}")
        materialized = dict(row)
        materialized["signal_channels"] = tuple(materialized.get("signal_channels") or ())
        materialized["undo_sequences"] = tuple(
            tuple(seq) for seq in materialized.get("undo_sequences") or ())
        materialized["solver_set"] = tuple(materialized.get("solver_set") or ())
        spec = cls(**materialized)
        spec.validate()
        return spec


def spec_from_workbench(payload: Mapping[str, Any], *, timestamp: str,
                        controller_version: str) -> AuthoredProbeSpec:
    """Build a content-addressed draft from the label-free UI payload."""
    leaked = sorted(set(payload) & _FORBIDDEN)
    if leaked:
        raise ProbeAuthoringError(
            f"probe authoring cannot accept final labels: {leaked}")
    allowed = {
        "name", "site", "action_type", "candidate_id", "state_id", "action_instance_id",
        "raw_action", "canonical_action", "signal_channels", "undo_sequences",
        "solver_set", "budget_k", "safety_level", "author",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ProbeAuthoringError(f"unknown workbench fields: {unknown}")
    identity = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
    spec = AuthoredProbeSpec(
        schema_version=PROBE_AUTHORING_SCHEMA_VERSION,
        spec_id="probe-spec-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20],
        name=str(payload.get("name") or ""),
        site=str(payload.get("site") or ""),
        action_type=str(payload.get("action_type") or ""),
        candidate_id=str(payload.get("candidate_id") or ""),
        state_id=str(payload.get("state_id") or ""),
        action_instance_id=str(payload.get("action_instance_id") or ""),
        raw_action=str(payload.get("raw_action") or ""),
        canonical_action=str(payload.get("canonical_action") or ""),
        signal_channels=tuple(payload.get("signal_channels") or ()),
        undo_sequences=tuple(tuple(seq) for seq in
                             (payload.get("undo_sequences") or ())),
        solver_set=tuple(payload.get("solver_set") or ()),
        budget_k=int(payload.get("budget_k") or 0),
        safety_level=str(payload.get("safety_level") or ""),
        controller_version=controller_version,
        author=str(payload.get("author") or "workbench"),
        timestamp=timestamp,
        requires_commit_flag=str(payload.get("safety_level") or "") == "destructive",
        requires_destructive_env=str(payload.get("safety_level") or "") == "destructive",
    )
    spec.validate()
    return spec


def load_authored_specs(path: Path) -> list[AuthoredProbeSpec]:
    out = []
    if not path.exists():
        return out
    seen = set()
    for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            spec = AuthoredProbeSpec.from_dict(json.loads(line))
        except (json.JSONDecodeError, TypeError, ProbeAuthoringError) as exc:
            raise ProbeAuthoringError(f"{path}:{line_no}: {exc}") from exc
        if spec.spec_id in seen:
            raise ProbeAuthoringError(f"{path}:{line_no}: duplicate spec_id")
        seen.add(spec.spec_id)
        out.append(spec)
    return out


def save_authored_spec(spec: AuthoredProbeSpec, path: Path) -> Path:
    """Atomically append an immutable spec; identical replay is idempotent."""
    spec.validate()
    existing = {row.spec_id: row for row in load_authored_specs(path)}
    if spec.spec_id in existing:
        old = existing[spec.spec_id].to_dict()
        new = spec.to_dict()
        # ``timestamp`` is receipt metadata, not part of the content-addressed
        # proposal identity.  Re-posting the same proposal is idempotent.
        old.pop("timestamp", None)
        new.pop("timestamp", None)
        if old != new:
            raise ProbeAuthoringError(f"immutable spec collision {spec.spec_id}")
        return path
    rows = [*existing.values(), spec]
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row.to_dict(), ensure_ascii=False,
                                        sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return path


@dataclass(frozen=True)
class ProbeExecutionSpec:
    """Reviewed point execution contract consumed by ``revact probe``.

    The draft and execution schemas are intentionally different lifecycle
    states, but this dataclass is the single tested promotion bridge between
    them.  Neither state has a label field.
    """

    schema_version: str
    authored_spec_id: str
    probe_name: str
    site: str
    action_type: str
    probe_point_id: str
    probe_run_id: str
    state_id: str
    candidate_id: str
    candidate_snapshot_hash: str
    action_instance_id: str
    raw_action: str
    canonical_action: str
    environment_family: str
    environment_instance: str
    environment_origin: str
    is_mock: bool
    task_id: str
    trajectory_id: str
    run_id: str
    seed: int | None
    url: str
    account: str
    privilege: str
    signal_channels: tuple[str, ...]
    undo_sequences: tuple[tuple[str, ...], ...]
    solver_set: tuple[str, ...]
    budget_k: int
    safety_level: str
    controller_version: str
    code_version: str
    fixture_status: str
    code_review_status: str
    reviewer: str
    review_timestamp: str
    requires_commit_flag: bool
    requires_destructive_env: bool
    product_url: str = ""
    submission_url: str = ""
    forum_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["signal_channels"] = list(self.signal_channels)
        row["undo_sequences"] = [list(seq) for seq in self.undo_sequences]
        row["solver_set"] = list(self.solver_set)
        return row

    def validate(self) -> None:
        errors: list[str] = []
        if self.schema_version != PROBE_EXECUTION_SCHEMA_VERSION:
            errors.append("bad schema_version")
        for name in (
                "authored_spec_id", "probe_name", "site", "action_type",
                "probe_point_id", "probe_run_id", "state_id", "candidate_id",
                "candidate_snapshot_hash", "action_instance_id", "raw_action",
                "canonical_action", "environment_family", "environment_instance",
                "environment_origin", "task_id", "trajectory_id", "run_id", "url",
                "account", "privilege", "controller_version", "code_version",
                "reviewer", "review_timestamp"):
            if not str(getattr(self, name) or "").strip():
                errors.append(f"missing {name}")
        if self.is_mock is not False:
            errors.append("formal execution spec requires is_mock=false")
        if self.signal_channels and set(self.signal_channels) <= set(SIGNAL_CHANNELS):
            pass
        else:
            errors.append("invalid signal_channels")
        if not self.undo_sequences or any(not seq for seq in self.undo_sequences):
            errors.append("at least one non-empty undo sequence is required")
        if not self.solver_set or not all(str(item).strip() for item in self.solver_set):
            errors.append("solver_set is required")
        if not isinstance(self.budget_k, int) or self.budget_k <= 0:
            errors.append("budget_k must be a positive integer")
        if self.safety_level not in SAFETY_LEVELS:
            errors.append("invalid safety_level")
        if self.fixture_status != "PASSED":
            errors.append("fixture_status must be PASSED before execution")
        if self.code_review_status != "APPROVED":
            errors.append("code_review_status must be APPROVED before execution")
        if self.safety_level == "destructive" and not (
                self.requires_commit_flag and self.requires_destructive_env):
            errors.append("destructive execution must retain both gates")
        if errors:
            raise ProbeAuthoringError(
                f"{self.probe_point_id or '<missing point>'}: " + "; ".join(errors))

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "ProbeExecutionSpec":
        leaked = sorted(set(row) & _FORBIDDEN)
        if leaked:
            raise ProbeAuthoringError(
                f"probe execution cannot accept final labels: {leaked}")
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(row) - known)
        missing = sorted(known - set(row))
        if unknown or missing:
            raise ProbeAuthoringError(
                f"execution spec fields unknown={unknown} missing={missing}")
        materialized = dict(row)
        materialized["signal_channels"] = tuple(materialized["signal_channels"])
        materialized["undo_sequences"] = tuple(
            tuple(seq) for seq in materialized["undo_sequences"])
        materialized["solver_set"] = tuple(materialized["solver_set"])
        spec = cls(**materialized)
        spec.validate()
        return spec

    def to_probe_row(self) -> dict[str, Any]:
        """Flat execution row used by the CLI; review fields remain auditable."""
        row = self.to_dict()
        row["budget"] = row.pop("budget_k")
        return row


_EXECUTION_CONTEXT_FIELDS = {
    "probe_point_id", "probe_run_id", "candidate_snapshot_hash",
    "environment_family", "environment_instance", "environment_origin",
    "is_mock", "task_id", "trajectory_id", "run_id", "seed", "url",
    "account", "privilege", "code_version", "product_url", "submission_url",
    "forum_url",
}
_APPROVAL_FIELDS = {
    "fixture_status", "code_review_status", "reviewer", "review_timestamp",
}


def promote_authored_spec(
        spec: AuthoredProbeSpec, execution_context: Mapping[str, Any],
        approval: Mapping[str, Any]) -> ProbeExecutionSpec:
    """Promote one immutable draft only with explicit point context + review."""
    unknown_context = sorted(set(execution_context) - _EXECUTION_CONTEXT_FIELDS)
    unknown_approval = sorted(set(approval) - _APPROVAL_FIELDS)
    if unknown_context or unknown_approval:
        raise ProbeAuthoringError(
            f"promotion fields unknown_context={unknown_context} "
            f"unknown_approval={unknown_approval}")
    values = dict(execution_context)
    values.update(approval)
    promoted = ProbeExecutionSpec(
        schema_version=PROBE_EXECUTION_SCHEMA_VERSION,
        authored_spec_id=spec.spec_id,
        probe_name=spec.name, site=spec.site, action_type=spec.action_type,
        probe_point_id=str(values.get("probe_point_id") or ""),
        probe_run_id=str(values.get("probe_run_id") or ""),
        state_id=spec.state_id, candidate_id=spec.candidate_id,
        candidate_snapshot_hash=str(values.get("candidate_snapshot_hash") or ""),
        action_instance_id=spec.action_instance_id, raw_action=spec.raw_action,
        canonical_action=spec.canonical_action,
        environment_family=str(values.get("environment_family") or ""),
        environment_instance=str(values.get("environment_instance") or ""),
        environment_origin=str(values.get("environment_origin") or ""),
        is_mock=values.get("is_mock"), task_id=str(values.get("task_id") or ""),
        trajectory_id=str(values.get("trajectory_id") or ""),
        run_id=str(values.get("run_id") or ""), seed=values.get("seed"),
        url=str(values.get("url") or ""), account=str(values.get("account") or ""),
        privilege=str(values.get("privilege") or ""),
        signal_channels=spec.signal_channels, undo_sequences=spec.undo_sequences,
        solver_set=spec.solver_set, budget_k=spec.budget_k,
        safety_level=spec.safety_level, controller_version=spec.controller_version,
        code_version=str(values.get("code_version") or ""),
        fixture_status=str(values.get("fixture_status") or ""),
        code_review_status=str(values.get("code_review_status") or ""),
        reviewer=str(values.get("reviewer") or ""),
        review_timestamp=str(values.get("review_timestamp") or ""),
        requires_commit_flag=spec.requires_commit_flag,
        requires_destructive_env=spec.requires_destructive_env,
        product_url=str(values.get("product_url") or ""),
        submission_url=str(values.get("submission_url") or ""),
        forum_url=str(values.get("forum_url") or ""),
    )
    promoted.validate()
    return promoted


def load_probe_execution_specs(path: Path) -> list[ProbeExecutionSpec]:
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        rows = json.loads(text)
    else:
        # JSONL also starts with ``{``.  Parse a whole JSON object only when the
        # complete file is one document; ``Extra data`` falls back to one
        # object per non-empty line.
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            if "Extra data" not in str(exc):
                raise
            rows = [json.loads(line) for line in text.splitlines()
                    if line.strip()]
        else:
            rows = (payload.get("points", [payload])
                    if isinstance(payload, dict) else payload)
    if not isinstance(rows, list) or not rows:
        raise ProbeAuthoringError("execution spec must contain at least one point")
    specs = [ProbeExecutionSpec.from_dict(row) for row in rows]
    for attr in ("probe_name", "probe_point_id"):
        values = [str(getattr(spec, attr)) for spec in specs]
        if len(values) != len(set(values)):
            raise ProbeAuthoringError(f"duplicate execution {attr}")
    run_ids = {spec.probe_run_id for spec in specs}
    if len(run_ids) != 1:
        raise ProbeAuthoringError(
            f"one execution batch must share exactly one probe_run_id, got "
            f"{sorted(run_ids)}")
    return specs
