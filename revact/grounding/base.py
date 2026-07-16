"""Grounding core: probe protocol, registry, result schema, destructive gating.

A *probe* measures the reversibility of ONE action class by actually executing
the action in the live environment, running its undo controller, and comparing
a BACKEND signal (cart count, order set, wishlist count, ...) before/after.

Every probe declares:
  * ``destructive``: NON_DESTRUCTIVE (no lasting state change),
    SELF_RECOVERING (mutates state but the probe itself restores it, e.g.
    add-then-delete an address WE created), or DESTRUCTIVE (leaves a lasting
    change, e.g. a real order).
  * ``grounding``: which backend signal decides the label.
  * ``undo``: what the undo controller does.

Gating policy (enforced by ``run_probe``, not left to callers):
  DESTRUCTIVE probes execute their mutating step only when BOTH the explicit
  ``commit=True`` argument AND the environment opt-in
  ``REVACT_ALLOW_DESTRUCTIVE=1`` are present; otherwise they run in dry-run
  mode (navigate up to the mutating control, never click it) and return
  UNKNOWN with provenance.
"""
from __future__ import annotations

import os
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .. import config
from .schema import (EFFECT_UNKNOWN, RECOVERY_UNKNOWN,
                     GroundingPoint, GroundingValidationError,
                     legacy_label_to_statuses, save_probe_points,
                     statuses_to_display_label)

DESTRUCTIVE_ENV_GATE = "REVACT_ALLOW_DESTRUCTIVE"

NON_DESTRUCTIVE = "non_destructive"
SELF_RECOVERING = "self_recovering"
DESTRUCTIVE = "destructive"

# Deprecated display vocabulary for class-level smoke probes.  New runs map an
# old ``IRREVERSIBLE`` return to NOT_RECOVERED_WITHIN_BUDGET before persistence.
LABELS = ("REVERSIBLE", "PARTIALLY_RECOVERABLE",
          "NOT_RECOVERED_WITHIN_BUDGET", "NO_EFFECT", "UNKNOWN")


@dataclass
class ReversibilityResult:
    action_type: str
    label: str            # compatibility/display label, never new IRREVERSIBLE
    grounding: str        # backend signal used
    destructive: bool     # did this run mutate lasting state?
    evidence: dict = field(default_factory=dict)
    # provenance (defaults keep old JSONL rows loadable)
    probe_id: str = ""
    timestamp: str = ""
    commit_mode: bool = False
    site: str = ""
    probe_name: str = ""
    effect_status: str = EFFECT_UNKNOWN
    recovery_status: str = RECOVERY_UNKNOWN
    undo_cost_steps: Optional[int] = None
    budget_k: Optional[int] = None
    solver_set: list[str] = field(default_factory=list)
    budget_exhausted: bool = False
    controller_version: str = ""


def mk_result(action_type: str, label: str, grounding: str, destructive: bool,
              evidence: dict, commit_mode: bool = False, site: str = "",
              probe_name: str = "") -> ReversibilityResult:
    effect_status, recovery_status = legacy_label_to_statuses(label)
    undo_steps = evidence.get("undo_steps")
    if not isinstance(undo_steps, int) or undo_steps < 0:
        undo_steps = None
    return ReversibilityResult(
        action_type=action_type,
        label=statuses_to_display_label(effect_status, recovery_status),
        grounding=grounding,
        destructive=destructive, evidence=evidence,
        probe_id=f"{action_type}-{uuid.uuid4().hex[:8]}",
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        commit_mode=commit_mode, site=site, probe_name=probe_name,
        effect_status=effect_status, recovery_status=recovery_status,
        undo_cost_steps=undo_steps,
        budget_exhausted=bool(evidence.get("undo_budget_exhausted", False)),
        controller_version=config.CONTROLLER_VERSION,
    )


def destructive_allowed() -> bool:
    """Environment opt-in gate for destructive probes."""
    return os.environ.get(DESTRUCTIVE_ENV_GATE, "") == "1"


# --------------------------------------------------------------------------- #
# Probe context + spec + registry
# --------------------------------------------------------------------------- #
@dataclass
class ProbeContext:
    renv: Any                 # RevActEnv (or mock-wrapped equivalent)
    base: str                 # site base url (WA_SHOPPING / WA_REDDIT / ...)
    product_url: str = ""     # a valid product page (shopping probes)
    admin_base: str = ""      # WA_SHOPPING_ADMIN
    commit: bool = False      # explicit request to run the destructive step
    budget: int = 12          # undo-controller step budget k
    submission_url: str = ""  # a valid reddit submission page (reddit probes)
    forum_url: str = ""       # a valid reddit forum page (reddit probes)
    # Point identity is optional for legacy/class smoke probes and mandatory for
    # formal persistence.  Missing values are rejected, never synthesized.
    probe_point_id: str = ""
    probe_run_id: str = ""
    state_id: str = ""
    candidate_id: str = ""
    candidate_snapshot_hash: str = ""
    action_instance_id: str = ""
    raw_action: str = ""
    canonical_action: str = ""
    environment_family: str = "webarena"
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
    solver_set: list[str] = field(default_factory=lambda: ["site_specific_deterministic"])
    code_version: str = ""


@dataclass(frozen=True)
class ProbeSpec:
    name: str                 # registry key, e.g. "shopping.add_to_cart"
    site: str
    action_type: str
    destructive: str          # NON_DESTRUCTIVE | SELF_RECOVERING | DESTRUCTIVE
    grounding: str            # backend-signal description
    undo: str                 # undo-controller description
    fn: Callable[[ProbeContext], ReversibilityResult]
    expected_spectrum: str = ""  # documentation only, never used as a label


_REGISTRY: dict[str, ProbeSpec] = {}


def register(spec: ProbeSpec) -> ProbeSpec:
    if spec.name in _REGISTRY:
        raise ValueError(f"duplicate probe name {spec.name!r}")
    _REGISTRY[spec.name] = spec
    return spec


def get_probe(name: str) -> ProbeSpec:
    if name not in _REGISTRY:
        raise KeyError(f"unknown probe {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_probes(site: Optional[str] = None) -> list[ProbeSpec]:
    return [s for s in _REGISTRY.values() if site is None or s.site == site]


def run_probe(name: str, ctx: ProbeContext) -> ReversibilityResult:
    """Run one probe with gating + error containment.

    Never raises: an in-probe exception yields an UNKNOWN result carrying the
    traceback tail, so a batch run can proceed and the failure stays auditable.
    """
    spec = get_probe(name)
    gate_note = ""
    if spec.destructive == DESTRUCTIVE and ctx.commit and not destructive_allowed():
        ctx = ProbeContext(**{**asdict_ctx(ctx), "commit": False})
        gate_note = (f"commit requested but {DESTRUCTIVE_ENV_GATE}!=1 -> forced dry-run")
    try:
        res = spec.fn(ctx)
    except Exception:
        res = mk_result(spec.action_type, "UNKNOWN", spec.grounding,
                        destructive=False,
                        evidence={"error": traceback.format_exc(limit=3)[-800:]})
    res.site = spec.site
    res.probe_name = spec.name
    res.budget_k = ctx.budget
    res.solver_set = list(ctx.solver_set)
    res.controller_version = config.CONTROLLER_VERSION
    if gate_note:
        res.evidence.setdefault("gate_note", gate_note)
    return res


def asdict_ctx(ctx: ProbeContext) -> dict:
    return {"renv": ctx.renv, "base": ctx.base, "product_url": ctx.product_url,
            "admin_base": ctx.admin_base, "commit": ctx.commit,
            "budget": ctx.budget, "submission_url": ctx.submission_url,
            "forum_url": ctx.forum_url,
            "probe_point_id": ctx.probe_point_id,
            "probe_run_id": ctx.probe_run_id, "state_id": ctx.state_id,
            "candidate_id": ctx.candidate_id,
            "candidate_snapshot_hash": ctx.candidate_snapshot_hash,
            "action_instance_id": ctx.action_instance_id,
            "raw_action": ctx.raw_action,
            "canonical_action": ctx.canonical_action,
            "environment_family": ctx.environment_family,
            "environment_instance": ctx.environment_instance,
            "environment_origin": ctx.environment_origin,
            "is_mock": ctx.is_mock, "task_id": ctx.task_id,
            "trajectory_id": ctx.trajectory_id, "run_id": ctx.run_id,
            "seed": ctx.seed, "url": ctx.url, "account": ctx.account,
            "privilege": ctx.privilege, "solver_set": list(ctx.solver_set),
            "code_version": ctx.code_version}


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_results(results: list[ReversibilityResult],
                 out_dir: Optional[Path] = None) -> Path:
    """Append new *class-level smoke* results outside the frozen legacy asset.

    This function intentionally cannot create formal supervision.  Use
    :func:`save_formal_probe_results`, which requires a fully identified
    :class:`ProbeContext`, for point-level data.
    """
    import json

    base = Path(out_dir) if out_dir else config.DATA_ROOT
    # ``grounded/reversibility.jsonl`` and its 30-row manifest are historical
    # audit inputs.  Future smoke runs get a separate append-only namespace so
    # running the CLI cannot mutate the frozen baseline by accident.
    smoke_dir = base / "grounded" / "smoke"
    path = smoke_dir / "reversibility.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    _append_manifest(smoke_dir, results)
    return path


def _append_manifest(grounded_dir: Path, results: list[ReversibilityResult]) -> None:
    """One manifest line per probe run: enough to audit where labels came from."""
    import json

    path = grounded_dir / "MANIFEST.jsonl"
    with path.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({
                "probe_id": r.probe_id, "probe_name": r.probe_name,
                "site": r.site, "action_type": r.action_type, "label": r.label,
                "effect_status": r.effect_status,
                "recovery_status": r.recovery_status,
                "undo_cost_steps": r.undo_cost_steps,
                "budget_k": r.budget_k, "solver_set": r.solver_set,
                "timestamp": r.timestamp, "commit_mode": r.commit_mode,
                "controller_version": config.CONTROLLER_VERSION,
            }, ensure_ascii=False) + "\n")


def grounding_point_from_result(result: ReversibilityResult,
                                ctx: ProbeContext) -> GroundingPoint:
    """Build a formal point without fabricating missing transition evidence.

    Probe implementations must place observation hashes and measured signals in
    ``result.evidence``.  Older class probes do not, so conversion fails the
    formal validation gate and remains a smoke asset.
    """
    ev = dict(result.evidence or {})
    # The hash is provenance from the reviewed candidate execution spec.  It is
    # re-joined against the immutable candidate body+manifest before persistence,
    # so a forged or stale value fails closed.
    if ctx.candidate_snapshot_hash:
        ev.setdefault("candidate_snapshot_hash", ctx.candidate_snapshot_hash)
    point = GroundingPoint(
        probe_point_id=ctx.probe_point_id,
        probe_run_id=ctx.probe_run_id,
        probe_name=result.probe_name,
        state_id=ctx.state_id,
        candidate_id=ctx.candidate_id,
        action_instance_id=ctx.action_instance_id,
        action_type=result.action_type,
        raw_action=ctx.raw_action,
        canonical_action=ctx.canonical_action,
        site=result.site,
        environment_family=ctx.environment_family,
        environment_instance=ctx.environment_instance or ctx.base,
        environment_origin=ctx.environment_origin,
        is_mock=ctx.is_mock,
        task_id=ctx.task_id or str(getattr(ctx.renv, "task_id", "")),
        trajectory_id=(ctx.trajectory_id or
                       str(getattr(ctx.renv, "trajectory_id", ""))),
        run_id=ctx.run_id or str(getattr(ctx.renv, "run_id", "")),
        seed=ctx.seed,
        url=ctx.url or str(getattr(ctx.renv, "_last_obs_view", {}).get("url", "")),
        account=ctx.account,
        privilege=ctx.privilege,
        budget_k=result.budget_k or ctx.budget,
        solver_set=list(result.solver_set or ctx.solver_set),
        controller_version=result.controller_version or config.CONTROLLER_VERSION,
        pre_observation_hash=str(ev.get("pre_observation_hash", "")),
        pre_signal=ev.get("pre_signal", {}),
        post_observation_hash=str(ev.get("post_observation_hash", "")),
        post_signal=ev.get("post_signal", {}),
        undo_actions=list(ev.get("undo_actions") or []),
        undo_semantic_actions=list(ev.get("undo_semantic_actions") or []),
        undo_observation_hashes=list(ev.get("undo_observation_hashes") or []),
        final_signal=ev.get("final_signal", {}),
        effect_status=result.effect_status,
        recovery_status=result.recovery_status,
        undo_cost_steps=result.undo_cost_steps,
        residual_diff=ev.get("residual_diff", {}),
        budget_exhausted=result.budget_exhausted,
        timestamp=result.timestamp,
        code_version=ctx.code_version,
        evidence=ev,
    )
    point.validate(formal=True)
    return point


def save_formal_probe_results(results_with_context: list[
        tuple[ReversibilityResult, ProbeContext]],
        out_dir: Optional[Path] = None, *,
        transitions: Optional[list[Any]] = None) -> tuple[Path, Path]:
    """Persist formal point-level data after all provenance gates pass.

    New ``point_runner.v2`` results are fail-closed unless their exact AXTree
    transition bodies are supplied.  Legacy/class formal callers retain the
    old signature, but they cannot claim observation-body supervision.
    """
    base = Path(out_dir) if out_dir else config.DATA_ROOT
    points_path = base / "grounded" / "probe_points.jsonl"
    manifest_path = base / "grounded" / "POINT_MANIFEST.jsonl"
    points = [grounding_point_from_result(result, ctx)
              for result, ctx in results_with_context]
    from ..data.candidates import (CandidateValidationError,
                                   assert_candidate_manifest_integrity,
                                   validate_candidate_snapshot_artifact)
    from ..data.governance import point_candidate_reasons
    try:
        candidates = assert_candidate_manifest_integrity(
            base / "raw" / "candidates" / "iris_candidates.v3.jsonl")
        validate_candidate_snapshot_artifact(
            candidates, base / "raw" / "state_bank")
    except CandidateValidationError as exc:
        raise GroundingValidationError(
            f"formal candidate artifact invalid: {exc}") from exc
    join_problems = [
        f"{point.probe_point_id}: {reason}"
        for point in points
        for reason in point_candidate_reasons(point, candidates)
    ]
    if join_problems:
        raise GroundingValidationError(
            "formal point/candidate join failed: " +
            " | ".join(join_problems[:10]))
    v2_ids = {
        ctx.probe_point_id for result, ctx in results_with_context
        if (result.evidence or {}).get("protocol") == "point_runner.v2"
    }
    if v2_ids:
        from .transitions import (
            TRANSITION_BODY_RELATIVE, TRANSITION_MANIFEST_RELATIVE,
            ProbeTransition, TransitionValidationError,
            assert_point_transition_integrity, load_probe_transitions,
            save_probe_transitions, transition_manifest_row)
        supplied = list(transitions or [])
        if not all(isinstance(row, ProbeTransition) for row in supplied):
            raise GroundingValidationError(
                "point_runner.v2 requires typed ProbeTransition bodies")
        supplied_by_point = {row.probe_point_id: row for row in supplied}
        if set(supplied_by_point) != v2_ids or len(supplied_by_point) != len(supplied):
            raise GroundingValidationError(
                "point_runner.v2 point/transition batch mismatch: "
                f"point_only={sorted(v2_ids - set(supplied_by_point))}, "
                f"transition_only={sorted(set(supplied_by_point) - v2_ids)}")
        try:
            assert_point_transition_integrity(
                {point.probe_point_id: point for point in points
                 if point.probe_point_id in v2_ids},
                {row.transition_id: row for row in supplied}, require_all=True)
            for point in points:
                if point.probe_point_id not in v2_ids:
                    continue
                ref = (point.evidence or {}).get("transition_ref") or {}
                transition = supplied_by_point[point.probe_point_id]
                expected_manifest = transition_manifest_row(transition)
                if ref != {
                    "schema_version": transition.schema_version,
                    "transition_id": transition.transition_id,
                    "probe_point_id": transition.probe_point_id,
                    "record_sha256": expected_manifest["record_sha256"],
                }:
                    raise TransitionValidationError(
                        f"{point.probe_point_id}: transition_ref is stale or forged")
            # Preflight immutable collision checks before touching either asset.
            existing = load_probe_transitions(
                base / TRANSITION_BODY_RELATIVE, validate=True)
            collisions = sorted(
                row.transition_id for row in supplied
                if row.transition_id in existing or any(
                    old.probe_point_id == row.probe_point_id
                    for old in existing.values()))
            if collisions:
                raise TransitionValidationError(
                    f"immutable transition collisions: {collisions}")
            from .schema import assert_manifest_integrity, load_probe_points
            if points_path.exists() or manifest_path.exists():
                assert_manifest_integrity(points_path, manifest_path)
            existing_points = load_probe_points(points_path, validate=True)
            point_collisions = sorted(
                point.probe_point_id for point in points
                if point.probe_point_id in existing_points)
            if point_collisions:
                raise TransitionValidationError(
                    f"immutable point collisions: {point_collisions}")
        except TransitionValidationError as exc:
            raise GroundingValidationError(
                f"formal transition artifact invalid: {exc}") from exc

        # Persist the body evidence first.  A process crash between the two
        # atomic writes can create a visible orphan transition, never a point
        # that falsely claims a missing transition.  Cross-asset integrity
        # gates reject orphans until the append is retried/audited.
        save_probe_transitions(
            supplied, base / TRANSITION_BODY_RELATIVE,
            base / TRANSITION_MANIFEST_RELATIVE, append=True)
    return save_probe_points(points, points_path, manifest_path, append=True)


def load_reversibility(path: Path) -> dict:
    """Legacy class-level loader; forbidden as a formal supervision join.

    Kept only for smoke-probe visualization and backwards-compatible tests.
    Formal assembly must load ``probe_points.jsonl`` and join a unique
    ``probe_point_id``.
    """
    return {at: r["label"] for at, r in load_reversibility_details(path).items()}


def load_reversibility_details(path: Path) -> dict:
    """action_type -> latest-non-UNKNOWN grounded record with the undo
    evidence (undo_steps / undo_actions / residual_diff) that the P2 sample
    format consumes when constructing <rev_check>/<undo>.

    Returns {action_type: {label, grounding, undo_steps, undo_actions,
                           residual_diff, probe_id}}.
    """
    import json

    rows_by_type: dict[str, list] = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            rows_by_type.setdefault(r["action_type"], []).append(r)
    out = {}
    for at, rows in rows_by_type.items():
        grounded = [r for r in rows if r.get("label") != "UNKNOWN"]
        pick = grounded[-1] if grounded else rows[-1]
        ev = pick.get("evidence") or {}
        out[at] = {
            "label": pick["label"],
            "grounding": pick.get("grounding", ""),
            "undo_steps": ev.get("undo_steps"),
            "undo_actions": ev.get("undo_actions") or [],
            "residual_diff": ev.get("residual_diff"),
            "probe_id": pick.get("probe_id", ""),
        }
    return out
