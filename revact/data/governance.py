"""Read-only integrity audits for IRIS data lineage and formal-set hygiene.

The workbench may browse legacy and quarantine artifacts, but a *formal*
release must never infer provenance which is absent from a row.  The release
gate below therefore validates the exact point-level join and immutable prompt
bundle before an SFT/DPO row can be copied to an export.
"""
from __future__ import annotations

import json
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .candidates import (FORMAL_CANDIDATE_ARTIFACT_VERSION,
                         FORMAL_CANDIDATE_BODY_NAME,
                         FORMAL_CANDIDATE_MANIFEST_NAME, Candidate,
                         CandidateValidationError,
                         assert_candidate_manifest_integrity,
                         validate_candidate_snapshot_artifact)
from ..grounding.schema import (GroundingPoint, GroundingValidationError,
                                assert_manifest_integrity, load_probe_points)
from ..eval.truth import (EvaluationTruthError, EvaluationTruthRecord,
                          assert_truth_manifest_integrity, load_truth_records)
from ..prompt_store import load_bundle, load_generation_bundle


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.open(encoding="utf-8") if ln.strip()]


@dataclass(frozen=True)
class FormalReleaseContext:
    """Validated, immutable inputs used by the formal export admission gate."""

    root: Path
    points: dict[str, GroundingPoint] = field(default_factory=dict)
    grounding_error: str = ""
    candidates: dict[str, Candidate] = field(default_factory=dict)
    candidate_error: str = ""
    candidate_asset_tier: str = FORMAL_CANDIDATE_ARTIFACT_VERSION
    truth: dict[str, EvaluationTruthRecord] = field(default_factory=dict)
    truth_error: str = ""


def formal_release_context(root: Path) -> FormalReleaseContext:
    """Load the canonical point body only when its manifest is exactly 1:1.

    An entirely empty body+manifest is a valid pre-data state (no row can pass
    because no point id can resolve).  A missing half, corrupt row, hash drift,
    or duplicate id invalidates the grounding source for the whole release.
    """
    root = Path(root)
    points_path = root / "grounded" / "probe_points.jsonl"
    manifest_path = root / "grounded" / "POINT_MANIFEST.jsonl"
    if points_path.exists() != manifest_path.exists():
        return FormalReleaseContext(
            root=root,
            grounding_error="formal grounding body/manifest must either both exist or both be absent",
        )
    if not points_path.exists():
        return FormalReleaseContext(root=root)
    try:
        assert_manifest_integrity(points_path, manifest_path)
        points = load_probe_points(points_path, validate=True)
    except (GroundingValidationError, json.JSONDecodeError, OSError, TypeError) as exc:
        return FormalReleaseContext(root=root, grounding_error=str(exc))
    candidate_path = root / "raw" / "candidates" / FORMAL_CANDIDATE_BODY_NAME
    candidate_manifest = candidate_path.with_name(
        FORMAL_CANDIDATE_MANIFEST_NAME)
    candidates: dict[str, Candidate] = {}
    candidate_error = ""
    if points:
        try:
            candidates = assert_candidate_manifest_integrity(
                candidate_path, candidate_manifest)
            validate_candidate_snapshot_artifact(
                candidates, root / "raw" / "state_bank")
        except (CandidateValidationError, json.JSONDecodeError, OSError,
                TypeError) as exc:
            candidate_error = (
                f"{FORMAL_CANDIDATE_ARTIFACT_VERSION}: {exc}; "
                "legacy iris_candidates.v3 is not a formal fallback")

    truth_path = root / "eval" / "truth.jsonl"
    truth_manifest = root / "eval" / "TRUTH_MANIFEST.jsonl"
    if truth_path.exists() != truth_manifest.exists():
        return FormalReleaseContext(
            root=root, points=points, candidates=candidates,
            candidate_error=candidate_error,
            truth_error="evaluation truth body/manifest must both exist")
    if not truth_path.exists():
        return FormalReleaseContext(
            root=root, points=points, candidates=candidates,
            candidate_error=candidate_error)
    try:
        assert_truth_manifest_integrity(truth_path, truth_manifest, points)
        truth = load_truth_records(truth_path, points=points)
    except (EvaluationTruthError, json.JSONDecodeError, OSError, TypeError) as exc:
        return FormalReleaseContext(
            root=root, points=points, candidates=candidates,
            candidate_error=candidate_error, truth_error=str(exc))
    return FormalReleaseContext(
        root=root, points=points, candidates=candidates,
        candidate_error=candidate_error, truth=truth)


_FORMAL_REQUIRED_TEXT = (
    "probe_point_id",
    "probe_run_id",
    "state_id",
    "candidate_id",
    "action_instance_id",
    "action_type",
    "site",
    "environment_origin",
    "environment_instance",
    "trajectory_id",
    "run_id",
    "privilege",
    "evaluation_case_id",
    "normative_truth_source",
    "normative_policy_id",
    "normative_policy_version",
    "prompts_fp",
    "prompt_generation_fp",
)

_DERIVED_IDENTITY_KEYS = (
    "probe_point_id",
    "probe_run_id",
    "state_id",
    "candidate_id",
    "action_instance_id",
    "action_type",
    "site",
    "environment_origin",
    "environment_instance",
    "trajectory_id",
    "run_id",
    "privilege",
    "evaluation_case_id",
    "effect_status",
    "recovery_status",
    "undo_cost_steps",
    "prompts_fp",
    "prompt_generation_fp",
)


def point_candidate_reasons(
        point: GroundingPoint, candidates: dict[str, Candidate],
        candidate_error: str = "") -> list[str]:
    """Verify the executed point against the exact S4 snapshot candidate."""
    if candidate_error:
        return ["formal_candidate_artifact_invalid"]
    candidate = candidates.get(point.candidate_id)
    if candidate is None:
        return ["unknown_source_candidate_id"]
    mismatched = []
    expected = {
        "state_id": point.state_id,
        "canonical_action": point.canonical_action,
        "snapshot_hash": str(
            (point.evidence or {}).get("candidate_snapshot_hash") or ""),
        "legal_at_snapshot": True,
    }
    for key, value in expected.items():
        if getattr(candidate, key) != value:
            mismatched.append(key)
    from ..train.validators import parse_action
    parsed = parse_action(point.raw_action)
    if parsed is None or parsed.bid != candidate.bid:
        mismatched.append("bid")
    expected_primitive = candidate.canonical_action.split(":", 1)[0]
    if parsed is None or parsed.name != expected_primitive:
        mismatched.append("action_primitive")
    return (["source_candidate_mismatch:" + ",".join(mismatched)]
            if mismatched else [])


def formal_release_reasons(meta: dict, context: FormalReleaseContext) -> list[str]:
    """Return every reason a row is ineligible for a formal release.

    This is deliberately stricter than a trainer's syntax validator.  Formal
    export is the publication boundary, so missing booleans are not treated as
    false/true defaults and point facts must match the canonical grounding row.
    Reason codes are stable and are written to ``excluded.jsonl``.
    """
    meta = meta if isinstance(meta, dict) else {}
    reasons: list[str] = []

    if meta.get("formal_dataset") is not True:
        reasons.append("formal_dataset_not_true")
    if meta.get("is_mock") is not False:
        reasons.append("is_mock_not_explicitly_false")
    if meta.get("collector_success") is not True:
        reasons.append("collector_success_not_true")
    if meta.get("history_source") != "trajectory":
        reasons.append("history_source_not_trajectory")
    if meta.get("prediction_source") != "probe_transition":
        reasons.append("prediction_source_not_probe_transition")
    if meta.get("undo_source") != "probe_point_id":
        reasons.append("undo_source_not_probe_point_id")
    if meta.get("reversibility_grounded") is not True:
        reasons.append("reversibility_grounded_not_true")
    if meta.get("format") != "iris.v3":
        reasons.append("format_not_iris_v3")
    if not isinstance(meta.get("normative_risk"), bool):
        reasons.append("normative_risk_not_boolean")

    missing = [key for key in _FORMAL_REQUIRED_TEXT
               if not str(meta.get(key) or "").strip()]
    if missing:
        reasons.append("missing_formal_provenance:" + ",".join(missing))

    point_id = str(meta.get("probe_point_id") or "")
    if str(meta.get("undo_source_probe_point_id") or "") != point_id:
        reasons.append("undo_point_id_mismatch")

    if context.grounding_error:
        reasons.append("formal_grounding_artifact_invalid")
    elif point_id:
        point = context.points.get(point_id)
        if point is None:
            reasons.append("unknown_probe_point_id")
        else:
            # These are the sample-side fields materialized by both the single
            # and multi-turn assemblers.  Exact equality makes the point id a
            # verified join, rather than an untrusted decorative string.
            expected = {
                "probe_run_id": point.probe_run_id,
                "state_id": point.state_id,
                "candidate_id": point.candidate_id,
                "action_instance_id": point.action_instance_id,
                "action_type": point.action_type,
                "site": point.site,
                "environment_origin": point.environment_origin,
                "environment_instance": point.environment_instance,
                "trajectory_id": point.trajectory_id,
                "run_id": point.run_id,
                "privilege": point.privilege,
                "account": point.account,
                "task_id": point.task_id,
                "url": point.url,
                "seed": point.seed,
                "environment_family": point.environment_family,
                "is_mock": point.is_mock,
                "effect_status": point.effect_status,
                "recovery_status": point.recovery_status,
                "undo_cost_steps": point.undo_cost_steps,
                "canonical_action": point.canonical_action,
                "risky_raw_action": point.raw_action,
                "pre_observation_hash": point.pre_observation_hash,
                "post_observation_hash": point.post_observation_hash,
                "post_signal_diff": {
                    "pre_signal": point.pre_signal,
                    "post_signal": point.post_signal,
                },
                "undo_actions": list(point.undo_actions),
                "undo_semantic_actions": list(point.undo_semantic_actions),
                "undo_observation_hashes": list(point.undo_observation_hashes),
                "residual_diff": point.residual_diff,
                "budget_k": point.budget_k,
                "solver_set": list(point.solver_set),
                "candidate_snapshot_hash": str(
                    (point.evidence or {}).get("candidate_snapshot_hash") or ""),
                "evidence": dict(point.evidence),
            }
            mismatched = [key for key, value in expected.items()
                          if meta.get(key) != value]
            if mismatched:
                reasons.append("probe_provenance_mismatch:" + ",".join(mismatched))
            reasons.extend(point_candidate_reasons(
                point, context.candidates, context.candidate_error))

    case_id = str(meta.get("evaluation_case_id") or "")
    if context.truth_error:
        reasons.append("evaluation_truth_artifact_invalid")
    elif case_id:
        truth = context.truth.get(case_id)
        if truth is None:
            reasons.append("unknown_evaluation_case_id")
        else:
            expected_truth = {
                "probe_point_id": truth.probe_point_id,
                "state_id": truth.state_id,
                "effect_status": truth.effect_status,
                "recovery_status": truth.recovery_status,
                "variant": truth.variant,
                "normative_risk": truth.normative_risk,
                "policy_constraint_truth": truth.policy_constraint_truth,
                "action_required_for_goal": truth.action_required_for_goal,
                "violates_constraint": truth.violates_constraint,
                "decision": truth.expected_decision,
                "normative_truth_source": truth.truth_source,
                "normative_policy_id": truth.policy_id,
                "normative_policy_version": truth.policy_version,
            }
            mismatched = [key for key, value in expected_truth.items()
                          if meta.get(key) != value]
            if mismatched:
                reasons.append("evaluation_truth_mismatch:" + ",".join(mismatched))

    prompt_fp = str(meta.get("prompts_fp") or "")
    if prompt_fp:
        try:
            load_bundle(prompt_fp, root=context.root)
        except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError, TypeError):
            reasons.append("prompt_bundle_missing_or_invalid")

    generation_fp = str(meta.get("prompt_generation_fp") or "")
    if generation_fp:
        try:
            generation = load_generation_bundle(
                generation_fp, root=context.root)
            if generation.get("prompts_fp") != prompt_fp:
                reasons.append("prompt_generation_bundle_prompt_mismatch")
        except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError,
                TypeError):
            reasons.append("prompt_generation_bundle_missing_or_invalid")

    return reasons


def formal_derivation_reasons(source_meta: dict, derived_meta: dict) -> list[str]:
    """Prevent teacher/DPO artifacts from rebinding a source sample.

    Both rows must independently pass :func:`formal_release_reasons`; this
    additional check pins the derived row to the exact source point and prompt
    provenance rather than accepting a different, individually-valid point.
    """
    source_meta = source_meta if isinstance(source_meta, dict) else {}
    derived_meta = derived_meta if isinstance(derived_meta, dict) else {}
    mismatched = [key for key in _DERIVED_IDENTITY_KEYS
                  if source_meta.get(key) != derived_meta.get(key)]
    return (["derived_provenance_mismatch:" + ",".join(mismatched)]
            if mismatched else [])


def formal_prompt_content_reasons(row: dict,
                                  context: FormalReleaseContext) -> list[str]:
    """Verify immutable prompt text and gold/chosen completion consistency."""
    from ..train.validators import (formal_completion_reasons,
                                    formal_negative_candidate_reasons)
    reasons = formal_completion_reasons(row)
    reasons.extend(formal_negative_candidate_reasons(
        row, context.candidates if not context.candidate_error else {}))
    meta = row.get("meta") if isinstance(row, dict) else {}
    meta = meta if isinstance(meta, dict) else {}
    fp = str(meta.get("prompts_fp") or "")
    if not fp:
        return reasons  # missing provenance is reported by the metadata gate
    try:
        bundle = load_bundle(fp, root=context.root)
    except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError, TypeError):
        return reasons  # same error is reported by formal_release_reasons
    messages = row.get("messages") or row.get("prompt") or []
    system_rows = [message for message in messages
                   if isinstance(message, dict) and message.get("role") == "system"]
    expected = (bundle.get("prompts") or {}).get("agent_system")
    if len(system_rows) != 1 or not isinstance(expected, str) \
            or system_rows[0].get("content") != expected:
        reasons.append("prompt_bundle_system_mismatch")
    return reasons


def audit_collection_lineage(root: Path, site: str = "shopping") -> dict:
    """Check the historical raw↔meta↔key-state inventory.

    This is a repository-health diagnostic, not the publication gate.  Frozen
    legacy rows remain visible here (including their duplicate logical IDs),
    while :func:`audit_formal_collection_lineage` validates only the immutable
    run closures actually referenced by formal samples.
    """
    root = Path(root)
    meta = _jsonl(root / "raw" / "trajectories_meta.jsonl")
    # ``out_dir`` collection fixtures place raw artifacts directly under root;
    # production places them below root/raw.
    raw_dir = root / "raw" / "trajectories"
    state_path = root / "raw" / "state_bank" / f"{site}_key_states.jsonl"
    if not raw_dir.exists():
        raw_dir = root / "trajectories"
        state_path = root / "state_bank" / f"{site}_key_states.jsonl"
        if not meta:
            meta = _jsonl(root / "trajectories_meta.jsonl")
    states = _jsonl(state_path)
    raw_files = {p.stem: p for p in raw_dir.glob("*.jsonl")} if raw_dir.exists() else {}

    meta_by_tid: dict[str, list[dict]] = defaultdict(list)
    state_by_tid: dict[str, list[dict]] = defaultdict(list)
    for row in meta:
        meta_by_tid[str(row.get("trajectory_id", ""))].append(row)
    for row in states:
        state_by_tid[str(row.get("trajectory_id", ""))].append(row)

    duplicate_meta = sorted(tid for tid, rows in meta_by_tid.items() if tid and len(rows) > 1)
    missing_raw = sorted(tid for tid in meta_by_tid if tid and tid not in raw_files)
    missing_meta = sorted(tid for tid in raw_files if tid not in meta_by_tid)
    run_mismatch, missing_run_new = [], []
    missing_policy_attempt_artifacts: list[str] = []
    invalid_policy_attempt_artifacts: list[str] = []
    for tid, rows in meta_by_tid.items():
        for m in rows:
            run_id = str(m.get("run_id", ""))
            if "__run_" in tid and not run_id:
                missing_run_new.append(tid)
            path = raw_files.get(tid)
            if not path or not run_id:
                continue
            steps = _jsonl(path)
            if any(str(s.get("run_id", "")) != run_id for s in steps):
                run_mismatch.append(tid)
            if any(str(s.get("trajectory_id", "")) != tid for s in steps):
                run_mismatch.append(tid)
            capture = m.get("policy_attempt_source_capture")
            if not isinstance(capture, dict):
                continue  # frozen rows collected before the sidecar existed
            artifact = str(m.get("policy_attempt_artifact") or "")
            attempt_path = None
            if artifact:
                candidate = (root / artifact).resolve()
                try:
                    candidate.relative_to(root.resolve())
                except ValueError:
                    invalid_policy_attempt_artifacts.append(tid)
                    continue
                attempt_path = candidate
            if attempt_path is None or not attempt_path.is_file():
                missing_policy_attempt_artifacts.append(tid)
                continue
            try:
                attempts = _jsonl(attempt_path)
                declared_n = capture.get("n_policy_attempts")
                declared_unexecuted = capture.get(
                    "n_unexecuted_policy_attempts")
                invalid = (
                    declared_n != len(attempts)
                    or declared_unexecuted != sum(
                        row.get("execution_status") == "NO_ACTION"
                        for row in attempts)
                    or any(str(row.get("run_id") or "") != run_id
                           or str(row.get("trajectory_id") or "") != tid
                           or str(row.get("task_id") or "") !=
                           str(m.get("task_id") or "")
                           for row in attempts)
                )
                for attempt in attempts:
                    messages = attempt.get("policy_input_messages") or []
                    completion = str(
                        attempt.get("proposed_completion") or "")
                    executed = str(
                        attempt.get("executed_completion") or "")
                    if messages and attempt.get(
                            "policy_input_messages_sha256") != hashlib.sha256(
                                json.dumps(
                                    messages, ensure_ascii=False,
                                    sort_keys=True, separators=(",", ":"),
                                    allow_nan=False,
                                ).encode("utf-8")).hexdigest():
                        invalid = True
                    if completion and attempt.get(
                            "proposed_completion_sha256") != hashlib.sha256(
                                completion.encode("utf-8")).hexdigest():
                        invalid = True
                    if executed and attempt.get(
                            "executed_completion_sha256") != hashlib.sha256(
                                executed.encode("utf-8")).hexdigest():
                        invalid = True
                if invalid:
                    invalid_policy_attempt_artifacts.append(tid)
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                invalid_policy_attempt_artifacts.append(tid)
    orphan_state_runs = sorted({
        tid for tid, rows in state_by_tid.items()
        if any(r.get("run_id") for r in rows) and tid not in meta_by_tid})
    state_run_mismatch = sorted({
        tid for tid, rows in state_by_tid.items()
        for r in rows
        if r.get("run_id") and meta_by_tid.get(tid)
        and str(r.get("run_id")) not in {str(m.get("run_id")) for m in meta_by_tid[tid]}})

    manifests_dir = root / "manifests" / "collection_runs"
    manifest_ids = {p.stem for p in manifests_dir.glob("*.json")} \
        if manifests_dir.exists() else set()
    incomplete_manifests = []
    invalid_manifests = []
    if manifests_dir.exists():
        for path in sorted(manifests_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, TypeError):
                invalid_manifests.append(path.stem)
                continue
            # Historical v1 manifests had no status but were written only at
            # the end; they remain readable.  New v2 records must be COMPLETE.
            if payload.get("schema_version") == "iris.collection-run.v2" \
                    and payload.get("status") != "COMPLETE":
                incomplete_manifests.append(path.stem)
    run_ids = {str(m.get("run_id")) for m in meta if m.get("run_id")}
    missing_manifest = sorted(run_ids - manifest_ids)
    counts = Counter(str(m.get("environment_origin") or "unknown") for m in meta)
    unknown_origins = [
        f"line-{index}:{str(row.get('trajectory_id') or '<missing>')}"
        for index, row in enumerate(meta, 1)
        if str(row.get("environment_origin") or "unknown").strip().lower()
        in {"", "unknown"}
    ]
    problems = (len(duplicate_meta) + len(missing_raw) + len(missing_meta)
                + len(set(run_mismatch)) + len(missing_run_new)
                + len(orphan_state_runs) + len(state_run_mismatch)
                + len(missing_manifest) + len(incomplete_manifests)
                + len(invalid_manifests) + len(unknown_origins)
                + len(missing_policy_attempt_artifacts)
                + len(invalid_policy_attempt_artifacts))
    return {
        "n_meta_rows": len(meta), "n_unique_trajectory_ids": len(meta_by_tid),
        "n_raw_files": len(raw_files), "n_key_states": len(states),
        "environment_origin": dict(counts),
        "unknown_environment_origin_rows": unknown_origins,
        "duplicate_meta_trajectory_ids": duplicate_meta,
        "meta_without_raw": missing_raw, "raw_without_meta": missing_meta,
        "missing_run_id_new": sorted(set(missing_run_new)),
        "raw_meta_run_mismatch": sorted(set(run_mismatch)),
        "orphan_state_runs": orphan_state_runs,
        "state_meta_run_mismatch": state_run_mismatch,
        "missing_collection_manifests": missing_manifest,
        "incomplete_collection_manifests": incomplete_manifests,
        "invalid_collection_manifests": invalid_manifests,
        "missing_policy_attempt_artifacts": sorted(set(
            missing_policy_attempt_artifacts)),
        "invalid_policy_attempt_artifacts": sorted(set(
            invalid_policy_attempt_artifacts)),
        "n_problems": problems, "ok": problems == 0,
    }


def audit_formal_collection_lineage(root: Path,
                                    formal_rows: list[dict]) -> dict:
    """Validate the collection closure referenced by the formal release rows.

    The historical collector appended several attempts under the same logical
    trajectory ID.  Those rows are quarantined and must not permanently make a
    clean future release impossible.  Conversely, accepting an arbitrary
    ``run_id`` string on a point is too weak.  For every formal sample this gate
    therefore requires one exact chain:

    ``COMPLETE run manifest -> one meta row -> one raw trajectory -> one key
    state``, with matching run/trajectory/state identities, a known environment
    origin, and successful-collector provenance.
    """
    root = Path(root)
    rows = [row for row in formal_rows
            if (row.get("meta") or {}).get("formal_dataset") is True]
    raw_root = root / "raw"
    raw_dir = raw_root / "trajectories"
    meta_path = raw_root / "trajectories_meta.jsonl"
    state_dir = raw_root / "state_bank"
    if not raw_dir.exists() and (root / "trajectories").exists():
        # Collection unit tests use an artifact root without the production
        # ``raw/`` prefix.
        raw_root = root
        raw_dir = root / "trajectories"
        meta_path = root / "trajectories_meta.jsonl"
        state_dir = root / "state_bank"

    meta_rows = _jsonl(meta_path)
    # ``formal_point_reached_states.jsonl`` is a derived assembler view of the
    # immutable collection row, not a second collection event.  Counting it in
    # the raw closure makes a valid single/multiturn dual view appear to have a
    # two-way state join.  Only source collection banks participate here.
    state_rows = [row for path in sorted(state_dir.glob("*.jsonl"))
                  if path.name != "formal_point_reached_states.jsonl"
                  for row in _jsonl(path)] if state_dir.exists() else []
    meta_by_identity: dict[tuple[str, str], list[dict]] = defaultdict(list)
    state_by_identity: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in meta_rows:
        meta_by_identity[(str(row.get("run_id") or ""),
                          str(row.get("trajectory_id") or ""))].append(row)
    for row in state_rows:
        state_by_identity[(str(row.get("run_id") or ""),
                           str(row.get("trajectory_id") or ""),
                           str(row.get("state_id") or ""))].append(row)

    issues: dict[str, list[str]] = defaultdict(list)

    def flag(kind: str, ident: str) -> None:
        if ident not in issues[kind]:
            issues[kind].append(ident)

    closures: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id") or f"row-{index}")
        meta = row.get("meta") or {}
        run_id = str(meta.get("run_id") or "")
        trajectory_id = str(meta.get("trajectory_id") or "")
        state_id = str(meta.get("state_id") or "")
        origin = str(meta.get("environment_origin") or "").strip()
        missing = [name for name, value in (
            ("run_id", run_id), ("trajectory_id", trajectory_id),
            ("state_id", state_id), ("environment_origin", origin),
        ) if not value]
        if missing:
            flag("missing_formal_fields", f"{sample_id}:{','.join(missing)}")
        if origin.lower() in {"", "unknown"}:
            flag("unknown_environment_origin", sample_id)
        if meta.get("collector_success") is not True:
            flag("formal_collector_not_successful", sample_id)
        if run_id and trajectory_id and state_id:
            closures[(run_id, trajectory_id, state_id)].append(sample_id)

    manifests_dir = root / "manifests" / "collection_runs"
    for (run_id, trajectory_id, state_id), sample_ids in sorted(closures.items()):
        ident = f"{run_id}/{trajectory_id}/{state_id}"
        metas = meta_by_identity.get((run_id, trajectory_id), [])
        if len(metas) != 1:
            flag("meta_join_not_one_to_one", f"{ident}:matches={len(metas)}")
        else:
            meta = metas[0]
            meta_origin = str(meta.get("environment_origin") or "").strip()
            if meta_origin.lower() in {"", "unknown"}:
                flag("unknown_environment_origin", f"{ident}:meta")
            if meta.get("collector_success") is not True or meta.get("success") is not True:
                flag("meta_collector_not_successful", ident)
            formal_origins = {
                str((row.get("meta") or {}).get("environment_origin") or "")
                for row in rows
                if str((row.get("meta") or {}).get("run_id") or "") == run_id
                and str((row.get("meta") or {}).get("trajectory_id") or "") == trajectory_id
                and str((row.get("meta") or {}).get("state_id") or "") == state_id
            }
            if formal_origins != {meta_origin}:
                flag("environment_origin_mismatch", ident)

        raw_path = raw_dir / f"{trajectory_id}.jsonl"
        raw_rows = _jsonl(raw_path)
        if not raw_path.exists():
            flag("missing_raw_trajectory", ident)
        elif not raw_rows:
            flag("empty_raw_trajectory", ident)
        elif any(str(step.get("run_id") or "") != run_id or
                 str(step.get("trajectory_id") or "") != trajectory_id
                 for step in raw_rows):
            flag("raw_identity_mismatch", ident)

        states = state_by_identity.get((run_id, trajectory_id, state_id), [])
        if len(states) != 1:
            flag("state_join_not_one_to_one", f"{ident}:matches={len(states)}")
        else:
            state = states[0]
            state_origin = str(state.get("environment_origin") or "").strip()
            if state_origin.lower() in {"", "unknown"}:
                flag("unknown_environment_origin", f"{ident}:state")
            formal_origins = {
                str((row.get("meta") or {}).get("environment_origin") or "")
                for row in rows
                if str((row.get("meta") or {}).get("run_id") or "") == run_id
                and str((row.get("meta") or {}).get("trajectory_id") or "")
                == trajectory_id
                and str((row.get("meta") or {}).get("state_id") or "") == state_id
            }
            if formal_origins != {state_origin}:
                flag("environment_origin_mismatch", f"{ident}:state")
            if state.get("collector_success") is not True or \
                    state.get("traj_success") is not True:
                flag("state_collector_not_successful", ident)

        manifest_path = manifests_dir / f"{run_id}.json"
        if not manifest_path.exists():
            flag("missing_run_manifest", ident)
        else:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, TypeError):
                flag("invalid_run_manifest", ident)
            else:
                if manifest.get("schema_version") != "iris.collection-run.v2" or \
                        manifest.get("run_id") != run_id:
                    flag("run_manifest_identity_mismatch", ident)
                if manifest.get("status") != "COMPLETE":
                    flag("run_manifest_not_complete", ident)
                summaries = [summary for summary in manifest.get("trajectories", [])
                             if str(summary.get("trajectory_id") or "") == trajectory_id]
                if len(summaries) != 1:
                    flag("manifest_trajectory_join_not_one_to_one",
                         f"{ident}:matches={len(summaries)}")
                else:
                    summary = summaries[0]
                    if str(summary.get("run_id") or "") != run_id:
                        flag("manifest_trajectory_run_mismatch", ident)
                    if summary.get("collector_success") is not True or \
                            summary.get("success") is not True:
                        flag("manifest_collector_not_successful", ident)
                    summary_origin = str(
                        summary.get("environment_origin") or "").strip()
                    if summary_origin.lower() in {"", "unknown"}:
                        flag("unknown_environment_origin", f"{ident}:manifest")
                    formal_origins = {
                        str((row.get("meta") or {}).get(
                            "environment_origin") or "")
                        for row in rows
                        if str((row.get("meta") or {}).get("run_id") or "") == run_id
                        and str((row.get("meta") or {}).get(
                            "trajectory_id") or "") == trajectory_id
                        and str((row.get("meta") or {}).get("state_id") or "")
                        == state_id
                    }
                    if formal_origins != {summary_origin}:
                        flag("environment_origin_mismatch", f"{ident}:manifest")

    normalized_issues = {key: sorted(value) for key, value in sorted(issues.items())}
    n_problems = sum(len(value) for value in normalized_issues.values())
    return {
        "scope": "formal_samples_only",
        "legacy_quarantine_excluded_from_gate": True,
        "evaluated": bool(rows),
        "n_formal_rows": len(rows),
        "n_unique_run_ids": len({key[0] for key in closures}),
        "n_unique_trajectory_ids": len({key[1] for key in closures}),
        "n_unique_state_ids": len({key[2] for key in closures}),
        "issues": normalized_issues,
        "n_problems": n_problems,
        "ok": bool(rows) and n_problems == 0,
    }
