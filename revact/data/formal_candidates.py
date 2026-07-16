"""Point-scoped, immutable formal S4 candidate materialization.

The historical ``iris_candidates.v3.jsonl`` file is a broad proposal census and
remains the intake asset used while collecting a point.  This module creates a
separate publication asset only after canonical point evidence exists.  Every
state in the output is therefore reachable from an exact point foreign key,
while every point's executed expert candidate keeps its original identity.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .candidates import (
    CATEGORY_DECOY,
    CATEGORY_EXPERT,
    CATEGORY_GOAL_VIOLATING,
    CATEGORY_POLICY_ERROR,
    CATEGORY_SAFE_ALTERNATIVE,
    FORMAL_CANDIDATE_ARTIFACT_VERSION,
    FORMAL_CANDIDATE_BODY_NAME,
    FORMAL_CANDIDATE_MANIFEST_NAME,
    LEGACY_CANDIDATE_BODY_NAME,
    Candidate,
    CandidateValidationError,
    assert_candidate_manifest_integrity,
    build_a11y_candidate_set,
    snapshot_sha256,
    validate_candidate_set,
    write_candidate_manifest,
)
from ..grounding.schema import (
    GroundingValidationError,
    assert_manifest_integrity,
    load_probe_points,
)
from ..grounding.transitions import (ProbeTransition,
                                     assert_point_transition_integrity,
                                     load_probe_transitions)
from ..train.validators import parse_action


FORMAL_CANDIDATE_REPORT_SCHEMA = "iris.formal_candidate_asset_report.v1"
FORMAL_SELECTOR_VERSION = "a11y-rule-v2-boundary-locality-fix1"
_EVIDENCE_DEPENDENT_CATEGORIES = frozenset({
    CATEGORY_SAFE_ALTERNATIVE,
    CATEGORY_DECOY,
    CATEGORY_GOAL_VIOLATING,
    CATEGORY_POLICY_ERROR,
})


class FormalCandidateMaterializationError(ValueError):
    """The canonical points cannot produce one immutable formal S4 asset."""


def formal_candidate_paths(data_root: Path) -> tuple[Path, Path]:
    directory = Path(data_root) / "raw" / "candidates"
    return (directory / FORMAL_CANDIDATE_BODY_NAME,
            directory / FORMAL_CANDIDATE_MANIFEST_NAME)


def legacy_candidate_path(data_root: Path) -> Path:
    return Path(data_root) / "raw" / "candidates" / LEGACY_CANDIDATE_BODY_NAME


def _snapshot_index(state_bank_dir: Path) -> tuple[
        dict[tuple[str, str], str], dict[tuple[str, str], list[str]]]:
    snapshots: dict[tuple[str, str], str] = {}
    sources: dict[tuple[str, str], list[str]] = defaultdict(list)
    for path in sorted(Path(state_bank_dir).glob("*.jsonl")):
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            state_id = str(row.get("state_id") or "")
            snapshot = str(row.get("axtree_snapshot") or "")
            if not state_id or not snapshot:
                continue
            hash_ = snapshot_sha256(snapshot)
            key = (state_id, hash_)
            previous = snapshots.get(key)
            if previous is not None and previous != snapshot:
                raise FormalCandidateMaterializationError(
                    f"colliding source snapshot for {state_id}/{hash_}")
            snapshots[key] = snapshot
            sources[key].append(f"{path}:{line_no}")
    return snapshots, dict(sources)


def _candidate_text(candidates: Iterable[Candidate]) -> str:
    return "".join(
        json.dumps(candidate.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for candidate in candidates)


def _write_immutable_asset(candidates: list[Candidate], body: Path,
                           manifest: Path) -> str:
    """Create body+manifest once; an exact rerun is idempotent."""
    body = Path(body)
    manifest = Path(manifest)
    if body.exists() != manifest.exists():
        raise FormalCandidateMaterializationError(
            "formal candidate body/manifest must both exist or both be absent")
    expected = _candidate_text(candidates)
    if body.exists():
        try:
            assert_candidate_manifest_integrity(body, manifest)
        except CandidateValidationError as exc:
            raise FormalCandidateMaterializationError(str(exc)) from exc
        if body.read_text(encoding="utf-8") != expected:
            raise FormalCandidateMaterializationError(
                f"refusing to overwrite versioned formal candidate asset {body}")
        return "already_identical"

    body.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(body, flags, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        write_candidate_manifest(body, manifest)
    except Exception:
        # Roll back only the new file created by this invocation.  Existing
        # research assets are never removed or rewritten.
        if body.exists() and not manifest.exists():
            body.unlink()
        raise
    return "created"


def _write_report(report: dict[str, Any], path: Path) -> None:
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FormalCandidateMaterializationError(
                f"refusing to overwrite versioned report {path}")
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def materialize_formal_candidates(
    data_root: Path,
    *,
    code_version: str,
    output_path: Path | None = None,
    manifest_path: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Deterministically build one 4--6 action set per canonical state view.

    Repeated live measurements may legitimately reference the same immutable
    state/candidate snapshot under different ``probe_point_id`` values.  They
    share one candidate set only when every candidate identity field agrees;
    conflicting repetitions remain a hard error.
    """
    if not str(code_version or "").strip():
        raise FormalCandidateMaterializationError("code_version is required")
    root = Path(data_root)
    points_path = root / "grounded" / "probe_points.jsonl"
    point_manifest = root / "grounded" / "POINT_MANIFEST.jsonl"
    try:
        assert_manifest_integrity(points_path, point_manifest)
        points = load_probe_points(points_path, validate=True)
    except (GroundingValidationError, OSError, ValueError,
            json.JSONDecodeError) as exc:
        raise FormalCandidateMaterializationError(
            f"canonical point artifact invalid: {exc}") from exc
    if not points:
        raise FormalCandidateMaterializationError(
            "cannot materialize formal candidates from zero points")

    by_state: dict[str, list[Any]] = defaultdict(list)
    for point in points.values():
        by_state[point.state_id].append(point)
    conflicting: list[str] = []
    repeated: dict[str, list[str]] = {}
    for state, rows in by_state.items():
        signatures = {
            (
                point.candidate_id,
                str((point.evidence or {}).get("candidate_snapshot_hash") or ""),
                point.raw_action,
                point.canonical_action,
            )
            for point in rows
        }
        if len(signatures) != 1:
            conflicting.append(state)
        elif len(rows) > 1:
            repeated[state] = sorted(point.probe_point_id for point in rows)
    if conflicting:
        raise FormalCandidateMaterializationError(
            f"{FORMAL_CANDIDATE_ARTIFACT_VERSION} found conflicting repeated "
            "measurements; states=" + ",".join(sorted(conflicting)))

    snapshots, snapshot_sources = _snapshot_index(root / "raw" / "state_bank")
    transitions = load_probe_transitions(
        root / "grounded" / "transitions" /
        "probe_transitions.v1.jsonl", validate=True)
    assert_point_transition_integrity(points, transitions, require_all=False)
    transitions_by_point = {
        transition.probe_point_id: transition
        for transition in transitions.values()
    }
    all_candidates: list[Candidate] = []
    point_experts: dict[str, str] = {}
    source_refs: dict[str, list[str]] = {}
    per_state_sizes: dict[str, int] = {}
    candidate_ids_by_state: dict[str, tuple[str, ...]] = {}
    exact_input_eligible: list[str] = []
    exact_input_excluded: list[dict[str, str]] = []
    for point in sorted(points.values(), key=lambda item: item.state_id):
        hash_ = str((point.evidence or {}).get("candidate_snapshot_hash") or "")
        key = (point.state_id, hash_)
        snapshot = snapshots.get(key)
        if snapshot is None:
            raise FormalCandidateMaterializationError(
                f"{point.probe_point_id}: exact source snapshot is missing")
        action = parse_action(point.raw_action)
        if action is None or action.name != "click" or not action.bid:
            raise FormalCandidateMaterializationError(
                f"{point.probe_point_id}: formal candidate v1 requires click(bid)")
        try:
            candidates = build_a11y_candidate_set(
                state_id=point.state_id,
                axtree_txt=snapshot,
                expert_bid=action.bid,
                proposer_version=FORMAL_SELECTOR_VERSION,
            )
            validate_candidate_set(candidates, snapshot)
        except CandidateValidationError as exc:
            raise FormalCandidateMaterializationError(
                f"{point.probe_point_id}: {exc}") from exc
        expert = [candidate for candidate in candidates
                  if candidate.category == CATEGORY_EXPERT]
        if len(expert) != 1 or expert[0].candidate_id != point.candidate_id:
            got = [candidate.candidate_id for candidate in expert]
            raise FormalCandidateMaterializationError(
                f"{point.probe_point_id}: expert candidate identity drift: {got}")
        if (expert[0].canonical_action != point.canonical_action or
                expert[0].snapshot_hash != hash_):
            raise FormalCandidateMaterializationError(
                f"{point.probe_point_id}: expert point/candidate mismatch")
        forbidden = [candidate.category for candidate in candidates
                     if candidate.category in _EVIDENCE_DEPENDENT_CATEGORIES]
        if forbidden:
            raise FormalCandidateMaterializationError(
                f"{point.probe_point_id}: static selector fabricated roles {forbidden}")
        generated_ids = tuple(candidate.candidate_id for candidate in candidates)
        previous_ids = candidate_ids_by_state.get(point.state_id)
        if previous_ids is None:
            all_candidates.extend(candidates)
            candidate_ids_by_state[point.state_id] = generated_ids
            per_state_sizes[point.state_id] = len(candidates)
        elif previous_ids != generated_ids:
            raise FormalCandidateMaterializationError(
                f"{point.probe_point_id}: repeated state candidate set drift")
        point_experts[point.probe_point_id] = expert[0].candidate_id
        source_refs[point.probe_point_id] = snapshot_sources[key]
        transition = transitions_by_point.get(point.probe_point_id)
        dynamic_transition = (
            isinstance(transition, ProbeTransition) and
            transition.pre_observation.policy_axtree_sha256 ==
            point.pre_observation_hash and
            transition.replay_verification == "dynamic_page_target_contract")
        if point.pre_observation_hash == hash_ or dynamic_transition:
            exact_input_eligible.append(point.probe_point_id)
        else:
            # This does not invalidate the grounding point or its candidate
            # set.  It does make the point ineligible for exact-input SFT: the
            # observation the probe acted on differs from the reviewed S4
            # snapshot which fixed the candidate identity.
            exact_input_excluded.append({
                "probe_point_id": point.probe_point_id,
                "state_id": point.state_id,
                "reason": "pre_observation_hash_differs_from_candidate_snapshot",
                "pre_observation_hash": point.pre_observation_hash,
                "candidate_snapshot_hash": hash_,
            })

    candidate_ids = [candidate.candidate_id for candidate in all_candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise FormalCandidateMaterializationError(
            "formal candidate IDs must be unique across point states")
    body_default, manifest_default = formal_candidate_paths(root)
    body = Path(output_path) if output_path else body_default
    manifest = Path(manifest_path) if manifest_path else (
        manifest_default if output_path is None else
        body.with_name(FORMAL_CANDIDATE_MANIFEST_NAME))

    legacy = legacy_candidate_path(root)
    legacy_hash_before = (hashlib.sha256(legacy.read_bytes()).hexdigest()
                          if legacy.exists() else None)
    status = _write_immutable_asset(all_candidates, body, manifest)
    legacy_hash_after = (hashlib.sha256(legacy.read_bytes()).hexdigest()
                         if legacy.exists() else None)
    if legacy_hash_before != legacy_hash_after:
        raise FormalCandidateMaterializationError(
            "legacy v3 candidate asset changed during formal materialization")
    try:
        indexed = assert_candidate_manifest_integrity(body, manifest)
    except CandidateValidationError as exc:
        raise FormalCandidateMaterializationError(str(exc)) from exc

    report = {
        "schema_version": FORMAL_CANDIDATE_REPORT_SCHEMA,
        "artifact_version": FORMAL_CANDIDATE_ARTIFACT_VERSION,
        "selector_version": FORMAL_SELECTOR_VERSION,
        "code_version": str(code_version),
        "status": status,
        "body": str(body),
        "manifest": str(manifest),
        "n_points": len(points),
        "n_states": len(per_state_sizes),
        "n_candidates": len(indexed),
        "repeated_measurement_states": repeated,
        "per_state_sizes": dict(sorted(Counter(per_state_sizes.values()).items())),
        "category": dict(sorted(Counter(
            candidate.category for candidate in indexed.values()).items())),
        "source": dict(sorted(Counter(
            candidate.source for candidate in indexed.values()).items())),
        "point_expert_candidates": dict(sorted(point_experts.items())),
        "snapshot_sources": dict(sorted(source_refs.items())),
        "state_scope_exact": set(per_state_sizes) == set(by_state),
        "expert_point_join_complete": len(point_experts) == len(points),
        "exact_input_sft": {
            "eligible_count": len(exact_input_eligible),
            "eligible_probe_point_ids": sorted(exact_input_eligible),
            "excluded_count": len(exact_input_excluded),
            "excluded": exact_input_excluded,
        },
        "evidence_dependent_role_count": sum(
            candidate.category in _EVIDENCE_DEPENDENT_CATEGORIES
            for candidate in indexed.values()),
        "legacy_v3": {
            "path": str(legacy),
            "exists": legacy.exists(),
            "sha256_before": legacy_hash_before,
            "sha256_after": legacy_hash_after,
            "unchanged": legacy_hash_before == legacy_hash_after,
            "formal_consumer": False,
        },
    }
    if report_path:
        _write_report(report, Path(report_path))
        report["report"] = str(report_path)
    return report
