#!/usr/bin/env python3
"""Deterministic post-remediation audit; never mutates source datasets."""
# ruff: noqa: E402 -- executable script adds the repository root before imports
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from revact import config, prompts
from revact.data.candidates import (FORMAL_CANDIDATE_ARTIFACT_VERSION,
                                    FORMAL_CANDIDATE_BODY_NAME,
                                    FORMAL_CANDIDATE_MANIFEST_NAME,
                                    LEGACY_CANDIDATE_BODY_NAME,
                                    CandidateValidationError,
                                    assert_candidate_manifest_integrity,
                                    snapshot_sha256)
from revact.data.episode_traces import audit_episode_trace_evidence
from revact.data.candidate_roles import (
    CANDIDATE_ROLES,
    CANDIDATE_ROLE_BODY_NAME,
    CANDIDATE_ROLE_MANIFEST_NAME,
    STATUS_EVIDENCED,
    CandidateRoleError,
    assert_candidate_role_manifest_integrity,
    discover_on_policy_role_sources,
    validate_candidate_role_evidence,
)
from revact.data.governance import (audit_collection_lineage,
                                    audit_formal_collection_lineage,
                                    formal_derivation_reasons,
                                    formal_release_context,
                                    formal_prompt_content_reasons,
                                    formal_release_reasons)
from revact.data.opinions import (OpinionLabelError,
                                  assert_opinion_manifest_integrity,
                                  load_opinion_records)
from revact.data.mutation_miner import audit_live_mutation_census
from revact.data.splits import (FORMAL_ISOLATION_AXES, audit_split_leakage,
                                load_formal_dpo_supplement, source_sample_id)
from revact.envs.obs_utils import bid_is_visible
from revact.eval.truth import (EvaluationTruthError,
                               assert_truth_manifest_integrity,
                               load_truth_records)
from revact.grounding.schema import (EFFECT_CHANGED, GroundingPoint,
                                     assert_manifest_integrity,
                                     load_probe_points)
from revact.grounding.backend_observers import (
    BACKEND_ATTESTATION_SCHEMA_VERSION, BACKEND_PROVIDER_SCHEMA_VERSION)
from revact.grounding.signal_evidence import (
    SIGNAL_EVIDENCE_REF_SCHEMA_VERSION, SIGNAL_EVIDENCE_SCHEMA_VERSION,
    SIGNAL_SNAPSHOT_REF_SCHEMA_VERSION, SIGNAL_SNAPSHOT_SCHEMA_VERSION)
from revact.grounding.transitions import (
    TRANSITION_BODY_RELATIVE, TRANSITION_MANIFEST_RELATIVE,
    TransitionValidationError, assert_point_transition_integrity,
    assert_transition_manifest_integrity, load_probe_transitions,
    transition_manifest_row)
from revact.train.distill import qc_full_sample
from revact.train.dpo import validate_rows as validate_dpo_rows
from revact.train.validators import answer_text, parse_action


OPINION_BODY_NAME = "opinion_labels.v2.jsonl"
OPINION_MANIFEST_NAME = "opinion_labels.v2.manifest.jsonl"
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_API_DB_CHANNELS = frozenset({"api", "backend_api", "db", "database"})
_SIGNAL_REF_FIELDS = {
    "schema_version", "probe_point_id", "channel", "path", "sha256"}
_SIGNAL_ASSET_FIELDS = {
    "schema_version", "evidence_id", "probe_point_id", "channel",
    "environment_instance", "collection_timestamp", "code_version",
    "observer_version", "collection_run_id", "endpoint_or_query_sha256",
    "provider", "collected_live", "is_fixture", "read_only_observer",
    "credential_value_stored", "pii_review_status", "redaction_applied",
    "payload_semantics", "redaction_key_persisted", "snapshots",
}
_SIGNAL_SNAPSHOT_FIELDS = {
    "schema_version", "probe_point_id", "channel", "phase",
    "environment_instance", "observed_at", "raw_payload",
    "normalized_state", "read_only_attestation",
}
_SIGNAL_SNAPSHOT_REF_FIELDS = {
    "schema_version", "phase", "path", "sha256",
    "normalized_state_sha256",
}
_BACKEND_PROVIDER_FIELDS = {
    "schema_version", "provider_id", "database_system", "transport",
    "query_id", "query_sha256", "source_instance_sha256",
    "read_only_enforcement", "projection_version", "redaction_strategy",
    "redaction_scope_sha256", "redaction_key_persisted",
    "container_image_ref", "container_id_sha256",
    "container_image_id_sha256",
}
_BACKEND_ATTESTATION_FIELDS = {
    "schema_version", "provider_id", "query_id", "transaction_read_only",
    "source_instance_sha256", "result_row_count",
}


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _last_user_observation(messages: list[dict]) -> str:
    user = next((message.get("content", "")
                 for message in reversed(messages)
                 if isinstance(message, dict) and message.get("role") == "user"), "")
    return prompts.parse_observation_message(str(user or ""))


def _visibility_summary(checked: list[str], missing: list[str],
                        invalid: list[str], non_bid: list[str]) -> dict:
    evaluated = bool(checked or invalid or non_bid)
    return {
        "n_actions": len(checked) + len(invalid) + len(non_bid),
        "n_bid_actions": len(checked),
        "n_visible": len(checked) - len(missing),
        "visibility_rate": ((len(checked) - len(missing)) / len(checked)
                            if checked else None),
        "missing_ids": missing,
        "invalid_action_ids": invalid,
        "non_bid_action_ids": non_bid,
        "checked_ids": checked,
        "evaluated": evaluated,
        "passed": (not missing and not invalid) if evaluated else None,
    }


def bid_visibility(rows: list[dict]) -> dict:
    """Audit both supervised answers and the pinned risky action.

    A safe answer may legitimately be a non-DOM action such as ``go_back`` or
    ``send_msg_to_user``.  The pinned risky action, however, must parse; when it
    is DOM-targeted its bid must be present in the exact user observation used
    for supervision.  This prevents a visible safe answer from hiding an
    invisible supervised risk-control target.
    """
    answer_checked, answer_missing, answer_invalid, answer_non_bid = [], [], [], []
    risky_checked, risky_missing, risky_invalid, risky_non_bid = [], [], [], []
    for row in rows:
        sid = str(row.get("sample_id") or row.get("pair_id") or "")
        messages = row.get("messages") or []
        last_user = ""
        for turn_index, message in enumerate(messages):
            if message.get("role") == "user":
                last_user = str(message.get("content") or "")
            if message.get("role") != "assistant":
                continue
            action = parse_action(answer_text(message.get("content") or ""))
            ident = f"{sid}:turn-{turn_index}"
            if not action:
                answer_invalid.append(ident)
                continue
            if not action.bid:
                answer_non_bid.append(ident)
                continue
            obs = prompts.parse_observation_message(last_user)
            ident += f":bid-{action.bid}"
            answer_checked.append(ident)
            if not bid_is_visible(obs, action.bid):
                answer_missing.append(ident)

        raw_risky = (row.get("meta") or {}).get("risky_raw_action")
        risky = parse_action(raw_risky)
        if not risky:
            risky_invalid.append(f"{sid}:risky_raw_action")
        elif not risky.bid:
            risky_non_bid.append(f"{sid}:risky_raw_action:{risky.name}")
        else:
            ident = f"{sid}:risky_raw_action:bid-{risky.bid}"
            risky_checked.append(ident)
            if not bid_is_visible(_last_user_observation(messages), risky.bid):
                risky_missing.append(ident)

    answer = _visibility_summary(
        answer_checked, answer_missing, answer_invalid, answer_non_bid)
    risky = _visibility_summary(
        risky_checked, risky_missing, risky_invalid, risky_non_bid)
    return {
        "assistant_answers": answer,
        "risky_raw_actions": risky,
        "evaluated": answer["evaluated"] and risky["evaluated"],
        "passed": answer["passed"] is True and risky["passed"] is True,
    }


def point_integrity(root: Path) -> dict:
    body = root / "grounded" / "probe_points.jsonl"
    manifest = root / "grounded" / "POINT_MANIFEST.jsonl"
    result = {
        "body_exists": body.exists(), "manifest_exists": manifest.exists(),
        "n_points": len(jsonl(body)), "n_manifest": len(jsonl(manifest)),
        "integrity": False, "error": "",
    }
    try:
        assert_manifest_integrity(body, manifest)
        points = load_probe_points(body, validate=True)
        result.update({
            "integrity": True,
            "missing_provenance": sum(bool(point.validation_errors(formal=True))
                                      for point in points.values()),
            "mock_points": sum(point.is_mock for point in points.values()),
            "not_recovered_points": sum(
                point.recovery_status == "NOT_RECOVERED_WITHIN_BUDGET"
                for point in points.values()),
        })
    except Exception as exc:  # audit should report all corruption, not hide it
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def transition_body_audit(root: Path, points: dict[str, GroundingPoint],
                          formal_rows: list[dict]) -> dict:
    """Verify exact observation bodies and every training-row transition join."""
    body = root / TRANSITION_BODY_RELATIVE
    manifest = root / TRANSITION_MANIFEST_RELATIVE
    result = {
        "body_path": str(body), "manifest_path": str(manifest),
        "body_exists": body.exists(), "manifest_exists": manifest.exists(),
        "n_transitions": len(jsonl(body)), "n_manifest": len(jsonl(manifest)),
        "integrity": False, "coverage": 0.0,
        "covered_point_count": 0,
        "missing_transition_point_ids": sorted(points),
        "formal_row_join_failures": {}, "formal_row_join_passed": False,
        "error": "",
    }
    if body.exists() != manifest.exists():
        result["error"] = (
            "transition body/manifest must either both exist or both be absent")
        return result
    if not body.exists():
        # Honest pre-data state: absence is not corrupted, but the separate
        # non-vacuous and row-join gates remain false.
        result["integrity"] = True
        return result
    try:
        assert_transition_manifest_integrity(body, manifest)
        transitions = load_probe_transitions(body, validate=True)
        coverage = assert_point_transition_integrity(
            points, transitions, require_all=False)
        result.update(coverage)
        refs_by_id = {
            transition.transition_id: transition
            for transition in transitions.values()
        }
        reference_failures: dict[str, list[str]] = {}
        for transition in transitions.values():
            point = points[transition.probe_point_id]
            ref = (point.evidence or {}).get("transition_ref") or {}
            expected = transition_manifest_row(transition)
            expected_ref = {
                "schema_version": transition.schema_version,
                "transition_id": transition.transition_id,
                "probe_point_id": transition.probe_point_id,
                "record_sha256": expected["record_sha256"],
            }
            if ref != expected_ref:
                reference_failures.setdefault(
                    transition.probe_point_id, []).append(
                        "point_transition_ref_mismatch")

        row_failures: dict[str, list[str]] = {}
        for index, row in enumerate(formal_rows):
            meta = row.get("meta") or {}
            ident = str(row.get("sample_id") or index)
            reasons: list[str] = []
            transition_id = str(meta.get("transition_id") or "")
            transition = refs_by_id.get(transition_id)
            if meta.get("prediction_source") != "probe_transition":
                reasons.append("prediction_source_not_probe_transition")
            if meta.get("transition_body_verified") is not True:
                reasons.append("transition_body_not_verified")
            if transition is None:
                reasons.append("transition_id_missing_or_unknown")
            else:
                expected = transition_manifest_row(transition)
                if meta.get("probe_point_id") != transition.probe_point_id:
                    reasons.append("transition_probe_point_mismatch")
                if meta.get("state_id") != transition.state_id:
                    reasons.append("transition_state_mismatch")
                if meta.get("candidate_id") != transition.candidate_id:
                    reasons.append("transition_candidate_mismatch")
                if meta.get("pre_observation_hash") != \
                        transition.pre_observation.policy_axtree_sha256:
                    reasons.append("transition_pre_hash_mismatch")
                if meta.get("post_observation_hash") != \
                        transition.post_observation.policy_axtree_sha256:
                    reasons.append("transition_post_hash_mismatch")
                if meta.get("transition_record_sha256") != \
                        expected["record_sha256"]:
                    reasons.append("transition_record_hash_mismatch")
            if reasons:
                row_failures[ident] = reasons
        result["point_reference_failures"] = reference_failures
        result["formal_row_join_failures"] = row_failures
        result["formal_row_join_passed"] = bool(formal_rows) and not row_failures
        result["integrity"] = not reference_failures
    except (TransitionValidationError, json.JSONDecodeError, OSError,
            TypeError, KeyError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def evaluation_truth_audit(root: Path, context) -> dict:
    """Audit the canonical normative truth body as a non-vacuous exact join.

    Recoverability rows and normative truth are intentionally separate assets.
    Readiness therefore requires both the truth body and its hash manifest, and
    every truth row must resolve to a canonical point rather than merely carry
    a syntactically valid ``probe_point_id`` string.
    """
    body = root / "eval" / "truth.jsonl"
    manifest = root / "eval" / "TRUTH_MANIFEST.jsonl"
    result = {
        "body_path": str(body),
        "manifest_path": str(manifest),
        "body_exists": body.exists(),
        "manifest_exists": manifest.exists(),
        "n_records": len(jsonl(body)),
        "n_manifest": len(jsonl(manifest)),
        "integrity": False,
        "unknown_probe_point_ids": [],
        "error": "",
    }
    if body.exists() != manifest.exists():
        result["error"] = (
            "evaluation truth body/manifest must either both exist or both be absent")
        return result
    if not body.exists():
        # Absence is an unmaterialized pre-data state, not corruption.  The
        # separate non-vacuous gate still prevents publication readiness.
        result["integrity"] = True
        return result
    try:
        assert_truth_manifest_integrity(body, manifest, context.points)
        records = load_truth_records(body, points=context.points)
        unknown = sorted({record.probe_point_id for record in records.values()
                          if record.probe_point_id not in context.points})
        result["unknown_probe_point_ids"] = unknown
        if context.grounding_error:
            raise EvaluationTruthError(
                "canonical grounding is invalid: " + context.grounding_error)
        if unknown:
            raise EvaluationTruthError(
                "unknown probe_point_id values: " + ",".join(unknown[:20]))
        result["integrity"] = True
    except (EvaluationTruthError, json.JSONDecodeError, OSError, TypeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def candidate_audit(root: Path) -> dict:
    directory = root / "raw" / "candidates"
    path = directory / FORMAL_CANDIDATE_BODY_NAME
    manifest = directory / FORMAL_CANDIDATE_MANIFEST_NAME
    legacy = directory / LEGACY_CANDIDATE_BODY_NAME
    try:
        indexed = assert_candidate_manifest_integrity(path, manifest)
    except (CandidateValidationError, json.JSONDecodeError, OSError, TypeError) as exc:
        return {
            "integrity": False, "error": f"{type(exc).__name__}: {exc}",
            "n_candidates": 0, "n_states": 0,
            "note": "candidate body/manifest failed before legality audit",
        }
    rows = [candidate.to_dict() for candidate in indexed.values()]
    by_state: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_state[str(row.get("state_id") or "")].append(row)
    sizes = Counter(len(items) for items in by_state.values())
    snapshots: dict[tuple[str, str], list[str]] = defaultdict(list)
    state_hashes: dict[str, set[str]] = defaultdict(set)
    for state_path in sorted((root / "raw" / "state_bank").glob("*.jsonl")):
        for state in jsonl(state_path):
            state_id = str(state.get("state_id") or "")
            axtree = str(state.get("axtree_snapshot") or "")
            if not state_id or not axtree:
                continue
            hash_ = snapshot_sha256(axtree)
            state_hashes[state_id].add(hash_)
            snapshots[(state_id, hash_)].append(axtree)
    missing_snapshot: list[str] = []
    invalid_snapshot: dict[str, str] = {}
    snapshot_legal = 0
    for candidate in indexed.values():
        matches = snapshots.get((candidate.state_id, candidate.snapshot_hash), [])
        if not matches:
            missing_snapshot.append(candidate.candidate_id)
            continue
        try:
            candidate.validate(matches[0], formal=True)
            snapshot_legal += 1
        except CandidateValidationError as exc:
            invalid_snapshot[candidate.candidate_id] = str(exc)
    return {
        "asset_tier": FORMAL_CANDIDATE_ARTIFACT_VERSION,
        "path": str(path), "manifest": str(manifest),
        "legacy_path": str(legacy), "legacy_formal_consumer": False,
        "integrity": (not missing_snapshot and not invalid_snapshot and
                      snapshot_legal == len(rows)),
        "error": "",
        "n_candidates": len(rows), "n_states": len(by_state),
        "per_state_sizes": dict(sorted(sizes.items())),
        "n_legal_declared": sum(row.get("legal_at_snapshot") is True for row in rows),
        "n_snapshot_matched_and_legal": snapshot_legal,
        "missing_snapshot_candidate_ids": missing_snapshot,
        "invalid_snapshot_candidates": invalid_snapshot,
        "ambiguous_state_ids": sorted(
            state_id for state_id, hashes in state_hashes.items()
            if len(hashes) > 1 and state_id in by_state),
        "all_candidate_ids_unique": len({row.get("candidate_id") for row in rows}) == len(rows),
        "category": dict(Counter(row.get("category") for row in rows)),
        "source": dict(Counter(row.get("source") for row in rows)),
        "note": "categories are proposal hypotheses; no candidate is grounded by this audit",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_points(root: Path) -> tuple[dict[str, GroundingPoint], str]:
    body = root / "grounded" / "probe_points.jsonl"
    manifest = root / "grounded" / "POINT_MANIFEST.jsonl"
    try:
        assert_manifest_integrity(body, manifest)
        return load_probe_points(body, validate=True), ""
    except Exception as exc:  # readiness reports corruption instead of aborting
        return {}, f"{type(exc).__name__}: {exc}"


def _canonical_truths(root: Path, points: dict[str, GroundingPoint]) -> tuple[dict, str]:
    body = root / "eval" / "truth.jsonl"
    manifest = root / "eval" / "TRUTH_MANIFEST.jsonl"
    try:
        assert_truth_manifest_integrity(body, manifest, points)
        return load_truth_records(body, points=points), ""
    except Exception as exc:  # readiness reports corruption instead of aborting
        return {}, f"{type(exc).__name__}: {exc}"


def _candidate_role_coverage_summary(
    records: dict,
    points: dict[str, GroundingPoint],
    replay: dict,
) -> tuple[dict, dict]:
    """Describe successful role evidence without treating rows as independent.

    ``validate_candidate_role_evidence`` is the authority on whether a role row
    can be replayed.  Failed rows and proposal-only rows are deliberately absent
    from both coverage and overlap counts.  The case overlap is important: one
    executed candidate can satisfy multiple role protocols for the same
    evaluation case, so summing per-role counts is not a sample-size estimate.
    """
    failed_ids = set((replay.get("failures") or {}).keys())
    by_role: dict[str, list[tuple[object, GroundingPoint]]] = defaultdict(list)
    by_case: dict[str, list[tuple[object, GroundingPoint]]] = defaultdict(list)
    for record in records.values():
        record_id = str(getattr(record, "candidate_role_id", "") or "")
        if getattr(record, "status", None) != STATUS_EVIDENCED or \
                record_id in failed_ids:
            continue
        point = points.get(str(getattr(record, "probe_point_id", "") or ""))
        if point is None:
            # A replay-valid record must have a point.  Stay fail-closed if a
            # future validator ever violates that invariant.
            continue
        role = str(getattr(record, "role", "") or "")
        evaluation_case_id = str(
            getattr(record, "evaluation_case_id", "") or "")
        by_role[role].append((record, point))
        by_case[evaluation_case_id].append((record, point))

    coverage: dict[str, dict] = {}
    for role, joined in sorted(by_role.items()):
        coverage[role] = {
            "n_records": len(joined),
            "n_unique_evaluation_cases": len({
                str(getattr(record, "evaluation_case_id"))
                for record, _point in joined}),
            "unique_evaluation_cases": sorted({
                str(getattr(record, "evaluation_case_id"))
                for record, _point in joined}),
            "n_unique_states": len({
                str(getattr(record, "state_id"))
                for record, _point in joined}),
            "unique_states": sorted({
                str(getattr(record, "state_id"))
                for record, _point in joined}),
            "n_unique_sites": len({point.site for _record, point in joined}),
            "unique_sites": sorted({point.site for _record, point in joined}),
            "n_unique_action_types": len({
                point.action_type for _record, point in joined}),
            "unique_action_types": sorted({
                point.action_type for _record, point in joined}),
        }

    overlaps: list[dict] = []
    for evaluation_case_id, joined in sorted(by_case.items()):
        roles = sorted({str(getattr(record, "role"))
                        for record, _point in joined})
        if len(roles) < 2:
            continue
        overlaps.append({
            "evaluation_case_id": evaluation_case_id,
            "n_records": len(joined),
            "n_unique_roles": len(roles),
            "roles": roles,
            "candidate_role_ids": sorted({
                str(getattr(record, "candidate_role_id"))
                for record, _point in joined}),
            "probe_point_ids": sorted({
                str(getattr(record, "probe_point_id"))
                for record, _point in joined}),
            "state_ids": sorted({str(getattr(record, "state_id"))
                                  for record, _point in joined}),
            "sites": sorted({point.site for _record, point in joined}),
            "action_types": sorted({
                point.action_type for _record, point in joined}),
        })
    overlap = {
        "n_evidenced_cases": len(by_case),
        "n_overlapping_cases": len(overlaps),
        "overlapping_case_ids": [
            item["evaluation_case_id"] for item in overlaps],
        "cases": overlaps,
    }
    return coverage, overlap


def candidate_role_evidence_audit(
    root: Path,
    points: dict[str, GroundingPoint],
    truths: dict,
    *,
    grounding_error: str = "",
    truth_error: str = "",
) -> dict:
    """Replay case-scoped roles; proposal categories never count as evidence."""
    directory = root / "raw" / "candidates"
    body = directory / CANDIDATE_ROLE_BODY_NAME
    manifest = directory / CANDIDATE_ROLE_MANIFEST_NAME
    result = {
        "body_path": str(body),
        "manifest_path": str(manifest),
        "body_exists": body.exists(),
        "manifest_exists": manifest.exists(),
        "integrity": False,
        "n_records": 0,
        "evidenced_role": {},
        "proposed_role": {},
        "evidenced_role_coverage": {},
        "evidenced_case_role_overlap": {
            "n_evidenced_cases": 0,
            "n_overlapping_cases": 0,
            "overlapping_case_ids": [],
            "cases": [],
        },
        "coverage_interpretation": {
            "gate_contract": (
                "at least one replay-valid EVIDENCED record per required role"),
            "minimum_records_per_required_role_for_gate": 1,
            "applies_n_ge_30_effect_evidence_threshold": False,
            "supports_n_ge_30_effect_claims": False,
            "note": (
                "This is a taxonomy-presence gate, not n>=30 evidence for any "
                "role, cell, effect estimate, or generalization claim."),
        },
        "failures": {},
        "error": grounding_error or truth_error,
    }
    if grounding_error or truth_error:
        return result
    try:
        candidate_body = directory / FORMAL_CANDIDATE_BODY_NAME
        candidate_manifest = directory / FORMAL_CANDIDATE_MANIFEST_NAME
        candidates = assert_candidate_manifest_integrity(
            candidate_body, candidate_manifest)
        records = assert_candidate_role_manifest_integrity(body, manifest)
        rejected_on_policy_sources = {}
        on_policy_traces = discover_on_policy_role_sources(
            root, truths, rejected=rejected_on_policy_sources)
        sft_rows = jsonl(
            root / "train" / "formal" / config.FORMAL_SFT_PATH.name)
        replay = validate_candidate_role_evidence(
            records.values(), candidates=candidates, points=points,
            truths=truths, sft_rows=sft_rows,
            on_policy_traces=on_policy_traces)
        result.update(replay)
        coverage, overlap = _candidate_role_coverage_summary(
            records, points, replay)
        result["evidenced_role_coverage"] = coverage
        result["evidenced_case_role_overlap"] = overlap
        result["on_policy_trace_sources"] = len(on_policy_traces)
        result["rejected_on_policy_trace_sources"] = \
            rejected_on_policy_sources
        result["error"] = "" if replay["integrity"] else \
            "one or more role/source joins failed replay"
    except (CandidateRoleError, CandidateValidationError, OSError,
            ValueError, TypeError, json.JSONDecodeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def opinion_evidence_audit(
    root: Path,
    points: dict[str, GroundingPoint],
    truths: dict,
    *,
    grounding_error: str = "",
    truth_error: str = "",
) -> dict:
    """Verify the one canonical opinion release against every formal point.

    A random JSONL line is not an opinion baseline.  Readiness requires the
    versioned opinion schema, its 1:1 hash manifest, an exact state/point join,
    full truth-case coverage, and both a human and an independent LLM rater for
    every case.  Two model seeds cannot masquerade as the promised human audit.
    """
    body = root / "opinions" / OPINION_BODY_NAME
    manifest = root / "opinions" / OPINION_MANIFEST_NAME
    result = {
        "body_path": str(body),
        "manifest_path": str(manifest),
        "body_exists": body.exists(),
        "manifest_exists": manifest.exists(),
        "integrity": False,
        "n_records": 0,
        "n_points": len(points),
        "n_truth_cases": len(truths),
        "covered_cases": 0,
        "coverage": None,
        "min_distinct_raters_per_case": 2,
        "rater_counts_by_case": {},
        "human_rater_counts_by_case": {},
        "llm_rater_counts_by_case": {},
        "missing_evaluation_case_ids": sorted(truths),
        "under_rated_evaluation_case_ids": sorted(truths),
        "missing_human_evaluation_case_ids": sorted(truths),
        "missing_llm_evaluation_case_ids": sorted(truths),
        "error": grounding_error or truth_error,
        "passed": False,
    }
    if grounding_error or truth_error:
        return result
    if body.exists() != manifest.exists():
        result["error"] = "opinion body/manifest must both exist or both be absent"
        return result
    if not body.exists():
        result["error"] = "canonical opinion body/manifest are absent"
        return result
    if not points:
        result["error"] = "canonical grounding contains zero formal points"
        return result
    if not truths:
        result["error"] = "canonical evaluation truth contains zero cases"
        return result
    try:
        assert_opinion_manifest_integrity(
            body, manifest, points=points, truths=truths)
        records = load_opinion_records(body, points=points, truths=truths)
        raters: dict[str, set[str]] = defaultdict(set)
        human_raters: dict[str, set[str]] = defaultdict(set)
        llm_raters: dict[str, set[str]] = defaultdict(set)
        for record in records.values():
            case_id = record.evaluation_case_id
            raters[case_id].add(record.rater_id)
            if record.rater_type == "HUMAN":
                human_raters[case_id].add(record.rater_id)
            elif record.rater_type == "LLM":
                llm_raters[case_id].add(record.rater_id)
        counts = {case_id: len(raters.get(case_id, set()))
                  for case_id in sorted(truths)}
        human_counts = {case_id: len(human_raters.get(case_id, set()))
                        for case_id in sorted(truths)}
        llm_counts = {case_id: len(llm_raters.get(case_id, set()))
                      for case_id in sorted(truths)}
        missing = [case_id for case_id, count in counts.items() if count == 0]
        under_rated = [case_id for case_id, count in counts.items() if count < 2]
        missing_human = [case_id for case_id, count in human_counts.items()
                         if count == 0]
        missing_llm = [case_id for case_id, count in llm_counts.items()
                       if count == 0]
        covered = len(truths) - len(missing)
        result.update({
            "integrity": True,
            "n_records": len(records),
            "covered_cases": covered,
            "coverage": covered / len(truths),
            "rater_counts_by_case": counts,
            "human_rater_counts_by_case": human_counts,
            "llm_rater_counts_by_case": llm_counts,
            "missing_evaluation_case_ids": missing,
            "under_rated_evaluation_case_ids": under_rated,
            "missing_human_evaluation_case_ids": missing_human,
            "missing_llm_evaluation_case_ids": missing_llm,
            "error": "",
            "passed": (bool(records) and not under_rated and
                       not missing_human and not missing_llm),
        })
    except (OpinionLabelError, json.JSONDecodeError, OSError, TypeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _technology_family(point: GroundingPoint) -> str:
    """Return application technology, not the enclosing benchmark name."""
    site = str(point.site or "").strip().lower().replace("-", "_")
    aliases = {
        "shopping": "magento",
        "shopping_admin": "magento",
        "reddit": "postmill",
        "gitlab": "gitlab",
        "wikipedia": "mediawiki",
        "map": "openstreetmap",
        "servicenow": "servicenow",
        "workarena": "servicenow",
    }
    return aliases.get(site, str(point.environment_family or "").strip().lower())


def environment_evidence_audit(points: list[GroundingPoint]) -> dict:
    """Audit independent families while retaining the concrete instance axis."""
    family_instances: dict[str, set[str]] = defaultdict(set)
    missing: list[str] = []
    instance_families: dict[str, set[str]] = defaultdict(set)
    for point in points:
        family = _technology_family(point)
        instance = str(point.environment_instance or "").strip()
        if not family or not instance:
            missing.append(point.probe_point_id)
            continue
        family_instances[family].add(instance)
        instance_families[instance].add(family)
    ambiguous = sorted(
        instance for instance, families in instance_families.items()
        if len(families) > 1)
    families = sorted(family_instances)
    pairs = sorted(
        {f"{family}::{instance}" for family, instances in family_instances.items()
         for instance in instances})
    return {
        "technology_families": families,
        "family_instances": {
            family: sorted(instances)
            for family, instances in sorted(family_instances.items())},
        "family_instance_pairs": pairs,
        "n_independent_families": len(families),
        "n_family_instance_pairs": len(pairs),
        "missing_identity_point_ids": sorted(missing),
        "ambiguous_instances": ambiguous,
        "passed": len(families) >= 3 and not missing and not ambiguous,
        "note": "shopping and shopping_admin are one Magento family",
    }


def _safe_asset_path(root: Path, relative: object) -> Path | None:
    if not isinstance(relative, str) or not relative.strip():
        return None
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _canonical_state_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_signal_snapshot(
    root: Path,
    point: GroundingPoint,
    channel: str,
    reference: dict,
    provider: dict,
) -> tuple[str, object, list[str]]:
    """Load one content-addressed raw observer snapshot without trusting its ref."""
    problems: list[str] = []
    phase = str(reference.get("phase") or "")
    if set(reference) != _SIGNAL_SNAPSHOT_REF_FIELDS:
        problems.append(f"{phase or '<missing>'} snapshot ref fields differ")
    if reference.get("schema_version") != SIGNAL_SNAPSHOT_REF_SCHEMA_VERSION:
        problems.append(f"{phase or '<missing>'} bad snapshot ref schema")
    expected_hash = str(reference.get("sha256") or "")
    if not _HEX_64.fullmatch(expected_hash):
        problems.append(f"{phase or '<missing>'} bad snapshot sha256")
    path = _safe_asset_path(root, reference.get("path"))
    if path is None or not path.is_file():
        problems.append(f"{phase or '<missing>'} snapshot is absent/outside root")
        return phase, None, problems
    if expected_hash and _sha256(path) != expected_hash:
        problems.append(f"{phase or '<missing>'} snapshot hash mismatch")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        problems.append(f"{phase or '<missing>'} unreadable signal snapshot: {exc}")
        return phase, None, problems
    if not isinstance(raw, dict):
        problems.append(f"{phase or '<missing>'} snapshot must be an object")
        return phase, None, problems
    if set(raw) != _SIGNAL_SNAPSHOT_FIELDS:
        problems.append(f"{phase or '<missing>'} snapshot fields differ")
    if raw.get("schema_version") != SIGNAL_SNAPSHOT_SCHEMA_VERSION:
        problems.append(f"{phase or '<missing>'} bad snapshot schema")
    for field in (
            "observed_at", "raw_payload", "normalized_state",
            "read_only_attestation"):
        if field not in raw:
            problems.append(f"{phase or '<missing>'} snapshot missing {field}")
    if raw.get("probe_point_id") != point.probe_point_id:
        problems.append(f"{phase or '<missing>'} snapshot point mismatch")
    if str(raw.get("channel") or "").lower() != channel:
        problems.append(f"{phase or '<missing>'} snapshot channel mismatch")
    if raw.get("phase") != phase:
        problems.append(f"{phase or '<missing>'} snapshot phase mismatch")
    if raw.get("environment_instance") != point.environment_instance:
        problems.append(f"{phase or '<missing>'} snapshot environment mismatch")
    normalized = raw.get("normalized_state")
    if reference.get("normalized_state_sha256") != _canonical_state_sha256(normalized):
        problems.append(f"{phase or '<missing>'} normalized state hash mismatch")
    attestation = raw.get("read_only_attestation")
    if not isinstance(attestation, dict) or set(attestation) != \
            _BACKEND_ATTESTATION_FIELDS:
        problems.append(f"{phase or '<missing>'} bad read-only attestation fields")
    else:
        if attestation.get("schema_version") != \
                BACKEND_ATTESTATION_SCHEMA_VERSION:
            problems.append(
                f"{phase or '<missing>'} bad read-only attestation schema")
        if attestation.get("transaction_read_only") is not True:
            problems.append(
                f"{phase or '<missing>'} database transaction was not read-only")
        for field in ("provider_id", "query_id", "source_instance_sha256"):
            if attestation.get(field) != provider.get(field):
                problems.append(
                    f"{phase or '<missing>'} attestation {field} mismatch")
        if not isinstance(attestation.get("result_row_count"), int) or \
                attestation.get("result_row_count", -1) < 0:
            problems.append(
                f"{phase or '<missing>'} bad attested result_row_count")
    payload = raw.get("raw_payload")
    if not isinstance(payload, dict) or set(payload) != {
            "payload_semantics", "query_id", "transaction_read_only",
            "row_count", "rows"}:
        problems.append(f"{phase or '<missing>'} raw payload fields differ")
    else:
        if payload.get("payload_semantics") != \
                "minimized_redacted_projection":
            problems.append(
                f"{phase or '<missing>'} raw payload is not minimized/redacted")
        if payload.get("query_id") != provider.get("query_id"):
            problems.append(f"{phase or '<missing>'} raw payload query mismatch")
        if payload.get("transaction_read_only") is not True:
            problems.append(
                f"{phase or '<missing>'} raw payload is not read-only")
        rows = payload.get("rows")
        count = payload.get("row_count")
        if not isinstance(rows, list) or not isinstance(count, int) or \
                len(rows) != count:
            problems.append(f"{phase or '<missing>'} raw payload row count mismatch")
        elif isinstance(attestation, dict) and \
                attestation.get("result_row_count") != count:
            problems.append(
                f"{phase or '<missing>'} attested row count mismatch")
        else:
            for index, row in enumerate(rows):
                if not isinstance(row, dict) or set(row) != {"row_token", "state"}:
                    problems.append(
                        f"{phase or '<missing>'} row {index} is not minimized")
                    continue
                if not _HEX_64.fullmatch(str(row.get("row_token") or "")):
                    problems.append(
                        f"{phase or '<missing>'} row {index} token is invalid")
    return phase, normalized, problems


def signal_evidence_audit(root: Path, points: list[GroundingPoint]) -> dict:
    """Require independent API/DB evidence, not a repackaged UI signal.

    Every admitted channel resolves to one immutable evidence object and three
    distinct content-addressed raw observations (pre, post, final). The point
    itself must carry the same normalized states, so a detached JSON file
    cannot upgrade an existing UI-only point after the fact.
    """
    declared_points: list[str] = []
    verified_assets: list[str] = []
    errors: dict[str, list[str]] = {}
    channel_counts: Counter[str] = Counter()
    for point in points:
        evidence = point.evidence if isinstance(point.evidence, dict) else {}
        channels = {
            str(channel).strip().lower()
            for channel in (evidence.get("signal_channels") or [])}
        required = sorted(channels & _API_DB_CHANNELS)
        if not required:
            continue
        declared_points.append(point.probe_point_id)
        refs = evidence.get("signal_evidence_assets") or []
        refs = refs if isinstance(refs, list) else []
        by_channel: dict[str, list[dict]] = defaultdict(list)
        for ref in refs:
            if isinstance(ref, dict):
                by_channel[str(ref.get("channel") or "").lower()].append(ref)
        for channel in required:
            ident = f"{point.probe_point_id}:{channel}"
            problems: list[str] = []
            matching = by_channel.get(channel, [])
            if len(matching) != 1:
                problems.append("requires exactly one signal evidence reference")
            if not matching:
                errors[ident] = problems
                continue
            ref = matching[0]
            if set(ref) != _SIGNAL_REF_FIELDS:
                problems.append("signal evidence reference fields differ")
            if ref.get("schema_version") != SIGNAL_EVIDENCE_REF_SCHEMA_VERSION:
                problems.append("bad signal evidence reference schema")
            if ref.get("probe_point_id") != point.probe_point_id:
                problems.append("reference probe_point_id mismatch")
            expected_hash = str(ref.get("sha256") or "")
            if not _HEX_64.fullmatch(expected_hash):
                problems.append("reference sha256 is not 64 lowercase hex")
            path = _safe_asset_path(root, ref.get("path"))
            if path is None or not path.is_file():
                problems.append("referenced asset is absent or outside data root")
            asset: dict = {}
            if path is not None and path.is_file():
                if expected_hash and _sha256(path) != expected_hash:
                    problems.append("referenced asset hash mismatch")
                try:
                    raw_asset = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(raw_asset, dict):
                        asset = raw_asset
                    else:
                        problems.append("signal evidence asset must be one JSON object")
                except (json.JSONDecodeError, OSError) as exc:
                    problems.append(f"unreadable signal evidence asset: {exc}")
            if asset:
                if set(asset) != _SIGNAL_ASSET_FIELDS:
                    problems.append("signal evidence asset fields differ")
                if asset.get("schema_version") != SIGNAL_EVIDENCE_SCHEMA_VERSION:
                    problems.append("bad signal evidence asset schema")
                for field in (
                    "evidence_id", "collection_timestamp", "code_version",
                    "observer_version", "collection_run_id",
                    "environment_instance",
                ):
                    if not str(asset.get(field) or "").strip():
                        problems.append(f"signal evidence asset missing {field}")
                if asset.get("probe_point_id") != point.probe_point_id:
                    problems.append("asset probe_point_id mismatch")
                if str(asset.get("channel") or "").lower() != channel:
                    problems.append("asset channel mismatch")
                if asset.get("environment_instance") != point.environment_instance:
                    problems.append("asset environment_instance mismatch")
                if asset.get("collected_live") is not True:
                    problems.append("signal evidence was not collected live")
                if asset.get("is_fixture") is not False:
                    problems.append("signal evidence is fixture/unspecified")
                if asset.get("read_only_observer") is not True:
                    problems.append("API/DB observer must be read-only")
                if asset.get("credential_value_stored") is not False:
                    problems.append("credential_value_stored must be false")
                if asset.get("pii_review_status") != "REDACTED_AND_REVIEWED":
                    problems.append(
                        "DB evidence must be redacted and PII-reviewed")
                if asset.get("redaction_applied") is not True:
                    problems.append("DB evidence must apply identifier redaction")
                if asset.get("payload_semantics") != \
                        "minimized_redacted_projection":
                    problems.append("DB payload is not minimized/redacted")
                if asset.get("redaction_key_persisted") is not False:
                    problems.append("redaction key must not be persisted")
                query_fp = str(asset.get("endpoint_or_query_sha256") or "")
                if not _HEX_64.fullmatch(query_fp):
                    problems.append(
                        "endpoint_or_query_sha256 is not 64 lowercase hex")
                provider = asset.get("provider")
                if not isinstance(provider, dict) or set(provider) != \
                        _BACKEND_PROVIDER_FIELDS:
                    provider = {}
                    problems.append("bad backend provider metadata fields")
                else:
                    if provider.get("schema_version") != \
                            BACKEND_PROVIDER_SCHEMA_VERSION:
                        problems.append("bad backend provider schema")
                    for field in (
                            "query_sha256", "source_instance_sha256",
                            "redaction_scope_sha256", "container_id_sha256",
                            "container_image_id_sha256"):
                        if not _HEX_64.fullmatch(str(provider.get(field) or "")):
                            problems.append(f"bad backend provider {field}")
                    if provider.get("database_system") not in {
                            "mariadb", "postgresql"}:
                        problems.append("bad backend provider database_system")
                    if provider.get("transport") != "docker_exec":
                        problems.append("bad backend provider transport")
                    if provider.get("redaction_strategy") != \
                            "ephemeral-hmac-sha256":
                        problems.append(
                            "bad backend provider redaction strategy")
                    if provider.get("redaction_key_persisted") is not False:
                        problems.append(
                            "backend provider persisted redaction key")
                snapshots = asset.get("snapshots")
                if not isinstance(snapshots, list):
                    snapshots = []
                    problems.append("signal evidence snapshots must be a list")
                phase_refs: dict[str, dict] = {}
                normalized: dict[str, object] = {}
                snapshot_paths: set[str] = set()
                for snapshot_ref in snapshots:
                    if not isinstance(snapshot_ref, dict):
                        problems.append(
                            "signal snapshot reference must be an object")
                        continue
                    phase = str(snapshot_ref.get("phase") or "")
                    if phase in phase_refs:
                        problems.append(
                            f"duplicate signal snapshot phase {phase!r}")
                        continue
                    phase_refs[phase] = snapshot_ref
                    path_text = str(snapshot_ref.get("path") or "")
                    if path_text in snapshot_paths:
                        problems.append(
                            "signal phases must use distinct snapshot paths")
                    snapshot_paths.add(path_text)
                    loaded_phase, state, snapshot_problems = _load_signal_snapshot(
                        root, point, channel, snapshot_ref, provider)
                    problems.extend(snapshot_problems)
                    if loaded_phase:
                        normalized[loaded_phase] = state
                if set(phase_refs) != {"pre", "post", "final"}:
                    problems.append(
                        "signal evidence requires exactly pre/post/final phases")
                summaries = evidence.get("api_db_signal_summaries")
                summaries = summaries if isinstance(summaries, dict) else {}
                channel_summary = summaries.get(channel)
                if not isinstance(channel_summary, dict):
                    problems.append(
                        "point evidence lacks API/DB normalized summary")
                else:
                    for phase in ("pre", "post", "final"):
                        if normalized.get(phase) != channel_summary.get(phase):
                            problems.append(
                                f"point {phase} normalized summary mismatch")
                pre_state = normalized.get("pre")
                post_state = normalized.get("post")
                final_state = normalized.get("final")
                if point.effect_status == EFFECT_CHANGED and pre_state == post_state:
                    problems.append(
                        "CHANGED point has identical API/DB pre/post state")
                if point.effect_status == "NO_EFFECT" and pre_state != post_state:
                    problems.append(
                        "NO_EFFECT point has differing API/DB pre/post state")
                if point.recovery_status == "RECOVERED" and final_state != pre_state:
                    problems.append(
                        "RECOVERED point API/DB final state differs from pre")
            if problems:
                errors[ident] = list(dict.fromkeys(problems))
            else:
                verified_assets.append(ident)
                channel_counts[channel] += 1
    return {
        "declared_point_ids": sorted(declared_points),
        "verified_assets": sorted(verified_assets),
        "verified_channel_counts": dict(sorted(channel_counts.items())),
        "errors": errors,
        "passed": bool(verified_assets) and not errors,
    }


def mutation_evidence_audit(root: Path) -> dict:
    """Verify the canonical schema/manifest; never trust hand-filled metrics."""
    return audit_live_mutation_census(root)


def research_evidence_audit(root: Path, formal_rows: list[dict],
                            candidates: dict) -> dict:
    """Non-vacuous gates for claims that artifact integrity alone cannot prove.

    These checks intentionally make the distinction between "the code accepts
    this schema" and "the current corpus contains evidence for the paper
    claim" machine-readable.  A fixture report never satisfies a live-data
    gate, and an empty split/history cell never passes vacuously.
    """
    # Keep direct formal-input lengths as a diagnostic only.  They do not prove
    # that the lines came from one real episode, nor that an intermediate
    # state-changing completion was actually supervised.  Publication gates
    # below consume the immutable stateless episode trace asset instead.
    formal_input_history_lengths: list[int] = []
    for row in formal_rows:
        messages = row.get("messages") or []
        user = next((str(message.get("content") or "")
                     for message in reversed(messages)
                     if isinstance(message, dict) and
                     message.get("role") == "user"), "")
        history = prompts.parse_user(user).get("history", "")
        formal_input_history_lengths.append(len([
            line for line in str(history).splitlines() if line.strip()]))
    formal_input_history_buckets = {
        "1_3": sum(1 <= length <= 3 for length in formal_input_history_lengths),
        "4_6": sum(4 <= length <= 6 for length in formal_input_history_lengths),
        "ge_7": sum(length >= 7 for length in formal_input_history_lengths),
    }

    point_index, grounding_error = _canonical_points(root)
    points = list(point_index.values())
    truth_index, truth_error = _canonical_truths(root, point_index)
    sites = sorted({point.site for point in points})
    signal_channels = Counter(
        channel for point in points
        for channel in ((point.evidence or {}).get("signal_channels") or []))
    negative_points = [
        point for point in points
        if point.recovery_status == "NOT_RECOVERED_WITHIN_BUDGET"]

    opinions = opinion_evidence_audit(
        root, point_index, truth_index,
        grounding_error=grounding_error, truth_error=truth_error)
    candidate_roles = candidate_role_evidence_audit(
        root, point_index, truth_index,
        grounding_error=grounding_error, truth_error=truth_error)
    episode_traces = audit_episode_trace_evidence(
        root, point_index, formal_rows)
    history_buckets = episode_traces["history_buckets"]
    intermediate = episode_traces["intermediate_state_changing"]
    environments = environment_evidence_audit(points)
    mutation = mutation_evidence_audit(root)
    signal_evidence = signal_evidence_audit(root, points)

    evidenced_role = candidate_roles.get("evidenced_role") or {}
    missing_roles = sorted(
        role for role in CANDIDATE_ROLES
        if int(evidenced_role.get(role) or 0) == 0)
    gates = {
        "independent_environment_families_ge_3": environments["passed"],
        "history_buckets_nonempty": (
            episode_traces["integrity"] and
            episode_traces["configuration_supports_ge_7"] and
            not episode_traces["configuration_mismatches"] and
            all(history_buckets.values())),
        "state_changing_intermediate_supervision": intermediate["passed"],
        "formal_negative_solver_union_evidence": len(negative_points) > 0,
        "candidate_taxonomy_coverage": (
            candidate_roles.get("integrity") is True and not missing_roles),
        "paired_opinion_labels": opinions["passed"],
        "live_mutation_census_ge_200": mutation["passed"],
        "api_or_db_signal_evidence": signal_evidence["passed"],
    }
    return {
        "gates": gates,
        "sites": sites,
        "environment_instances": sorted(
            {point.environment_instance for point in points}),
        "environment_coverage": environments,
        "history_length": {
            "evidence_source": "stateless_episode_trace_exact_sft_join",
            "n": episode_traces["n_supervised_verified_steps"],
            "min": (min(episode_traces["history_lengths"])
                    if episode_traces["history_lengths"] else None),
            "max": (max(episode_traces["history_lengths"])
                    if episode_traces["history_lengths"] else None),
            "buckets": history_buckets,
            "state_changing_turns": intermediate["verified_count"],
            "formal_input_diagnostic": {
                "n": len(formal_input_history_lengths),
                "min": (min(formal_input_history_lengths)
                        if formal_input_history_lengths else None),
                "max": (max(formal_input_history_lengths)
                        if formal_input_history_lengths else None),
                "buckets": formal_input_history_buckets,
                "counts_as_episode_evidence": False,
            },
        },
        "stateless_episode_traces": episode_traces,
        "intermediate_state_changing": intermediate,
        "formal_negative_points": len(negative_points),
        "signal_channels": dict(sorted(signal_channels.items())),
        "signal_evidence": signal_evidence,
        # Kept for report-reader compatibility, but values are now roles whose
        # EVIDENCED count is zero, never candidate.category strings.
        "candidate_missing_categories": missing_roles,
        "candidate_role_missing_evidence": missing_roles,
        "candidate_role_evidence": candidate_roles,
        "opinion_rows": opinions["n_records"],
        "opinion_paths": [opinions["body_path"], opinions["manifest_path"]],
        "opinion_evidence": opinions,
        "mutation_live_report": mutation["body_path"],
        "mutation_evidence": mutation,
        "mutation_live_census_passed": mutation["passed"],
        "note": ("research-evidence gates are non-vacuous corpus/live checks; "
                 "unit tests and fixture reports cannot satisfy them"),
    }


def _release_failures(rows: list[dict], context) -> dict[str, list[str]]:
    failures = {}
    for index, row in enumerate(rows):
        key = str(row.get("sample_id") or row.get("pair_id") or index)
        reasons = formal_release_reasons(row.get("meta") or {}, context)
        reasons.extend(formal_prompt_content_reasons(row, context))
        # Preserve order while removing duplicate codes from overlapping gates.
        reasons = list(dict.fromkeys(reasons))
        if reasons:
            failures[key] = reasons
    return failures


def _prompt_content_failures(rows: list[dict], context) -> dict[str, list[str]]:
    failures = {}
    for index, row in enumerate(rows):
        reasons = formal_prompt_content_reasons(row, context)
        if reasons:
            key = str(row.get("sample_id") or row.get("pair_id") or index)
            failures[key] = reasons
    return failures


def deployment_source_share(rows: list[dict], minimum: float = .50) -> dict:
    """Return the release-wide share of deployment-shaped DPO negatives."""
    sources = Counter(
        str((row.get("meta") or {}).get("negative_source") or "missing")
        for row in rows)
    deployment_sources = {"legal_candidate", "on_policy"}
    deployment = sum(sources[source] for source in deployment_sources)
    share = deployment / len(rows) if rows else None
    return {
        "n_pairs": len(rows),
        "source_counts": dict(sorted(sources.items())),
        "legal_or_on_policy_n": deployment,
        "legal_or_on_policy_share": share,
        "minimum_required_share": minimum,
        "passed": bool(rows) and share is not None and share >= minimum,
    }


def formal_dpo_audit(root: Path, formal_rows: list[dict], train: list[dict],
                     dev: list[dict], context) -> dict:
    """Audit formal DPO bodies, exact SFT derivation, and split membership.

    A non-empty DPO file is not sufficient: the pair must be an exact derived
    view of a formal point-grounded SFT row, its split must follow that source
    row, and at least half of both the authored body and the train split must be
    legal-candidate or on-policy negatives.
    """
    base = root / "train" / "formal"
    body_paths = [
        base / config.FORMAL_DPO_PATH.name,
        base / config.FORMAL_MULTITURN_DPO_PATH.name,
    ]
    supplement_paths = sorted(base.glob(config.FORMAL_DPO_SUPPLEMENT_GLOB))
    body_paths.extend(supplement_paths)
    train_paths = [
        base / "splits" / "dpo_train.jsonl",
        base / "splits" / "dpo_train_multiturn.jsonl",
    ]
    dev_paths = [
        base / "splits" / "dpo_dev.jsonl",
        base / "splits" / "dpo_dev_multiturn.jsonl",
    ]
    body_rows = [row for path in body_paths[:2] for row in jsonl(path)]
    supplement_releases: list[dict] = []
    supplement_failures: dict[str, str] = {}
    for path in supplement_paths:
        try:
            rows, manifest = load_formal_dpo_supplement(path)
            body_rows.extend(rows)
            supplement_releases.append({
                "path": str(path),
                "release_id": manifest["release_id"],
                "body_sha256": manifest["body_sha256"],
                "n_rows": manifest["n_rows"],
            })
        except (ValueError, json.JSONDecodeError, OSError) as exc:
            supplement_failures[str(path)] = f"{type(exc).__name__}: {exc}"
    train_rows = [row for path in train_paths for row in jsonl(path)]
    dev_rows = [row for path in dev_paths for row in jsonl(path)]

    issues: dict[str, list[str]] = defaultdict(list)

    def flag(kind: str, ident: str) -> None:
        if ident not in issues[kind]:
            issues[kind].append(ident)

    for path, error in supplement_failures.items():
        flag("supplement_manifest_integrity", f"{path}:{error}")

    def index_unique(rows: list[dict], key: str, scope: str) -> dict[str, dict]:
        indexed: dict[str, dict] = {}
        counts = Counter(str(row.get(key) or "") for row in rows)
        for ident, count in counts.items():
            if not ident:
                flag(f"{scope}_missing_{key}", "<missing>")
            elif count > 1:
                flag(f"{scope}_duplicate_{key}", f"{ident}:count={count}")
        for row in rows:
            ident = str(row.get(key) or "")
            if ident and counts[ident] == 1:
                indexed[ident] = row
        return indexed

    sources = index_unique(formal_rows, "sample_id", "source_sft")
    body = index_unique(body_rows, "pair_id", "body")
    split_train = index_unique(train_rows, "pair_id", "train_split")
    split_dev = index_unique(dev_rows, "pair_id", "dev_split")

    if body_rows:
        validation = validate_dpo_rows(
            body_rows, formal=True, min_deployment_negative_ratio=0.0)
        for problem in validation["problems"]:
            flag("body_schema_or_policy", problem)
    else:
        validation = {
            "n_rows": 0, "n_problems": 0, "problems": [],
            "negative_source_dist": {}, "deployment_negative_ratio": None,
        }

    release_failures = _release_failures(body_rows, context)
    for pair_id, reasons in release_failures.items():
        for reason in reasons:
            flag("body_release_gate", f"{pair_id}:{reason}")

    for pair_id, pair in body.items():
        source_id = source_sample_id(pair)
        source = sources.get(source_id)
        if source is None:
            flag("unknown_source_sample_id", f"{pair_id}:{source_id or '<missing>'}")
            continue
        source_messages = source.get("messages") or []
        if pair.get("prompt") != source_messages[:-1]:
            flag("derived_input_messages_mismatch", pair_id)
        expected_chosen = (
            source_messages[-1].get("content")
            if source_messages and source_messages[-1].get("role") == "assistant"
            else None)
        if pair.get("chosen") != expected_chosen:
            flag("derived_chosen_mismatch", pair_id)
        for reason in formal_derivation_reasons(
                source.get("meta") or {}, pair.get("meta") or {}):
            flag("derived_provenance_mismatch", f"{pair_id}:{reason}")

    source_split_ids = {
        "train": {source_sample_id(row) for row in train},
        "dev": {source_sample_id(row) for row in dev},
    }
    actual_split = {"train": split_train, "dev": split_dev}
    for side, indexed in actual_split.items():
        for pair_id, pair in indexed.items():
            authored = body.get(pair_id)
            if authored is None:
                flag("split_pair_absent_from_body", f"{side}:{pair_id}")
                continue
            if pair != authored:
                flag("split_pair_content_mismatch", f"{side}:{pair_id}")
            source_id = source_sample_id(pair)
            if source_id not in source_split_ids[side]:
                flag("split_source_membership_mismatch",
                     f"{side}:{pair_id}:{source_id or '<missing>'}")

    overlapping = sorted(set(split_train) & set(split_dev))
    for pair_id in overlapping:
        flag("train_dev_pair_overlap", pair_id)
    for side in ("train", "dev"):
        expected = {
            pair_id for pair_id, pair in body.items()
            if source_sample_id(pair) in source_split_ids[side]
        }
        actual = set(actual_split[side])
        for pair_id in sorted(expected - actual):
            flag("split_missing_expected_pair", f"{side}:{pair_id}")
        for pair_id in sorted(actual - expected):
            flag("split_unexpected_pair", f"{side}:{pair_id}")

    normalized_issues = {
        key: sorted(values) for key, values in sorted(issues.items())}
    body_share = deployment_source_share(body_rows)
    train_share = deployment_source_share(train_rows)
    dev_share = deployment_source_share(dev_rows)
    structural_issue_count = sum(
        len(values) for values in normalized_issues.values())
    return {
        "body_paths": [str(path) for path in body_paths],
        "supplement_releases": supplement_releases,
        "supplement_failures": supplement_failures,
        "train_split_paths": [str(path) for path in train_paths],
        "dev_split_paths": [str(path) for path in dev_paths],
        "n_body_pairs": len(body_rows),
        "n_train_pairs": len(train_rows),
        "n_dev_pairs": len(dev_rows),
        "body_validation": validation,
        "body_source_share": body_share,
        "train_source_share": train_share,
        "dev_source_share": dev_share,
        "issues": normalized_issues,
        "n_structural_issues": structural_issue_count,
        # An empty pre-data state is not structural corruption.  Non-vacuous
        # gates below still make it publication-blocking.
        "integrity": structural_issue_count == 0,
        "split_integrity": structural_issue_count == 0,
        "source_share_passed": (
            body_share["passed"] and train_share["passed"]),
    }


def teacher_audit(root: Path, source_families: dict[str, list[dict]],
                  context) -> dict:
    """Audit independent single/multiturn teacher artifacts as one corpus."""
    base = root / "train" / "formal"
    paths = {
        "single": base / config.FORMAL_DISTILLED_SFT_PATH.name,
        "multiturn": (
            base / config.FORMAL_MULTITURN_DISTILLED_SFT_PATH.name),
    }
    family_reports = {}
    all_source_ids: dict[str, str] = {}
    cross_family_source_overlap = []
    all_duplicates = []
    all_orphans = []
    all_teacher_failures = {}
    all_release_failures = {}
    total_source = total_rows = total_matched = total_qualified = 0

    for family, path in paths.items():
        sources = list(source_families.get(family) or [])
        rows = jsonl(path)
        source_id_list = [str(row.get("sample_id") or "") for row in sources]
        source_counts = Counter(source_id_list)
        source_duplicate_ids = sorted(
            sample_id for sample_id, count in source_counts.items()
            if sample_id and count > 1)
        source_ids = {sample_id for sample_id in source_id_list if sample_id}
        for sample_id in source_ids:
            other = all_source_ids.get(sample_id)
            if other and other != family:
                cross_family_source_overlap.append(sample_id)
            all_source_ids[sample_id] = family

        distilled_ids = [str(row.get("sample_id") or "") for row in rows]
        duplicate_ids = sorted(
            sample_id for sample_id, count in Counter(distilled_ids).items()
            if sample_id and count > 1)
        orphan_ids = sorted({sample_id for sample_id in distilled_ids
                             if sample_id not in source_ids})
        matched = {sample_id for sample_id in distilled_ids
                   if sample_id in source_ids}
        release_failures = _release_failures(rows, context)
        teacher_failures = {}
        qualified = set()
        for index, row in enumerate(rows):
            sample_id = str(row.get("sample_id") or index)
            meta = row.get("meta") or {}
            reasons = []
            if meta.get("prose_source") != "teacher":
                reasons.append("teacher_prose_source_not_teacher")
            if meta.get("rev_check_source") != "teacher":
                reasons.append("teacher_rev_check_source_not_teacher")
            if meta.get("teacher_qc_status") != "passed":
                reasons.append("teacher_qc_status_not_passed")
            qc_reason = qc_full_sample(row)
            if qc_reason:
                reasons.append(f"teacher_qc:{qc_reason}")
            if reasons:
                teacher_failures[sample_id] = list(dict.fromkeys(reasons))
            elif sample_id in source_ids:
                qualified.add(sample_id)

        source_n = len(source_ids)
        coverage = len(qualified) / source_n if source_n else None
        artifact_coverage = len(matched) / source_n if source_n else None
        active = bool(sources)
        passed = ((not rows) if not active else (
            coverage is not None and coverage >= .95 and
            not source_duplicate_ids and not duplicate_ids and
            not orphan_ids and not release_failures and not teacher_failures))
        family_reports[family] = {
            "path": str(path), "active": active,
            "n_source_rows": len(sources),
            "n_unique_source_rows": source_n,
            "n_rows": len(rows),
            "n_matched_source_rows": len(matched),
            "n_qualified_teacher_rows": len(qualified),
            "artifact_coverage": artifact_coverage,
            "coverage": coverage,
            "source_duplicate_sample_ids": source_duplicate_ids,
            "duplicate_sample_ids": duplicate_ids,
            "orphan_sample_ids": orphan_ids,
            "teacher_source_failures": teacher_failures,
            "release_or_prompt_failures": release_failures,
            "passed": passed,
        }
        total_source += source_n
        total_rows += len(rows)
        total_matched += len(matched)
        total_qualified += len(qualified)
        all_duplicates.extend(f"{family}:{item}" for item in duplicate_ids)
        all_duplicates.extend(
            f"{family}:source:{item}" for item in source_duplicate_ids)
        all_orphans.extend(f"{family}:{item}" for item in orphan_ids)
        all_teacher_failures.update({
            f"{family}:{key}": value for key, value in teacher_failures.items()})
        all_release_failures.update({
            f"{family}:{key}": value for key, value in release_failures.items()})

    coverage = total_qualified / total_source if total_source else None
    artifact_coverage = total_matched / total_source if total_source else None
    cross_family_source_overlap = sorted(set(cross_family_source_overlap))
    return {
        # Backward-compatible single path plus the authoritative family map.
        "path": str(paths["single"]),
        "paths": {family: str(path) for family, path in paths.items()},
        "families": family_reports,
        "n_rows": total_rows,
        "n_unique_source_rows": total_source,
        "n_matched_source_rows": total_matched,
        "n_qualified_teacher_rows": total_qualified,
        "artifact_coverage": artifact_coverage,
        "coverage": coverage,
        "duplicate_sample_ids": sorted(all_duplicates),
        "orphan_sample_ids": sorted(all_orphans),
        "cross_family_source_overlap": cross_family_source_overlap,
        "teacher_source_failures": all_teacher_failures,
        "n_teacher_source_failures": len(all_teacher_failures),
        "n_release_or_prompt_failures": len(all_release_failures),
        "release_or_prompt_failures": all_release_failures,
        "coverage_target": 0.95,
        "passed": bool(total_source) and coverage is not None and coverage >= .95
                  and all(report["passed"] for report in family_reports.values())
                  and not cross_family_source_overlap,
    }


def _split_has_no_overlap(leakage: dict) -> bool:
    return all(axis in leakage and leakage[axis].get("n_overlap") == 0
               for axis in FORMAL_ISOLATION_AXES)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-blocked", action="store_true",
        help="write/print a blocked report but return exit code 0; never changes ready")
    args = parser.parse_args()
    root = args.data_root
    single = jsonl(root / "train" / "formal" / config.FORMAL_SFT_PATH.name)
    multi = jsonl(
        root / "train" / "formal" / config.FORMAL_MULTITURN_SFT_PATH.name)
    formal_rows = single + multi
    train = jsonl(root / "train" / "formal" / "splits" / "sft_train.jsonl")
    train += jsonl(root / "train" / "formal" / "splits" /
                   "sft_train_multiturn.jsonl")
    dev = jsonl(root / "train" / "formal" / "splits" / "sft_dev.jsonl")
    dev += jsonl(root / "train" / "formal" / "splits" /
                 "sft_dev_multiturn.jsonl")
    test = jsonl(root / "train" / "formal" / "splits" / "sft_test.jsonl")
    test += jsonl(root / "train" / "formal" / "splits" /
                  "sft_test_multiturn.jsonl")
    context = formal_release_context(root)
    release_failures = _release_failures(formal_rows, context)
    prompt_content_failures = _prompt_content_failures(formal_rows, context)
    visibility = bid_visibility(formal_rows)
    grounding = point_integrity(root)
    transitions = transition_body_audit(root, context.points, formal_rows)
    evaluation_truth = evaluation_truth_audit(root, context)
    leakage = {
        "train_dev": audit_split_leakage(train, dev),
        "train_test": audit_split_leakage(train, test),
        "dev_test": audit_split_leakage(dev, test),
    }
    lineage = audit_formal_collection_lineage(root, formal_rows)
    legacy_lineage = audit_collection_lineage(root)
    candidates = candidate_audit(root)
    research_evidence = research_evidence_audit(
        root, formal_rows, candidates)
    teacher = teacher_audit(
        root, {"single": single, "multiturn": multi}, context)
    dpo = formal_dpo_audit(root, formal_rows, train, dev, context)
    quarantine_report_path = root / "train" / "quarantine" / "migration-report.json"
    quarantine = (json.loads(quarantine_report_path.read_text(encoding="utf-8"))
                  if quarantine_report_path.exists() else {})
    non_vacuous_gates = {
        "grounding_points": grounding.get("n_points", 0) > 0,
        "probe_transition_bodies": transitions.get("n_transitions", 0) > 0,
        "evaluation_truth_records": evaluation_truth.get("n_records", 0) > 0,
        "formal_sft": len(formal_rows) > 0,
        "formal_train_split": len(train) > 0,
        "formal_dev_split": len(dev) > 0,
        "formal_test_split": len(test) > 0,
        "formal_dpo": dpo["n_body_pairs"] > 0,
        "formal_dpo_train_split": dpo["n_train_pairs"] > 0,
    }
    integrity_gates = {
        "grounding_manifest_integrity": grounding.get("integrity") is True,
        "grounding_provenance_complete": grounding.get("missing_provenance", 0) == 0,
        "grounding_has_no_mock": grounding.get("mock_points", 0) == 0,
        "transition_body_manifest_integrity": transitions.get("integrity") is True,
        "formal_transition_body_join": (
            transitions.get("formal_row_join_passed") is True),
        "evaluation_truth_manifest_integrity": (
            evaluation_truth.get("integrity") is True),
        "candidate_manifest_and_snapshot_integrity": candidates.get("integrity") is True,
        "formal_release_and_prompt_content": not release_failures,
        "formal_has_no_mock": not any(
            (row.get("meta") or {}).get("is_mock") is True for row in formal_rows),
        "formal_collectors_succeeded": not any(
            (row.get("meta") or {}).get("collector_success") is not True
            for row in formal_rows),
        "answer_and_risky_bid_visibility": visibility["passed"],
        "split_overlap_zero": all(
            _split_has_no_overlap(pair) for pair in leakage.values()),
        "formal_collection_lineage_integrity": lineage.get("ok") is True,
        "formal_dpo_integrity": dpo["integrity"],
        "formal_dpo_split_integrity": dpo["split_integrity"],
        "formal_dpo_source_share": dpo["source_share_passed"],
        "teacher_coverage_and_integrity": teacher["passed"],
    }
    non_vacuous = all(non_vacuous_gates.values())
    research_gates = research_evidence["gates"]
    ready = (non_vacuous and all(integrity_gates.values()) and
             all(research_gates.values()))
    blocked = [name for name, passed in {
        **non_vacuous_gates, **integrity_gates}.items() if not passed]
    blocked.extend(
        f"research:{name}" for name, passed in research_gates.items()
        if not passed)
    report = {
        "schema_version": "iris.formal_readiness_audit.v5",
        "ready": ready,
        "non_vacuous": non_vacuous,
        "non_vacuous_gates": non_vacuous_gates,
        "integrity_gates": integrity_gates,
        "grounding": grounding,
        "probe_transitions": transitions,
        "formal_training": {
            "n_single_sft": len(single), "n_multiturn_sft": len(multi),
            "n_total_sft": len(formal_rows),
            "n_release_gate_failures": len(release_failures),
            "release_gate_failures": release_failures,
            "n_prompt_content_failures": len(prompt_content_failures),
            "prompt_content_failures": prompt_content_failures,
            "n_mock": sum((row.get("meta") or {}).get("is_mock") is True
                          for row in formal_rows),
            "n_failed_collector": sum(
                (row.get("meta") or {}).get("collector_success") is not True
                for row in formal_rows),
            "bid_visibility": visibility,
        },
        "evaluation_truth": evaluation_truth,
        "formal_dpo": dpo,
        "teacher": teacher,
        "formal_split_leakage": leakage,
        "collection_lineage": lineage,
        "legacy_collection_lineage": legacy_lineage,
        "candidates": candidates,
        "research_evidence": research_evidence,
        "quarantine": {key: quarantine.get(key) for key in (
            "n_indexed", "n_formal_eligible", "n_quarantined",
            "reason_counts", "migration_mode", "rollback")},
        "blocked": blocked,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if ready or args.allow_blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())
