#!/usr/bin/env python3
"""Deterministic post-remediation audit; never mutates source datasets."""
# ruff: noqa: E402 -- executable script adds the repository root before imports
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from revact import config, prompts
from revact.data.candidates import (CandidateValidationError,
                                    assert_candidate_manifest_integrity,
                                    snapshot_sha256)
from revact.data.governance import (audit_collection_lineage,
                                    audit_formal_collection_lineage,
                                    formal_derivation_reasons,
                                    formal_release_context,
                                    formal_prompt_content_reasons,
                                    formal_release_reasons)
from revact.data.splits import (FORMAL_ISOLATION_AXES, audit_split_leakage,
                                source_sample_id)
from revact.envs.obs_utils import bid_is_visible
from revact.eval.truth import (EvaluationTruthError,
                               assert_truth_manifest_integrity,
                               load_truth_records)
from revact.grounding.schema import assert_manifest_integrity, load_probe_points
from revact.train.dpo import validate_rows as validate_dpo_rows
from revact.train.validators import answer_text, parse_action


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
    path = root / "raw" / "candidates" / "iris_candidates.v3.jsonl"
    try:
        indexed = assert_candidate_manifest_integrity(path)
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
        base / "iris_dpo_point_v1.jsonl",
        base / "iris_dpo_multiturn_point_v1.jsonl",
    ]
    train_paths = [
        base / "splits" / "dpo_train.jsonl",
        base / "splits" / "dpo_train_multiturn.jsonl",
    ]
    dev_paths = [
        base / "splits" / "dpo_dev.jsonl",
        base / "splits" / "dpo_dev_multiturn.jsonl",
    ]
    body_rows = [row for path in body_paths for row in jsonl(path)]
    train_rows = [row for path in train_paths for row in jsonl(path)]
    dev_rows = [row for path in dev_paths for row in jsonl(path)]

    issues: dict[str, list[str]] = defaultdict(list)

    def flag(kind: str, ident: str) -> None:
        if ident not in issues[kind]:
            issues[kind].append(ident)

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


def teacher_audit(root: Path, source_rows: list[dict], context) -> dict:
    """Measure coverage from the one canonical formal distilled artifact."""
    path = (root / "train" / "formal" /
            config.FORMAL_DISTILLED_SFT_PATH.name)
    rows = jsonl(path)
    source_ids = {str(row.get("sample_id") or "") for row in source_rows}
    distilled_ids = [str(row.get("sample_id") or "") for row in rows]
    matched = {sample_id for sample_id in distilled_ids if sample_id in source_ids}
    duplicate_ids = sorted(sample_id for sample_id, count
                           in Counter(distilled_ids).items()
                           if sample_id and count > 1)
    orphan_ids = sorted({sample_id for sample_id in distilled_ids
                         if sample_id not in source_ids})
    failures = _release_failures(rows, context)
    coverage = len(matched) / len(source_ids) if source_ids else None
    return {
        "path": str(path),
        "n_rows": len(rows),
        "n_unique_source_rows": len(source_ids),
        "n_matched_source_rows": len(matched),
        "coverage": coverage,
        "duplicate_sample_ids": duplicate_ids,
        "orphan_sample_ids": orphan_ids,
        "n_release_or_prompt_failures": len(failures),
        "release_or_prompt_failures": failures,
        "coverage_target": 0.95,
        "passed": bool(source_ids) and coverage is not None and coverage >= .95
                  and not duplicate_ids and not orphan_ids and not failures,
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
    single = jsonl(root / "train" / "formal" / "iris_sft_point_v1.jsonl")
    multi = jsonl(root / "train" / "formal" / "iris_sft_multiturn_point_v1.jsonl")
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
    evaluation_truth = evaluation_truth_audit(root, context)
    leakage = {
        "train_dev": audit_split_leakage(train, dev),
        "train_test": audit_split_leakage(train, test),
        "dev_test": audit_split_leakage(dev, test),
    }
    lineage = audit_formal_collection_lineage(root, formal_rows)
    legacy_lineage = audit_collection_lineage(root)
    candidates = candidate_audit(root)
    teacher = teacher_audit(root, formal_rows, context)
    dpo = formal_dpo_audit(root, formal_rows, train, dev, context)
    quarantine_report_path = root / "train" / "quarantine" / "migration-report.json"
    quarantine = (json.loads(quarantine_report_path.read_text(encoding="utf-8"))
                  if quarantine_report_path.exists() else {})
    non_vacuous_gates = {
        "grounding_points": grounding.get("n_points", 0) > 0,
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
    ready = non_vacuous and all(integrity_gates.values())
    blocked = [name for name, passed in {
        **non_vacuous_gates, **integrity_gates}.items() if not passed]
    report = {
        "schema_version": "iris.formal_readiness_audit.v3",
        "ready": ready,
        "non_vacuous": non_vacuous,
        "non_vacuous_gates": non_vacuous_gates,
        "integrity_gates": integrity_gates,
        "grounding": grounding,
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
