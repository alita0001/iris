"""Regression tests for the non-vacuous publication-readiness audit."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from revact import config
from revact.data.candidates import (CANDIDATE_SCHEMA_VERSION,
                                    FORMAL_CANDIDATE_BODY_NAME,
                                    FORMAL_CANDIDATE_MANIFEST_NAME, Candidate,
                                    save_candidate_set, snapshot_sha256)
from revact.data.opinions import (NOT_RISKY, OPINION_ARTIFACT_ROLE,
                                  OPINION_LABEL_SCHEMA_VERSION,
                                  PERCEIVED_CHANGE, PERCEIVED_RECOVERABLE,
                                  RATER_HUMAN, RATER_LLM, OpinionLabelRecord,
                                  make_opinion_label_id,
                                  save_opinion_records)
from revact.data.mutation_miner import (
    MUTATION_REPORT_BODY_NAME, MUTATION_REPORT_MANIFEST_NAME,
    MUTATION_SAMPLE_SCHEMA_VERSION, MutationCensusSample,
    make_mutation_sample_id, mutation_control_set_sha256,
    save_live_mutation_census,
)
from revact.data.splits import write_formal_dpo_supplement_manifest
from revact.eval.truth import (EVALUATION_TRUTH_SCHEMA_VERSION,
                               EvaluationTruthRecord)
from revact.grounding.schema import (EFFECT_CHANGED, RECOVERY_RECOVERED,
                                     GroundingPoint)
from revact.prompt_store import store_bundle


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_formal_readiness.py"
SPEC = importlib.util.spec_from_file_location("audit_formal_readiness", SCRIPT)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _run(root: Path, *extra: str) -> tuple[subprocess.CompletedProcess, dict]:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--data-root", str(root), *extra],
        cwd=ROOT, text=True, capture_output=True, check=False)
    return result, json.loads(result.stdout)


def test_empty_formal_assets_fail_non_vacuous_gate(tmp_path):
    result, report = _run(tmp_path)
    assert result.returncode == 1
    assert report["ready"] is False
    assert report["non_vacuous"] is False
    assert report["non_vacuous_gates"] == {
        "evaluation_truth_records": False,
        "formal_dpo": False,
        "formal_dpo_train_split": False,
        "formal_dev_split": False,
        "formal_sft": False,
        "formal_test_split": False,
        "formal_train_split": False,
        "grounding_points": False,
        "probe_transition_bodies": False,
    }
    assert report["grounding"]["n_points"] == 0
    assert report["teacher"]["coverage"] is None

    allowed, same_report = _run(tmp_path, "--allow-blocked")
    assert allowed.returncode == 0
    assert same_report["ready"] is False
    assert same_report["non_vacuous"] is False


def test_research_history_gate_does_not_trust_unlinked_prompt_lines(tmp_path):
    history = "\n".join(
        f"{index}. click('{index}') -> [nav] page {index}"
        for index in range(1, 8))
    row = {
        "sample_id": "metadata-only-long-history",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": (
                f"<goal>\ng\n\n<history>\n{history}\n\n"
                "<observation>\n[42] button 'Do'")},
            {"role": "assistant", "content": _full_iris()},
        ],
        "meta": {"assistant_turn_types": ["state_changing"]},
    }
    report = AUDIT.research_evidence_audit(
        tmp_path, [row], {"category": {}})
    assert report["history_length"]["formal_input_diagnostic"]["buckets"][
        "ge_7"] == 1
    assert report["history_length"]["buckets"]["ge_7"] == 0
    assert report["gates"]["history_buckets_nonempty"] is False
    assert report["gates"]["state_changing_intermediate_supervision"] is False
    assert report["stateless_episode_traces"]["integrity"] is False


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _point(point_id: str = "point-1", *, site: str = "shopping",
           instance: str = "shopping:7770", **updates) -> GroundingPoint:
    point = GroundingPoint(
        probe_point_id=point_id,
        probe_run_id="probe-run-1",
        probe_name=f"{site}.fixture",
        state_id=f"state-{point_id}",
        candidate_id=f"candidate-{point_id}",
        action_instance_id=f"action-{point_id}",
        action_type="add_to_cart",
        raw_action="click('42')",
        canonical_action="click:add_to_cart:sku-1",
        site=site,
        environment_family="webarena",
        environment_instance=instance,
        environment_origin="webarena",
        is_mock=False,
        task_id="webarena.1",
        trajectory_id=f"traj-{point_id}",
        run_id="run-1",
        seed=0,
        url="http://example.test/item",
        account="user-1",
        privilege="customer",
        budget_k=12,
        solver_set=["site_specific_deterministic"],
        controller_version="test-controller",
        pre_observation_hash="pre-hash",
        pre_signal={"count": 0},
        post_observation_hash="post-hash",
        post_signal={"count": 1},
        undo_actions=["click('43')"],
        undo_semantic_actions=["remove_item"],
        undo_observation_hashes=["undo-hash"],
        final_signal={"count": 0},
        effect_status=EFFECT_CHANGED,
        recovery_status=RECOVERY_RECOVERED,
        undo_cost_steps=1,
        residual_diff={},
        budget_exhausted=False,
        timestamp="2026-07-14T00:00:00Z",
        code_version="deadbeef",
        evidence={
            "candidate_snapshot_hash": "candidate-snapshot-hash",
            "signal_channels": ["ui_structural"],
        },
    )
    return replace(point, **updates)


def _full_iris(action: str = "click('42')") -> str:
    return (
        "<think>\n"
        "<observation>target visible\n"
        "<reasoning>goal requires the target\n"
        "<prediction>the measured state changes\n"
        "<rev_check>probe evidence identifies an undo route\n"
        "<reversibility>RECOVERED\n"
        "<undo>remove the created item\n"
        "<decision>EXECUTE risk=0.1\n"
        "</think>\n"
        f"<answer> {action}"
    )


def _truth(point: GroundingPoint, case_id: str = "case-1",
           variant: str = "request") -> EvaluationTruthRecord:
    return EvaluationTruthRecord(
        schema_version=EVALUATION_TRUTH_SCHEMA_VERSION,
        evaluation_case_id=case_id,
        probe_point_id=point.probe_point_id,
        state_id=point.state_id,
        variant=variant,
        effect_status=point.effect_status,
        recovery_status=point.recovery_status,
        normative_risk=False,
        policy_constraint_truth=False,
        action_required_for_goal=True,
        violates_constraint=False,
        expected_decision="EXECUTE",
        actual_action=None,
        action_legal=None,
        risky_attempt=None,
        backend_commit=None,
        guarded=None,
        policy_id="fixture-policy",
        policy_version="v1",
        truth_source="fixture",
        author="test",
        timestamp="2026-07-14T00:00:00Z",
        code_version="deadbeef",
        evidence={"source": "fixture"},
    )


def _opinion(point: GroundingPoint, truth: EvaluationTruthRecord,
             rater_id: str, rater_type: str) -> OpinionLabelRecord:
    instrument_id, instrument_version = "fixture-instrument", "v1"
    goal_sha256 = hashlib.sha256(b"fixture goal").hexdigest()
    raw = json.dumps({
        "effect": PERCEIVED_CHANGE,
        "recovery": PERCEIVED_RECOVERABLE,
        "normative_risk": NOT_RISKY,
        "confidence": .8,
        "rationale": "independent fixture rating",
    }, sort_keys=True)
    return OpinionLabelRecord(
        schema_version=OPINION_LABEL_SCHEMA_VERSION,
        artifact_role=OPINION_ARTIFACT_ROLE,
        opinion_label_id=make_opinion_label_id(
            point.probe_point_id, truth.evaluation_case_id, goal_sha256,
            rater_id, instrument_id, instrument_version),
        probe_point_id=point.probe_point_id,
        state_id=point.state_id,
        evaluation_case_id=truth.evaluation_case_id,
        variant=truth.variant,
        goal_sha256=goal_sha256,
        opinion_input_sha256=hashlib.sha256(b"input").hexdigest(),
        input_messages_sha256=hashlib.sha256(b"messages").hexdigest(),
        raw_response=raw,
        raw_response_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        provider_response_id=f"response-{rater_id}",
        response_model=("human-instrument" if rater_type == RATER_HUMAN
                        else "provider/model"),
        finish_reason="completed",
        rater_id=rater_id,
        rater_type=rater_type,
        provider=("direct-study" if rater_type == RATER_HUMAN else "openrouter"),
        model=None if rater_type == RATER_HUMAN else "provider/model",
        prompt_generation_fp="prompt-fp-v1",
        instrument_id=instrument_id,
        instrument_version=instrument_version,
        perceived_effect=PERCEIVED_CHANGE,
        perceived_recoverability=PERCEIVED_RECOVERABLE,
        normative_risk_opinion=NOT_RISKY,
        confidence=.8,
        rationale="independent fixture rating",
        source_record_id=f"source-{rater_id}",
        collection_timestamp="2026-07-14T00:00:00Z",
        import_batch_id="batch-1",
        code_version="deadbeef",
    )


def _formal_collection_fixture(root: Path, *, origin: str = "webarena") -> dict:
    run_id, trajectory_id, state_id = "run-1", "traj-1", "state-1"
    _write_jsonl(root / "raw" / "trajectories_meta.jsonl", [
        # Frozen legacy dirt remains visible to the historical diagnostic but
        # must not permanently block a clean, exact formal run closure.
        {"trajectory_id": "legacy-duplicate"},
        {"trajectory_id": "legacy-duplicate"},
        {
            "run_id": run_id, "trajectory_id": trajectory_id,
            "environment_origin": origin, "success": True,
            "collector_success": True,
        },
    ])
    _write_jsonl(root / "raw" / "trajectories" / f"{trajectory_id}.jsonl", [
        {"run_id": run_id, "trajectory_id": trajectory_id, "step_id": 0},
    ])
    _write_jsonl(root / "raw" / "state_bank" / "shopping_key_states.jsonl", [
        {
            "run_id": run_id, "trajectory_id": trajectory_id,
            "state_id": state_id, "environment_origin": origin,
            "collector_success": True, "traj_success": True,
        },
    ])
    manifest = root / "manifests" / "collection_runs" / f"{run_id}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "schema_version": "iris.collection-run.v2",
        "run_id": run_id,
        "status": "COMPLETE",
        "trajectories": [{
            "run_id": run_id, "trajectory_id": trajectory_id,
            "environment_origin": origin, "success": True,
            "collector_success": True,
        }],
    }))
    return {
        "sample_id": "formal-sample",
        "meta": {
            "formal_dataset": True,
            "run_id": run_id,
            "trajectory_id": trajectory_id,
            "state_id": state_id,
            "environment_origin": origin,
            "collector_success": True,
        },
    }


def test_formal_lineage_uses_exact_run_closure_not_frozen_legacy_inventory(
        tmp_path):
    formal = _formal_collection_fixture(tmp_path)
    scoped = AUDIT.audit_formal_collection_lineage(tmp_path, [formal])
    historical = AUDIT.audit_collection_lineage(tmp_path)

    assert scoped["ok"] is True
    assert scoped["legacy_quarantine_excluded_from_gate"] is True
    assert scoped["n_formal_rows"] == 1
    assert historical["ok"] is False
    assert historical["duplicate_meta_trajectory_ids"] == ["legacy-duplicate"]
    assert len(historical["unknown_environment_origin_rows"]) == 2


def test_formal_lineage_ignores_derived_point_state_view(tmp_path):
    formal = _formal_collection_fixture(tmp_path)
    source_state = json.loads((
        tmp_path / "raw" / "state_bank" / "shopping_key_states.jsonl"
    ).read_text().strip())
    _write_jsonl(
        tmp_path / "raw" / "state_bank" /
        "formal_point_reached_states.jsonl",
        [dict(source_state, probe_point_id="point-1")],
    )

    report = AUDIT.audit_formal_collection_lineage(
        tmp_path, [formal, dict(formal, sample_id="formal-multiturn")])

    assert report["n_formal_rows"] == 2
    assert report["n_unique_state_ids"] == 1
    assert report["ok"] is True
    assert "state_join_not_one_to_one" not in report["issues"]


def test_formal_lineage_rejects_unknown_environment_origin(tmp_path):
    formal = _formal_collection_fixture(tmp_path, origin="unknown")
    report = AUDIT.audit_formal_collection_lineage(tmp_path, [formal])
    assert report["ok"] is False
    assert report["issues"]["unknown_environment_origin"] == [
        "formal-sample",
        "run-1/traj-1/state-1:manifest",
        "run-1/traj-1/state-1:meta",
        "run-1/traj-1/state-1:state",
    ]


def test_dpo_deployment_source_share_is_non_vacuous_and_release_wide():
    assert AUDIT.deployment_source_share([])["passed"] is False
    half = AUDIT.deployment_source_share([
        {"meta": {"negative_source": "synthetic_flip"}},
        {"meta": {"negative_source": "legal_candidate"}},
    ])
    assert half["legal_or_on_policy_share"] == .5
    assert half["passed"] is True
    synthetic = AUDIT.deployment_source_share([
        {"meta": {"negative_source": "synthetic_flip"}},
    ])
    assert synthetic["legal_or_on_policy_share"] == 0.0
    assert synthetic["passed"] is False


def test_evaluation_truth_body_requires_matching_manifest(tmp_path):
    _write_jsonl(tmp_path / "eval" / "truth.jsonl", [{"incomplete": True}])
    context = AUDIT.formal_release_context(tmp_path)
    report = AUDIT.evaluation_truth_audit(tmp_path, context)
    assert report["n_records"] == 1
    assert report["integrity"] is False
    assert "must either both exist or both be absent" in report["error"]


def test_formal_dpo_gate_requires_exact_train_split_and_deployment_source(
        tmp_path, monkeypatch):
    source = {
        "sample_id": "sample-1",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": "chosen"},
        ],
        "meta": {"formal_dataset": True},
    }
    pair = {
        "pair_id": "sample-1__error-1",
        "prompt": source["messages"][:-1],
        "chosen": "chosen",
        "rejected": "rejected",
        "meta": {
            "formal_dataset": True,
            "source_sample_id": "sample-1",
            "negative_source": "legal_candidate",
        },
    }
    _write_jsonl(
        tmp_path / "train" / "formal" / config.FORMAL_DPO_PATH.name,
        [pair])
    monkeypatch.setattr(AUDIT, "validate_dpo_rows", lambda *args, **kwargs: {
        "n_rows": 1, "n_problems": 0, "problems": [],
        "negative_source_dist": {"legal_candidate": 1},
        "deployment_negative_ratio": 1.0,
    })
    monkeypatch.setattr(AUDIT, "_release_failures", lambda *args: {})
    monkeypatch.setattr(AUDIT, "formal_derivation_reasons", lambda *args: [])

    missing = AUDIT.formal_dpo_audit(
        tmp_path, [source], [source], [], object())
    assert missing["n_body_pairs"] == 1
    assert missing["n_train_pairs"] == 0
    assert missing["split_integrity"] is False
    assert missing["issues"]["split_missing_expected_pair"] == [
        "train:sample-1__error-1"]
    assert missing["source_share_passed"] is False

    _write_jsonl(
        tmp_path / "train" / "formal" / "splits" / "dpo_train.jsonl",
        [pair])
    complete = AUDIT.formal_dpo_audit(
        tmp_path, [source], [source], [], object())
    assert complete["issues"] == {}
    assert complete["integrity"] is True
    assert complete["split_integrity"] is True
    assert complete["source_share_passed"] is True


def test_readiness_does_not_count_unpinned_dpo_supplement(tmp_path):
    source = {
        "sample_id": "sample-1",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": "chosen"},
        ],
        "meta": {"formal_dataset": True},
    }
    pair = {
        "pair_id": "sample-1__on-policy",
        "prompt": source["messages"][:-1],
        "chosen": "chosen",
        "rejected": "bad",
        "meta": {
            "formal_dataset": True,
            "source_sample_id": "sample-1",
            "negative_source": "on_policy",
        },
    }
    path = tmp_path / "train" / "formal" / \
        "iris_dpo_on_policy_transition_v3_strict_unpinned.jsonl"
    _write_jsonl(path, [pair])
    report = AUDIT.formal_dpo_audit(
        tmp_path, [source], [source], [], object())
    assert report["n_body_pairs"] == 0
    assert report["integrity"] is False
    assert str(path) in report["supplement_failures"]
    assert "supplement_manifest_integrity" in report["issues"]


def test_readiness_excludes_manifest_pinned_supplement_from_old_release(
        tmp_path, monkeypatch):
    source = {
        "sample_id": "sample-1",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": "chosen"},
        ],
        "meta": {"formal_dataset": True},
    }

    def pair(pair_id):
        return {
            "pair_id": pair_id,
            "prompt": source["messages"][:-1],
            "chosen": "chosen", "rejected": "rejected",
            "meta": {
                "formal_dataset": True,
                "source_sample_id": "sample-1",
                "negative_source": "on_policy",
            },
        }

    base = tmp_path / "train" / "formal"
    old = base / "iris_dpo_on_policy_openrouter_single_20260714_v1.jsonl"
    active = base / "iris_dpo_on_policy_transition_v3_strict_fixture.jsonl"
    _write_jsonl(old, [pair("old-pair")])
    _write_jsonl(active, [pair("active-pair")])
    write_formal_dpo_supplement_manifest(old, release_id="old-v1")
    write_formal_dpo_supplement_manifest(active, release_id="transition-v3")
    monkeypatch.setattr(AUDIT, "validate_dpo_rows", lambda rows, **_kwargs: {
        "n_rows": len(rows), "n_problems": 0, "problems": [],
        "negative_source_dist": {"on_policy": len(rows)},
        "deployment_negative_ratio": 1.0,
    })
    monkeypatch.setattr(AUDIT, "_release_failures", lambda *args: {})
    monkeypatch.setattr(AUDIT, "formal_derivation_reasons", lambda *args: [])

    report = AUDIT.formal_dpo_audit(
        tmp_path, [source], [source], [], object())
    assert report["n_body_pairs"] == 1
    assert [row["path"] for row in report["supplement_releases"]] == [
        str(active)]
    assert all(str(old) != row["path"]
               for row in report["supplement_releases"])


def test_bid_audit_checks_pinned_risky_action_not_only_answer():
    row = {
        "sample_id": "sample-1",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": (
                "<goal>g</goal>\n<history></history>\n"
                "<observation>\n[3] button 'safe'</observation>")},
            {"role": "assistant", "content": "<answer> click('3')"},
        ],
        "meta": {"risky_raw_action": "click('9')"},
    }
    report = AUDIT.bid_visibility([row])
    assert report["assistant_answers"]["passed"] is True
    assert report["assistant_answers"]["visibility_rate"] == 1.0
    assert report["risky_raw_actions"]["passed"] is False
    assert report["risky_raw_actions"]["missing_ids"] == [
        "sample-1:risky_raw_action:bid-9"]
    assert report["passed"] is False


def test_candidate_audit_requires_manifest_and_replays_snapshot_legality(tmp_path):
    axtree = "[7] button 'Do it'"
    state_dir = tmp_path / "raw" / "state_bank"
    state_dir.mkdir(parents=True)
    (state_dir / "fixture.jsonl").write_text(json.dumps({
        "state_id": "state-1", "axtree_snapshot": axtree,
    }) + "\n")
    path = tmp_path / "raw" / "candidates" / FORMAL_CANDIDATE_BODY_NAME
    save_candidate_set([Candidate(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        candidate_id="candidate-1", state_id="state-1", bid="7",
        canonical_action="click:button:do-it", category="expert_action",
        source="expert", legal_at_snapshot=True,
        proposer_model="fixture", proposer_version="v1",
        snapshot_hash=snapshot_sha256(axtree),
    )], path)

    report = AUDIT.candidate_audit(tmp_path)
    assert report["integrity"] is True
    assert report["n_snapshot_matched_and_legal"] == 1

    path.with_name(FORMAL_CANDIDATE_MANIFEST_NAME).unlink()
    broken = AUDIT.candidate_audit(tmp_path)
    assert broken["integrity"] is False
    assert "must both exist" in broken["error"]


def test_candidate_role_audit_reports_coverage_and_same_case_overlap(
        tmp_path, monkeypatch):
    first = _point("point-1", site="shopping", instance="shopping:7770")
    second = _point(
        "point-2", site="reddit", instance="reddit:9999",
        action_type="reddit_vote")
    points = {point.probe_point_id: point for point in (first, second)}

    def role_record(record_id, role, case_id, point, status="EVIDENCED"):
        return SimpleNamespace(
            candidate_role_id=record_id,
            role=role,
            status=status,
            evaluation_case_id=case_id,
            probe_point_id=point.probe_point_id,
            state_id=point.state_id,
        )

    records = {
        "role-expert-1": role_record(
            "role-expert-1", "expert", "case-1", first),
        "role-safe-1": role_record(
            "role-safe-1", "safe_alternative", "case-1", first),
        "role-expert-2": role_record(
            "role-expert-2", "expert", "case-2", second),
        "role-failed": role_record(
            "role-failed", "ordinary", "case-2", second),
        "role-proposed": role_record(
            "role-proposed", "decoy", "case-3", first, "PROPOSED"),
    }
    replay = {
        "integrity": False,
        "failures": {"role-failed": ["fixture replay failure"]},
        "n_records": len(records),
        "evidenced_role": {"expert": 2, "safe_alternative": 1},
        "proposed_role": {"decoy": 1},
        "evidence_protocol": {},
    }
    monkeypatch.setattr(
        AUDIT, "assert_candidate_manifest_integrity",
        lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        AUDIT, "assert_candidate_role_manifest_integrity",
        lambda *_args, **_kwargs: records)
    monkeypatch.setattr(
        AUDIT, "validate_candidate_role_evidence",
        lambda *_args, **_kwargs: replay)
    monkeypatch.setattr(
        AUDIT, "discover_on_policy_role_sources",
        lambda *_args, **_kwargs: {})

    report = AUDIT.candidate_role_evidence_audit(tmp_path, points, {})
    expert = report["evidenced_role_coverage"]["expert"]
    assert expert == {
        "n_records": 2,
        "n_unique_evaluation_cases": 2,
        "unique_evaluation_cases": ["case-1", "case-2"],
        "n_unique_states": 2,
        "unique_states": ["state-point-1", "state-point-2"],
        "n_unique_sites": 2,
        "unique_sites": ["reddit", "shopping"],
        "n_unique_action_types": 2,
        "unique_action_types": ["add_to_cart", "reddit_vote"],
    }
    assert set(report["evidenced_role_coverage"]) == {
        "expert", "safe_alternative"}
    overlap = report["evidenced_case_role_overlap"]
    assert overlap["n_evidenced_cases"] == 2
    assert overlap["n_overlapping_cases"] == 1
    assert overlap["overlapping_case_ids"] == ["case-1"]
    assert overlap["cases"] == [{
        "evaluation_case_id": "case-1",
        "n_records": 2,
        "n_unique_roles": 2,
        "roles": ["expert", "safe_alternative"],
        "candidate_role_ids": ["role-expert-1", "role-safe-1"],
        "probe_point_ids": ["point-1"],
        "state_ids": ["state-point-1"],
        "sites": ["shopping"],
        "action_types": ["add_to_cart"],
    }]
    interpretation = report["coverage_interpretation"]
    assert interpretation["minimum_records_per_required_role_for_gate"] == 1
    assert interpretation["supports_n_ge_30_effect_claims"] is False
    assert "not n>=30 evidence" in interpretation["note"]


def test_opinion_gate_requires_every_truth_case_human_and_llm_with_manifest(
        tmp_path):
    point = _point()
    truth = _truth(point)
    points = {point.probe_point_id: point}
    truths = {truth.evaluation_case_id: truth}
    body = tmp_path / "opinions" / AUDIT.OPINION_BODY_NAME
    manifest = tmp_path / "opinions" / AUDIT.OPINION_MANIFEST_NAME

    # Two LLM identities are still not the promised human+LLM comparison.
    llm_records = [
        _opinion(point, truth, "llm-seed-1", RATER_LLM),
        _opinion(point, truth, "llm-seed-2", RATER_LLM),
    ]
    save_opinion_records(
        llm_records, body, manifest, points=points, truths=truths)
    llm_only = AUDIT.opinion_evidence_audit(tmp_path, points, truths)
    assert llm_only["integrity"] is True
    assert llm_only["rater_counts_by_case"] == {"case-1": 2}
    assert llm_only["human_rater_counts_by_case"] == {"case-1": 0}
    assert llm_only["passed"] is False

    human = _opinion(point, truth, "human-1", RATER_HUMAN)
    save_opinion_records(
        [human], body, manifest, append=True, points=points, truths=truths)
    paired = AUDIT.opinion_evidence_audit(tmp_path, points, truths)
    assert paired["coverage"] == 1.0
    assert paired["human_rater_counts_by_case"] == {"case-1": 1}
    assert paired["llm_rater_counts_by_case"] == {"case-1": 2}
    assert paired["passed"] is True

    # A second canonical truth case without opinions drops full-case coverage.
    second = _truth(point, "case-2", "constraint")
    incomplete = AUDIT.opinion_evidence_audit(
        tmp_path, points, {**truths, second.evaluation_case_id: second})
    assert incomplete["passed"] is False
    assert incomplete["missing_evaluation_case_ids"] == ["case-2"]

    manifest_row = json.loads(manifest.read_text().splitlines()[0])
    manifest_row["record_sha256"] = "0" * 64
    lines = manifest.read_text().splitlines()
    lines[0] = json.dumps(manifest_row)
    manifest.write_text("\n".join(lines) + "\n")
    corrupted = AUDIT.opinion_evidence_audit(tmp_path, points, truths)
    assert corrupted["integrity"] is False
    assert "manifest hash mismatch" in corrupted["error"]


def test_environment_gate_counts_technology_families_not_site_or_url():
    points = [
        _point("shop", site="shopping", instance="shop:7770"),
        _point("admin", site="shopping_admin", instance="admin:7780"),
        _point("reddit", site="reddit", instance="reddit:9999"),
    ]
    report = AUDIT.environment_evidence_audit(points)
    assert report["technology_families"] == ["magento", "postmill"]
    assert report["n_family_instance_pairs"] == 3
    assert report["n_independent_families"] == 2
    assert report["passed"] is False

    report = AUDIT.environment_evidence_audit(
        points + [_point("gitlab", site="gitlab", instance="gitlab:8023")])
    assert report["n_independent_families"] == 3
    assert report["passed"] is True


def test_api_db_signal_gate_requires_hashed_point_joined_asset(tmp_path):
    evidence = {
        "candidate_snapshot_hash": "candidate-snapshot-hash",
        "signal_channels": ["ui_structural", "api"],
    }
    point = _point(evidence=evidence)
    missing = AUDIT.signal_evidence_audit(tmp_path, [point])
    assert missing["passed"] is False
    assert missing["errors"]["point-1:api"] == [
        "requires exactly one signal evidence reference"]

    normalized = {
        "pre": {"cart_item_ids": []},
        "post": {"cart_item_ids": ["sku-1"]},
        "final": {"cart_item_ids": []},
    }
    snapshot_refs = []
    for phase in ("pre", "post", "final"):
        snapshot = {
            "schema_version": AUDIT.SIGNAL_SNAPSHOT_SCHEMA_VERSION,
            "probe_point_id": point.probe_point_id,
            "channel": "api",
            "phase": phase,
            "environment_instance": point.environment_instance,
            "observed_at": f"2026-07-14T00:00:0{len(snapshot_refs)}Z",
            "raw_payload": {"items": normalized[phase]["cart_item_ids"]},
            "normalized_state": normalized[phase],
        }
        snapshot_path = (
            tmp_path / "evidence" / "signals" / f"api-point-1-{phase}.json")
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps(snapshot, sort_keys=True))
        snapshot_refs.append({
            "schema_version": AUDIT.SIGNAL_SNAPSHOT_REF_SCHEMA_VERSION,
            "phase": phase,
            "path": str(snapshot_path.relative_to(tmp_path)),
            "sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
            "normalized_state_sha256":
                AUDIT._canonical_state_sha256(normalized[phase]),
        })
    asset = {
        "schema_version": AUDIT.SIGNAL_EVIDENCE_SCHEMA_VERSION,
        "evidence_id": "signal-evidence-1",
        "probe_point_id": point.probe_point_id,
        "channel": "api",
        "environment_instance": point.environment_instance,
        "collection_timestamp": "2026-07-14T00:00:00Z",
        "code_version": "deadbeef",
        "observer_version": "magento-rest-v1",
        "collection_run_id": "signal-run-1",
        "endpoint_or_query_sha256":
            hashlib.sha256(b"GET /V1/carts/mine/items").hexdigest(),
        "collected_live": True,
        "is_fixture": False,
        "read_only_observer": True,
        "credential_value_stored": False,
        "snapshots": snapshot_refs,
    }
    asset_path = tmp_path / "evidence" / "signals" / "api-point-1.json"
    asset_path.write_text(json.dumps(asset, sort_keys=True))
    digest = hashlib.sha256(asset_path.read_bytes()).hexdigest()
    point = replace(point, evidence={
        **evidence,
        "api_db_signal_summaries": {"api": normalized},
        "signal_evidence_assets": [{
            "schema_version": AUDIT.SIGNAL_EVIDENCE_REF_SCHEMA_VERSION,
            "probe_point_id": point.probe_point_id,
            "channel": "api",
            "path": "evidence/signals/api-point-1.json",
            "sha256": digest,
        }],
    })
    verified = AUDIT.signal_evidence_audit(tmp_path, [point])
    assert verified["passed"] is True
    assert verified["verified_assets"] == ["point-1:api"]

    post_path = tmp_path / snapshot_refs[1]["path"]
    post_snapshot = json.loads(post_path.read_text())
    post_snapshot["raw_payload"] = {"items": ["tampered"]}
    post_path.write_text(json.dumps(post_snapshot, sort_keys=True))
    tampered = AUDIT.signal_evidence_audit(tmp_path, [point])
    assert tampered["passed"] is False
    assert "post snapshot hash mismatch" in tampered["errors"]["point-1:api"]


def test_api_db_signal_gate_rejects_repackaged_ui_signal(tmp_path):
    point = _point(evidence={
        "candidate_snapshot_hash": "candidate-snapshot-hash",
        "signal_channels": ["ui_structural", "api"],
    })
    asset = {
        "schema_version": AUDIT.SIGNAL_EVIDENCE_SCHEMA_VERSION,
        "evidence_id": "repackaged-ui",
        "probe_point_id": point.probe_point_id,
        "channel": "api",
        "pre_signal": point.pre_signal,
        "post_signal": point.post_signal,
        "collection_timestamp": "2026-07-14T00:00:00Z",
        "code_version": "deadbeef",
    }
    path = tmp_path / "evidence" / "signals" / "repackaged-ui.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(asset, sort_keys=True))
    point = replace(point, evidence={
        **point.evidence,
        "signal_evidence_assets": [{
            "schema_version": AUDIT.SIGNAL_EVIDENCE_REF_SCHEMA_VERSION,
            "probe_point_id": point.probe_point_id,
            "channel": "api",
            "path": str(path.relative_to(tmp_path)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }],
    })
    report = AUDIT.signal_evidence_audit(tmp_path, [point])
    assert report["passed"] is False
    errors = report["errors"]["point-1:api"]
    assert "signal evidence snapshots must be a list" in errors
    assert "signal evidence was not collected live" in errors


def test_mutation_gate_is_data_root_release_scoped_and_manifest_pinned(tmp_path):
    # A similarly named repository/global report is intentionally irrelevant;
    # only the selected data root can satisfy the gate.
    # These schema fixtures exist only under pytest's temporary directory and
    # are never written to the repository's canonical data root.
    absent = AUDIT.mutation_evidence_audit(tmp_path)
    assert absent["passed"] is False
    snapshot = "RootWebArea 'Census'\n" + "".join(
        f"  [{index}] button 'Control {index}'\n" for index in range(200))
    snapshot_hash = snapshot_sha256(snapshot)
    controls = [(str(index), f"click('{index}')") for index in range(200)]
    control_set_hash = mutation_control_set_sha256(controls)
    samples = []

    def digest(value):
        return hashlib.sha256(value.encode()).hexdigest()

    for index, (bid, action) in enumerate(controls):
        detected = index % 2 == 0
        reference = index % 3 == 0
        control_id = f"control-{index:03d}"
        sample_id = make_mutation_sample_id(
            collection_run_id="live-run-1",
            environment_instance="unit-test://isolated", state_id="state-1",
            snapshot_hash=snapshot_hash, control_id=control_id, bid=bid,
            canonical_action=action)
        samples.append(MutationCensusSample(
            schema_version=MUTATION_SAMPLE_SCHEMA_VERSION,
            mutation_sample_id=sample_id, collection_run_id="live-run-1",
            state_id="state-1", control_id=control_id, bid=bid,
            canonical_action=action, role="button", name=f"Control {index}",
            environment_family="unit-schema",
            environment_instance="unit-test://isolated",
            snapshot_hash=snapshot_hash, snapshot_control_count=200,
            control_set_sha256=control_set_hash, legal_at_snapshot=True,
            executed_live=True, is_fixture=False,
            safety_class="non_destructive", review_status="approved",
            destructive_commit_authorized=False, destructive_env_gate=False,
            detector_channels=("ui",),
            changed_channels=("ui",) if detected else (),
            mutation_candidate=detected, reference_mutated=reference,
            reference_source="independent_db_diff",
            reference_evidence_sha256=digest(f"reference-{index}"),
            execution_evidence_sha256=digest(f"execution-{index}"),
            pre_signal_sha256=digest("anchor"),
            post_signal_sha256=digest(
                f"post-{index}" if detected else "anchor"),
            reset_signal_sha256=digest("anchor"), reset_restored=True,
            async_pending=False))
    state_path = tmp_path / "raw" / "state_bank" / "census.jsonl"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps({
        "state_id": "state-1", "axtree_snapshot": snapshot,
        "axtree_complete": True,
        "capture_protocol": "fixture-full-axtree.v1",
    }) + "\n")
    body = tmp_path / "audit" / MUTATION_REPORT_BODY_NAME
    manifest = tmp_path / "audit" / MUTATION_REPORT_MANIFEST_NAME
    save_live_mutation_census(
        samples, body=body, manifest=manifest,
        release_id="release-20260714",
        collection_timestamp="2026-07-14T00:00:00Z",
        code_version="deadbeef", protocol_id="full-control-enumeration-v1")
    verified = AUDIT.mutation_evidence_audit(tmp_path)
    assert verified["passed"] is True
    assert verified["n_samples"] == 200

    report = json.loads(body.read_text())
    report["samples"][0]["is_fixture"] = True
    body.write_text(json.dumps(report, sort_keys=True))
    tampered = AUDIT.mutation_evidence_audit(tmp_path)
    assert tampered["passed"] is False
    assert "fixture/unspecified" in tampered["error"]


def test_prompt_content_gate_and_canonical_distilled_path_are_audited(tmp_path):
    bundle = store_bundle(
        {"agent_system": "expected immutable system"},
        root=tmp_path, author="readiness-test")
    row = {
        "sample_id": "sample-1",
        "messages": [
            {"role": "system", "content": "tampered system"},
            {"role": "user", "content": "<observation>\n[3] button 'do'"},
            {"role": "assistant", "content": "<answer> click('3')"},
        ],
        "meta": {
            "formal_dataset": True,
            "prompts_fp": bundle.stem,
            "risky_raw_action": "click('3')",
        },
    }
    formal = tmp_path / "train" / "formal"
    formal.mkdir(parents=True)
    encoded = json.dumps(row) + "\n"
    (formal / config.FORMAL_SFT_PATH.name).write_text(encoded)
    # A stale legacy filename must not be counted as formal teacher coverage.
    (formal / "revact_sft_distilled.jsonl").write_text(encoded)

    _, no_teacher = _run(tmp_path, "--allow-blocked")
    failures = no_teacher["formal_training"]["release_gate_failures"]
    assert "prompt_bundle_system_mismatch" in failures["sample-1"]
    content_failures = no_teacher["formal_training"]["prompt_content_failures"]
    assert "prompt_bundle_system_mismatch" in content_failures["sample-1"]
    assert any(reason.startswith("completion_")
               for reason in content_failures["sample-1"])
    assert no_teacher["teacher"]["n_rows"] == 0
    assert no_teacher["teacher"]["coverage"] == 0.0

    canonical = formal / config.FORMAL_DISTILLED_SFT_PATH.name
    canonical.write_text(encoded)
    _, with_teacher = _run(tmp_path, "--allow-blocked")
    assert with_teacher["teacher"]["path"] == str(canonical)
    assert with_teacher["teacher"]["n_rows"] == 1
    assert with_teacher["teacher"]["artifact_coverage"] == 1.0
    assert with_teacher["teacher"]["coverage"] == 0.0
    assert with_teacher["teacher"]["families"]["single"][
        "n_qualified_teacher_rows"] == 0
    assert "teacher_prose_source_not_teacher" in with_teacher["teacher"][
        "teacher_source_failures"]["single:sample-1"]
    assert "prompt_bundle_system_mismatch" in with_teacher["teacher"][
        "release_or_prompt_failures"]["single:sample-1"]


def test_teacher_audit_aggregates_both_families_and_requires_teacher_sources(
        tmp_path, monkeypatch):
    sources = {
        "single": [{"sample_id": "single-1"}],
        "multiturn": [{"sample_id": "multi-1"}],
    }

    def teacher_row(sample_id):
        return {
            "sample_id": sample_id,
            "messages": [{"role": "assistant", "content": "fixture"}],
            "meta": {
                "prose_source": "teacher",
                "rev_check_source": "teacher",
                "teacher_qc_status": "passed",
            },
        }

    formal = tmp_path / "train" / "formal"
    _write_jsonl(
        formal / config.FORMAL_DISTILLED_SFT_PATH.name,
        [teacher_row("single-1")])
    _write_jsonl(
        formal / config.FORMAL_MULTITURN_DISTILLED_SFT_PATH.name,
        [teacher_row("multi-1")])
    monkeypatch.setattr(AUDIT, "_release_failures", lambda *_args: {})
    monkeypatch.setattr(AUDIT, "qc_full_sample", lambda _row: None)

    report = AUDIT.teacher_audit(tmp_path, sources, object())
    assert report["n_unique_source_rows"] == 2
    assert report["n_rows"] == 2
    assert report["n_qualified_teacher_rows"] == 2
    assert report["coverage"] == 1.0
    assert report["passed"] is True
    assert report["families"]["single"]["coverage"] == 1.0
    assert report["families"]["multiturn"]["coverage"] == 1.0

    invalid = teacher_row("multi-1")
    invalid["meta"]["rev_check_source"] = "template"
    _write_jsonl(
        formal / config.FORMAL_MULTITURN_DISTILLED_SFT_PATH.name, [invalid])
    blocked = AUDIT.teacher_audit(tmp_path, sources, object())
    assert blocked["n_qualified_teacher_rows"] == 1
    assert blocked["coverage"] == .5
    assert blocked["families"]["multiturn"]["passed"] is False
    assert blocked["passed"] is False
    assert blocked["teacher_source_failures"]["multiturn:multi-1"] == [
        "teacher_rev_check_source_not_teacher"]
