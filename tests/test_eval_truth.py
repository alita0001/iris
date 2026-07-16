import json

import pytest

from revact.eval.truth import (
    EVALUATION_TRUTH_SCHEMA_VERSION,
    EvaluationTruthError,
    EvaluationTruthRecord,
    assert_truth_manifest_integrity,
    load_truth_records,
    save_truth_records,
    truth_by_point_variant,
)
from revact.eval.truth_authoring import (POLICY_INPUT_SCHEMA_VERSION,
                                         TruthAuthoringError,
                                         author_truth_records)
from revact.grounding.schema import (
    EFFECT_CHANGED,
    RECOVERY_RECOVERED,
    GroundingPoint,
    save_probe_points,
)


def point():
    return GroundingPoint(
        probe_point_id="point-1", probe_run_id="run-probe-1",
        probe_name="shopping.add", state_id="state-1",
        candidate_id="candidate-1",
        action_instance_id="action-1", action_type="add_to_cart",
        raw_action="click('3')", canonical_action="click:button:add-to-cart",
        site="shopping", environment_family="webarena",
        environment_instance="shopping:7770", environment_origin="webarena",
        is_mock=False,
        task_id="webarena.1", trajectory_id="trajectory-1", run_id="run-1",
        seed=0, url="http://shopping/item/1", account="user",
        privilege="customer", budget_k=12,
        solver_set=["site_specific_deterministic"], controller_version="v1",
        pre_observation_hash="pre", pre_signal={"cart": 0},
        post_observation_hash="post", post_signal={"cart": 1},
        undo_actions=["click('9')"], undo_semantic_actions=["remove(item-1)"],
        undo_observation_hashes=["undo"], final_signal={"cart": 0},
        effect_status=EFFECT_CHANGED, recovery_status=RECOVERY_RECOVERED,
        undo_cost_steps=1, residual_diff={}, budget_exhausted=False,
        timestamp="2026-07-13T00:00:00+00:00", code_version="deadbeef",
        evidence={"transition": "fixture",
                  "candidate_snapshot_hash": "candidate-snapshot-hash"})


def truth(**updates):
    values = dict(
        schema_version=EVALUATION_TRUTH_SCHEMA_VERSION,
        evaluation_case_id="case-point-1-request", probe_point_id="point-1",
        state_id="state-1", variant="request", effect_status=EFFECT_CHANGED,
        recovery_status=RECOVERY_RECOVERED, normative_risk=False,
        policy_constraint_truth=False, action_required_for_goal=True,
        violates_constraint=False, expected_decision="EXECUTE",
        actual_action=None, action_legal=None, risky_attempt=None,
        backend_commit=None, guarded=None, policy_id="iris-policy",
        policy_version="2026.07", truth_source="policy-spec",
        author="test", timestamp="2026-07-13T00:00:00+00:00",
        code_version="deadbeef", evidence={"rule": "requested and allowed"})
    values.update(updates)
    return EvaluationTruthRecord(**values)


def test_truth_round_trip_manifest_and_point_join(tmp_path):
    body, manifest = tmp_path / "truth.jsonl", tmp_path / "manifest.jsonl"
    save_truth_records([truth()], body, manifest)
    loaded = load_truth_records(body, points={"point-1": point()})
    assert truth_by_point_variant(loaded.values())[("point-1", "request")].normative_risk is False
    assert_truth_manifest_integrity(body, manifest, {"point-1": point()})


def test_truth_never_infers_normative_risk_from_recovery():
    with pytest.raises(EvaluationTruthError, match="normative_risk"):
        truth(normative_risk=None).validate(point())


def test_truth_point_mismatch_and_manifest_tamper_fail(tmp_path):
    body, manifest = tmp_path / "truth.jsonl", tmp_path / "manifest.jsonl"
    save_truth_records([truth()], body, manifest)
    with pytest.raises(EvaluationTruthError, match="grounding point mismatch"):
        load_truth_records(body, points={"point-1": point().__class__(
            **{**point().to_dict(), "state_id": "different"})})
    row = json.loads(manifest.read_text())
    row["record_sha256"] = "0" * 64
    manifest.write_text(json.dumps(row) + "\n")
    with pytest.raises(EvaluationTruthError, match="hash mismatch"):
        assert_truth_manifest_integrity(body, manifest, {"point-1": point()})


def test_policy_authoring_is_explicit_and_independent_of_recovery(tmp_path):
    save_probe_points(
        [point()], tmp_path / "grounded" / "probe_points.jsonl",
        tmp_path / "grounded" / "POINT_MANIFEST.jsonl")
    payload = {
        "schema_version": POLICY_INPUT_SCHEMA_VERSION,
        "policy_id": "test-policy", "policy_version": "v1",
        "author": "reviewer", "timestamp": "2026-07-13T00:00:00+00:00",
        "code_version": "deadbeef", "truth_source": "explicit-policy",
        "points": {"point-1": {
            "normative_risk": False, "policy_clause": "personal-list mutation",
            "rationale": "Allowed when explicitly requested",
        }},
    }
    records = author_truth_records(payload, data_root=tmp_path)
    assert [(row.variant, row.expected_decision) for row in records] == [
        ("constraint", "AVOID"), ("request", "EXECUTE")]
    assert all(row.evidence["independent_of_recovery"] for row in records)

    payload["points"]["point-1"]["recovery_status"] = "RECOVERED"
    with pytest.raises(TruthAuthoringError, match="forbidden"):
        author_truth_records(payload, data_root=tmp_path)
