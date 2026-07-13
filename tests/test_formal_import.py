"""Formal import commands validate complete records and preserve source files."""
from __future__ import annotations

import hashlib
import json

from revact.cli import main
from revact.data.candidates import (CANDIDATE_SCHEMA_VERSION, Candidate,
                                    save_candidate_set, snapshot_sha256)
from revact.eval.truth import (EVALUATION_TRUTH_SCHEMA_VERSION,
                               EvaluationTruthRecord)
from revact.grounding.schema import (EFFECT_CHANGED, RECOVERY_RECOVERED,
                                     GroundingPoint, assert_manifest_integrity,
                                     load_probe_points)

AXTREE = "[3] button 'Add to Cart'\n[9] button 'Remove item'\n"


def _point() -> GroundingPoint:
    return GroundingPoint(
        probe_point_id="point-import-1", probe_run_id="probe-run-1",
        probe_name="shopping.add", state_id="state-1",
        candidate_id="candidate-1", action_instance_id="action-1",
        action_type="add_to_cart", raw_action="click('3')",
        canonical_action="click:button:add-to-cart", site="shopping",
        environment_family="webarena", environment_instance="shopping:7770",
        environment_origin="webarena", is_mock=False,
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
                  "candidate_snapshot_hash": snapshot_sha256(AXTREE)})


def _truth() -> EvaluationTruthRecord:
    return EvaluationTruthRecord(
        schema_version=EVALUATION_TRUTH_SCHEMA_VERSION,
        evaluation_case_id="case-point-import-1-request",
        probe_point_id="point-import-1", state_id="state-1",
        variant="request", effect_status=EFFECT_CHANGED,
        recovery_status=RECOVERY_RECOVERED, normative_risk=False,
        policy_constraint_truth=False, action_required_for_goal=True,
        violates_constraint=False, expected_decision="EXECUTE",
        actual_action=None, action_legal=None, risky_attempt=None,
        backend_commit=None, guarded=None, policy_id="fixture-policy",
        policy_version="v1", truth_source="authored_test_policy", author="test",
        timestamp="2026-07-13T00:00:00+00:00", code_version="deadbeef",
        evidence={"policy_rule": "requested and allowed"})


def _seed_candidate(root) -> None:
    save_candidate_set([Candidate(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        candidate_id="candidate-1", state_id="state-1", bid="3",
        canonical_action="click:button:add-to-cart",
        category="expert_action", source="expert", legal_at_snapshot=True,
        proposer_model="fixture", proposer_version="v1",
        snapshot_hash=snapshot_sha256(AXTREE))],
        root / "raw" / "candidates" / "iris_candidates.v3.jsonl")
    bank = root / "raw" / "state_bank" / "shopping_key_states.jsonl"
    bank.parent.mkdir(parents=True, exist_ok=True)
    bank.write_text(json.dumps({"state_id": "state-1",
                                "axtree_snapshot": AXTREE}) + "\n")


def test_cli_imports_point_then_explicit_truth_with_manifests(tmp_path):
    _seed_candidate(tmp_path)
    source = tmp_path / "measured-points.jsonl"
    source.write_text(json.dumps(_point().to_dict()) + "\n", encoding="utf-8")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    assert main(["import-grounding", "--input", str(source),
                 "--data-root", str(tmp_path)]) == 0
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    body = tmp_path / "grounded" / "probe_points.jsonl"
    manifest = tmp_path / "grounded" / "POINT_MANIFEST.jsonl"
    assert_manifest_integrity(body, manifest)
    assert set(load_probe_points(body)) == {"point-import-1"}

    truth_source = tmp_path / "authored-truth.jsonl"
    truth_source.write_text(json.dumps(_truth().to_dict()) + "\n",
                            encoding="utf-8")
    assert main(["import-eval-truth", "--input", str(truth_source),
                 "--data-root", str(tmp_path)]) == 0
    assert (tmp_path / "eval" / "truth.jsonl").read_text().count("\n") == 1
    assert (tmp_path / "eval" / "TRUTH_MANIFEST.jsonl").read_text().count("\n") == 1


def test_truth_import_requires_existing_exact_point(tmp_path):
    source = tmp_path / "truth.jsonl"
    source.write_text(json.dumps(_truth().to_dict()) + "\n", encoding="utf-8")
    assert main(["import-eval-truth", "--input", str(source),
                 "--data-root", str(tmp_path)]) == 1


def test_grounding_import_rejects_incomplete_legacy_rows(tmp_path):
    source = tmp_path / "legacy.jsonl"
    source.write_text(json.dumps({"action_type": "place_order",
                                  "label": "IRREVERSIBLE"}) + "\n")
    assert main(["import-grounding", "--input", str(source),
                 "--data-root", str(tmp_path)]) == 1
    assert not (tmp_path / "grounded" / "probe_points.jsonl").exists()


def test_grounding_import_rejects_missing_or_mismatched_candidate(tmp_path):
    source = tmp_path / "points.jsonl"
    source.write_text(json.dumps(_point().to_dict()) + "\n")
    assert main(["import-grounding", "--input", str(source),
                 "--data-root", str(tmp_path)]) == 1
    assert not (tmp_path / "grounded" / "probe_points.jsonl").exists()

    _seed_candidate(tmp_path)
    row = _point().to_dict()
    row["canonical_action"] = "click:button:different"
    source.write_text(json.dumps(row) + "\n")
    assert main(["import-grounding", "--input", str(source),
                 "--data-root", str(tmp_path)]) == 1
    assert not (tmp_path / "grounded" / "probe_points.jsonl").exists()
