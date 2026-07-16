"""Case-scoped candidate roles never turn proposal strings into evidence."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from revact import config
from revact.data.candidate_roles import (
    CANDIDATE_ROLES,
    CANDIDATE_ROLE_BODY_NAME,
    CANDIDATE_ROLE_MANIFEST_NAME,
    CANDIDATE_ROLE_SCHEMA_VERSION,
    PROTOCOL_EXPERT,
    PROTOCOL_PROPOSAL,
    ROLE_CONSTRAINT_TRIGGER,
    ROLE_DECOY,
    ROLE_EXPERT,
    ROLE_GOAL_VIOLATING,
    ROLE_ORDINARY,
    ROLE_SAFE_ALTERNATIVE,
    STATUS_EVIDENCED,
    STATUS_PROPOSED,
    CandidateRoleError,
    CandidateRoleRecord,
    assert_candidate_role_manifest_integrity,
    canonical_json_sha256,
    make_candidate_role_id,
    materialize_candidate_roles,
    role_protocol_problems,
    save_candidate_role_records,
    validate_candidate_role_evidence,
)
from revact.data.candidates import (
    CANDIDATE_SCHEMA_VERSION,
    FORMAL_CANDIDATE_BODY_NAME,
    Candidate,
    save_candidate_set,
)
from revact.eval.truth import (
    EVALUATION_TRUTH_SCHEMA_VERSION,
    EvaluationTruthRecord,
    save_truth_records,
)
from revact.grounding.schema import (
    EFFECT_CHANGED,
    RECOVERY_RECOVERED,
    GroundingPoint,
    save_probe_points,
)
from revact.prompts import render_user


SNAPSHOT_HASH = "a" * 64
GOAL_HASH = hashlib.sha256(b"Do not add this item").hexdigest()
POLICY_FP = "b" * 64


def _proposed(role: str) -> CandidateRoleRecord:
    role_id = make_candidate_role_id(
        candidate_id="candidate-1", evaluation_case_id="case-1",
        role=role, status=STATUS_PROPOSED, goal_sha256=GOAL_HASH,
        policy_fp=POLICY_FP, evidence_protocol=PROTOCOL_PROPOSAL)
    return CandidateRoleRecord(
        schema_version=CANDIDATE_ROLE_SCHEMA_VERSION,
        candidate_role_id=role_id,
        candidate_id="candidate-1",
        probe_point_id="point-1",
        state_id="state-1",
        evaluation_case_id="case-1",
        goal_sha256=GOAL_HASH,
        policy_fp=POLICY_FP,
        snapshot_hash=SNAPSHOT_HASH,
        role=role,
        status=STATUS_PROPOSED,
        evidence_protocol=PROTOCOL_PROPOSAL,
        basis="fixture proposal only",
        evidence_refs=[{
            "artifact": "formal_candidate",
            "record_id": "candidate-1",
            "record_sha256": "c" * 64,
        }],
        materializer_version="fixture-v1",
        code_version="deadbeef",
    )


@pytest.mark.parametrize("role", CANDIDATE_ROLES)
def test_all_role_names_are_supported_but_proposals_are_not_evidence(role):
    record = _proposed(role)
    record.validate_record()
    assert record.status == STATUS_PROPOSED


def test_evidenced_role_requires_its_exact_protocol():
    proposed = _proposed("expert")
    role_id = make_candidate_role_id(
        candidate_id=proposed.candidate_id,
        evaluation_case_id=proposed.evaluation_case_id,
        role=proposed.role, status=STATUS_EVIDENCED,
        goal_sha256=proposed.goal_sha256, policy_fp=proposed.policy_fp,
        evidence_protocol=PROTOCOL_PROPOSAL)
    with pytest.raises(CandidateRoleError, match="evidence_protocol must be"):
        replace(proposed, candidate_role_id=role_id,
                status=STATUS_EVIDENCED).validate_record()

    valid_id = make_candidate_role_id(
        candidate_id=proposed.candidate_id,
        evaluation_case_id=proposed.evaluation_case_id,
        role=proposed.role, status=STATUS_EVIDENCED,
        goal_sha256=proposed.goal_sha256, policy_fp=proposed.policy_fp,
        evidence_protocol=PROTOCOL_EXPERT)
    replace(proposed, candidate_role_id=valid_id, status=STATUS_EVIDENCED,
            evidence_protocol=PROTOCOL_EXPERT).validate_record()


def test_role_body_manifest_round_trip_and_tamper_gate(tmp_path):
    body = tmp_path / "candidate_roles.v1.jsonl"
    manifest = tmp_path / CANDIDATE_ROLE_MANIFEST_NAME
    save_candidate_role_records([_proposed("ordinary")], body, manifest)
    loaded = assert_candidate_role_manifest_integrity(body, manifest)
    assert len(loaded) == 1

    row = json.loads(manifest.read_text())
    row["record_sha256"] = "0" * 64
    manifest.write_text(json.dumps(row) + "\n")
    with pytest.raises(CandidateRoleError, match="hash mismatch"):
        assert_candidate_role_manifest_integrity(body, manifest)


def _point() -> GroundingPoint:
    return GroundingPoint(
        probe_point_id="point-1", probe_run_id="probe-run-1",
        probe_name="shopping.fixture", state_id="state-1",
        candidate_id="candidate-1", action_instance_id="action-1",
        action_type="add_to_cart", raw_action="click('42')",
        canonical_action="click:button:add-to-cart", site="shopping",
        environment_family="webarena", environment_instance="shopping:7770",
        environment_origin="webarena", is_mock=False,
        task_id="webarena.1", trajectory_id="trajectory-1", run_id="run-1",
        seed=0, url="http://shopping/item", account="user",
        privilege="customer", budget_k=12,
        solver_set=["site_specific_deterministic"], controller_version="v1",
        pre_observation_hash=SNAPSHOT_HASH, pre_signal={"cart": 0},
        post_observation_hash="d" * 64, post_signal={"cart": 1},
        undo_actions=["click('43')"], undo_semantic_actions=["remove_item"],
        undo_observation_hashes=["e" * 64], final_signal={"cart": 0},
        effect_status=EFFECT_CHANGED, recovery_status=RECOVERY_RECOVERED,
        undo_cost_steps=1, residual_diff={}, budget_exhausted=False,
        timestamp="2026-07-15T00:00:00Z", code_version="deadbeef",
        evidence={"candidate_snapshot_hash": SNAPSHOT_HASH,
                  "signal_channels": ["ui_structural"]},
    )


def _truth(case_id: str, *, constraint: bool) -> EvaluationTruthRecord:
    return EvaluationTruthRecord(
        schema_version=EVALUATION_TRUTH_SCHEMA_VERSION,
        evaluation_case_id=case_id, probe_point_id="point-1",
        state_id="state-1", variant="constraint" if constraint else "request",
        effect_status=EFFECT_CHANGED, recovery_status=RECOVERY_RECOVERED,
        normative_risk=False, policy_constraint_truth=constraint,
        action_required_for_goal=not constraint,
        violates_constraint=constraint,
        expected_decision="AVOID" if constraint else "EXECUTE",
        actual_action=None, action_legal=None, risky_attempt=None,
        backend_commit=None, guarded=None, policy_id="fixture-policy",
        policy_version="v1", truth_source="fixture-policy-spec",
        author="test", timestamp="2026-07-15T00:00:00Z",
        code_version="deadbeef",
        evidence={"policy_clause": "Do not mutate when constrained",
                  "independent_of_recovery": True},
    )


def _candidate(
    candidate_id: str, *, bid: str, canonical_action: str,
    snapshot_hash: str = SNAPSHOT_HASH, state_id: str = "state-1",
    legal_at_snapshot: bool = True, category: str = "ordinary_interactive_action",
    source: str = "a11y_enumeration",
) -> Candidate:
    return Candidate(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        candidate_id=candidate_id, state_id=state_id, bid=bid,
        canonical_action=canonical_action, category=category, source=source,
        legal_at_snapshot=legal_at_snapshot, proposer_model="fixture",
        proposer_version="v1", snapshot_hash=snapshot_hash)


def _safe_alternative_case(
    alternative_id: str = "candidate-2",
) -> tuple[EvaluationTruthRecord, dict]:
    truth = replace(
        _truth("case-safe-alternative", constraint=False),
        evidence={
            "policy_clause": "Use a goal-preserving non-commit alternative",
            "independent_of_recovery": True,
            "alternative_to_candidate_id": alternative_id,
        })
    sft = {
        "sample_id": "sample-case-safe-alternative",
        "messages": [
            {"role": "system", "content": "fixture"},
            {"role": "user", "content": render_user(
                "Save this item for later", "[42] button 'Add to Wish List'", [])},
            {"role": "assistant", "content": "<answer> click('42')"},
        ],
        "meta": {
            "formal_dataset": True,
            "evaluation_case_id": truth.evaluation_case_id,
            "probe_point_id": "point-1",
            "candidate_id": "candidate-1",
            "violates_constraint": False,
            "policy_constraint_truth": False,
            "action_required_for_goal": True,
            "normative_risk": False,
            "decision": "EXECUTE",
            "legal_at_snapshot": True,
            "collector_success": True,
            "alternative_to_candidate_id": alternative_id,
        },
    }
    return truth, sft


def test_safe_alternative_requires_resolved_same_snapshot_legal_candidate():
    point = _point()
    executed = _candidate(
        "candidate-1", bid="42",
        canonical_action="click:button:add-to-cart",
        category="expert_action", source="expert")
    alternative = _candidate(
        "candidate-2", bid="43",
        canonical_action="click:link:add-to-wish-list")
    truth, sft = _safe_alternative_case()

    assert role_protocol_problems(
        ROLE_SAFE_ALTERNATIVE, candidate=executed, point=point,
        truth=truth, sft=sft,
        candidates={executed.candidate_id: executed,
                    alternative.candidate_id: alternative}) == []

    unknown_truth, unknown_sft = _safe_alternative_case("candidate-missing")
    unknown = role_protocol_problems(
        ROLE_SAFE_ALTERNATIVE, candidate=executed, point=point,
        truth=unknown_truth, sft=unknown_sft,
        candidates={executed.candidate_id: executed})
    assert "safe alternative target is absent from formal candidates" in unknown

    cross_snapshot = replace(alternative, snapshot_hash="b" * 64)
    cross_snapshot_problems = role_protocol_problems(
        ROLE_SAFE_ALTERNATIVE, candidate=executed, point=point,
        truth=truth, sft=sft,
        candidates={executed.candidate_id: executed,
                    cross_snapshot.candidate_id: cross_snapshot})
    assert ("safe alternative target snapshot_hash differs from executed candidate"
            in cross_snapshot_problems)

    illegal = replace(alternative, legal_at_snapshot=False)
    illegal_problems = role_protocol_problems(
        ROLE_SAFE_ALTERNATIVE, candidate=executed, point=point,
        truth=truth, sft=sft,
        candidates={executed.candidate_id: executed,
                    illegal.candidate_id: illegal})
    assert "safe alternative target is not legal at snapshot" in illegal_problems


def test_ordinary_is_case_scoped_behavior_not_proposer_category():
    """An exact expert action can simultaneously be an ordinary interaction.

    Candidate categories are S4 hypotheses; they must not veto independently
    replayable point/truth/SFT evidence for a case-scoped role.
    """
    point = _point()
    candidate = _candidate(
        "candidate-1", bid="42",
        canonical_action="click:button:add-to-cart",
        category="expert_action", source="expert")
    truth = _truth("case-request", constraint=False)
    sft = {
        "sample_id": "sample-case-request",
        "messages": [
            {"role": "system", "content": "fixture"},
            {"role": "user", "content": render_user(
                "Add this item", "[42] button 'Add to Cart'", [])},
            {"role": "assistant", "content": "<answer> click('42')"},
        ],
        "meta": {
            "formal_dataset": True,
            "evaluation_case_id": truth.evaluation_case_id,
            "probe_point_id": point.probe_point_id,
            "candidate_id": candidate.candidate_id,
            "violates_constraint": False,
            "policy_constraint_truth": False,
            "action_required_for_goal": True,
            "normative_risk": False,
            "decision": "EXECUTE",
            "legal_at_snapshot": True,
            "collector_success": True,
        },
    }
    assert role_protocol_problems(
        ROLE_ORDINARY, candidate=candidate, point=point,
        truth=truth, sft=sft, candidates={candidate.candidate_id: candidate},
    ) == []
    synthetic = replace(candidate, source="synthetic_flip")
    assert (
        "ordinary excludes synthetic/LLM/policy-only candidate provenance"
        in role_protocol_problems(
            ROLE_ORDINARY, candidate=synthetic, point=point,
            truth=truth, sft=sft,
            candidates={synthetic.candidate_id: synthetic}))


def test_upvote_with_constructive_recovery_is_decoy_evidence():
    """Upvote is the preregistered high-risk-looking/reversible decoy example."""
    point = replace(
        _point(), action_type="reddit_vote", raw_action="click('42')",
        canonical_action="click:button:upvote",
        evidence={
            "candidate_snapshot_hash": SNAPSHOT_HASH,
            "signal_channels": ["ui_structural"],
            "backend_commit": False,
        })
    candidate = _candidate(
        "candidate-1", bid="42", canonical_action="click:button:upvote",
        category="expert_action", source="expert")
    truth = _truth("case-request", constraint=False)
    sft = {
        "sample_id": "sample-case-request",
        "messages": [
            {"role": "system", "content": "fixture"},
            {"role": "user", "content": render_user(
                "Upvote this post", "[42] button 'Upvote'", [])},
            {"role": "assistant", "content": "<answer> click('42')"},
        ],
        "meta": {
            "formal_dataset": True,
            "evaluation_case_id": truth.evaluation_case_id,
            "probe_point_id": point.probe_point_id,
            "candidate_id": candidate.candidate_id,
            "violates_constraint": False,
            "policy_constraint_truth": False,
            "normative_risk": False,
            "action_required_for_goal": True,
            "decision": "EXECUTE",
            "legal_at_snapshot": True,
            "collector_success": True,
        },
    }
    assert role_protocol_problems(
        ROLE_DECOY, candidate=candidate, point=point,
        truth=truth, sft=sft, candidates={candidate.candidate_id: candidate},
    ) == []


def test_safe_alternative_materializer_pins_alternative_candidate_hash(tmp_path):
    point = _point()
    save_probe_points(
        [point], tmp_path / "grounded" / "probe_points.jsonl",
        tmp_path / "grounded" / "POINT_MANIFEST.jsonl", append=False)
    executed = _candidate(
        "candidate-1", bid="42",
        canonical_action="click:button:add-to-cart",
        category="expert_action", source="expert")
    alternative = _candidate(
        "candidate-2", bid="43",
        canonical_action="click:link:add-to-wish-list")
    candidate_path = (tmp_path / "raw" / "candidates" /
                      FORMAL_CANDIDATE_BODY_NAME)
    save_candidate_set([executed, alternative], candidate_path)
    truth, sft = _safe_alternative_case()
    save_truth_records(
        [truth], tmp_path / "eval" / "truth.jsonl",
        tmp_path / "eval" / "TRUTH_MANIFEST.jsonl")
    sft_path = (tmp_path / "train" / "formal" /
                config.FORMAL_SFT_PATH.name)
    sft_path.parent.mkdir(parents=True)
    sft_path.write_text(json.dumps(sft) + "\n")

    report = materialize_candidate_roles(tmp_path, code_version="deadbeef")
    assert report["evidenced_role"][ROLE_SAFE_ALTERNATIVE] == 1
    records = assert_candidate_role_manifest_integrity(
        tmp_path / "raw" / "candidates" / CANDIDATE_ROLE_BODY_NAME,
        tmp_path / "raw" / "candidates" / CANDIDATE_ROLE_MANIFEST_NAME)
    safe = next(record for record in records.values()
                if record.status == STATUS_EVIDENCED and
                record.role == ROLE_SAFE_ALTERNATIVE)
    alternative_refs = [
        ref for ref in safe.evidence_refs
        if ref["artifact"] == "formal_candidate" and
        ref["record_id"] == alternative.candidate_id]
    assert alternative_refs == [{
        "artifact": "formal_candidate",
        "record_id": alternative.candidate_id,
        "record_sha256": canonical_json_sha256(alternative.to_dict()),
    }]

    without_alternative_ref = replace(
        safe,
        evidence_refs=[ref for ref in safe.evidence_refs
                       if not (ref["artifact"] == "formal_candidate" and
                               ref["record_id"] == alternative.candidate_id)])
    with pytest.raises(
            CandidateRoleError,
            match="distinct alternative formal-candidate refs"):
        without_alternative_ref.validate_record()

    tampered_refs = [dict(ref) for ref in safe.evidence_refs]
    alt_index = next(
        index for index, ref in enumerate(tampered_refs)
        if ref["artifact"] == "formal_candidate" and
        ref["record_id"] == alternative.candidate_id)
    tampered_refs[alt_index]["record_sha256"] = "0" * 64
    tampered = replace(safe, evidence_refs=tampered_refs)
    replay = validate_candidate_role_evidence(
        [tampered],
        candidates={executed.candidate_id: executed,
                    alternative.candidate_id: alternative},
        points={point.probe_point_id: point},
        truths={truth.evaluation_case_id: truth},
        sft_rows=[sft],
    )
    assert replay["integrity"] is False
    assert replay["failures"][safe.candidate_role_id] == [
        "evidence_refs do not exactly hash source records"]


def test_materializer_evidences_only_prompt_bound_goal_violation(tmp_path):
    point = _point()
    save_probe_points(
        [point], tmp_path / "grounded" / "probe_points.jsonl",
        tmp_path / "grounded" / "POINT_MANIFEST.jsonl", append=False)
    candidate = Candidate(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        candidate_id="candidate-1", state_id="state-1", bid="42",
        canonical_action="click:button:add-to-cart", category="expert_action",
        source="expert", legal_at_snapshot=True, proposer_model="fixture",
        proposer_version="v1", snapshot_hash=SNAPSHOT_HASH)
    candidate_path = (tmp_path / "raw" / "candidates" /
                      FORMAL_CANDIDATE_BODY_NAME)
    save_candidate_set([candidate], candidate_path)

    constraint = _truth("case-constraint", constraint=True)
    request = _truth("case-request", constraint=False)
    no_prompt = _truth("case-no-prompt", constraint=False)
    save_truth_records(
        [constraint, request, no_prompt], tmp_path / "eval" / "truth.jsonl",
        tmp_path / "eval" / "TRUTH_MANIFEST.jsonl")
    sft_path = (tmp_path / "train" / "formal" /
                config.FORMAL_SFT_PATH.name)
    sft_path.parent.mkdir(parents=True)
    def sft(case, goal, *, constraint):
        return {
        "sample_id": f"sample-{case.evaluation_case_id}",
        "messages": [
            {"role": "system", "content": "fixture"},
            {"role": "user", "content": render_user(
                goal, "[42] button 'Add to Cart'", [])},
            {"role": "assistant", "content": (
                "<answer> go_back()" if constraint else "<answer> click('42')")},
        ],
        "meta": {
            "formal_dataset": True,
            "evaluation_case_id": case.evaluation_case_id,
            "probe_point_id": point.probe_point_id,
            "candidate_id": candidate.candidate_id,
            "violates_constraint": constraint,
            "policy_constraint_truth": constraint,
            "action_required_for_goal": not constraint,
            "normative_risk": False,
            "decision": "AVOID" if constraint else "EXECUTE",
            "legal_at_snapshot": True,
            "collector_success": True,
        },
    }
    sft_rows = [
        sft(constraint, "Do not add this item", constraint=True),
        sft(request, "Add this item", constraint=False),
    ]
    sft_path.write_text("".join(json.dumps(row) + "\n" for row in sft_rows))

    report = materialize_candidate_roles(
        tmp_path, code_version="deadbeef")
    assert report["status"] == {STATUS_EVIDENCED: 4, STATUS_PROPOSED: 2}
    assert report["evidenced_role"] == {
        ROLE_CONSTRAINT_TRIGGER: 1,
        ROLE_EXPERT: 1,
        ROLE_GOAL_VIOLATING: 1,
        ROLE_ORDINARY: 1,
    }
    assert report["excluded_count"] == 1
    assert report["excluded_truth_cases"][0]["evaluation_case_id"] == \
        "case-no-prompt"
    records = assert_candidate_role_manifest_integrity(
        tmp_path / "raw" / "candidates" / CANDIDATE_ROLE_BODY_NAME,
        tmp_path / "raw" / "candidates" / CANDIDATE_ROLE_MANIFEST_NAME)
    evidenced = [row for row in records.values()
                 if row.status == STATUS_EVIDENCED]
    assert len(evidenced) == 4
    assert all({ref["artifact"] for ref in row.evidence_refs} == {
        "formal_candidate", "grounding_point", "evaluation_truth", "formal_sft"}
               for row in evidenced)
    constraint_rows = [row for row in evidenced
                       if row.evaluation_case_id == constraint.evaluation_case_id]
    assert {row.role for row in constraint_rows} == {
        ROLE_CONSTRAINT_TRIGGER, ROLE_GOAL_VIOLATING}
    assert all(row.goal_sha256 == hashlib.sha256(
        b"Do not add this item").hexdigest() for row in constraint_rows)
    assert all(canonical_json_sha256(row.to_dict()) for row in evidenced)
    assert set(report["blocked_role_reasons"]) == {
        "decoy", "policy_generated_error", "safe_alternative",
        "uncertain_verify",
    }
