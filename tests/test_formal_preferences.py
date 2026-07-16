"""Formal DPO comes from legal/on-policy errors, never implicit label flips."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from revact import config
from revact.cli import main
from revact.data.candidates import (CATEGORY_SAFE_ALTERNATIVE, SOURCE_A11Y,
                                    candidate_from_proposal, snapshot_sha256)
from revact.data.preferences import (
    PREFERENCE_NEGATIVE_SCHEMA_VERSION,
    PreferenceMaterializationError,
    PreferenceNegative,
    build_formal_dpo_pairs,
    build_reviewed_legal_counterfactual_negatives,
)
from revact.train.dpo import validate_rows


OBS = "RootWebArea 'Item'\n[12] button 'Add'\n[13] link 'Back'"


def _completion(answer: str, decision: str) -> str:
    return (
        "<think>\n<observation> target visible\n<reasoning> fixture\n"
        "<prediction> measured signal changed from {\"x\": 0} to {\"x\": 1}\n"
        "<rev_check> point-1 measured fixture\n"
        "<reversibility> RECOVERED\n<undo> click('9'); undo_cost_steps=1\n"
        f"<decision> {decision} risk=0.1\n</think>\n<answer> {answer}")


def _sft() -> dict:
    return {
        "sample_id": "sample-1",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content":
             f"<goal>\nadd\n\n<history>\n(none)\n\n<observation>\n{OBS}"},
            {"role": "assistant", "content": _completion("click('12')", "EXECUTE")},
        ],
        "meta": {
            "formal_dataset": True, "is_mock": False,
            "collector_success": True, "probe_point_id": "point-1",
            "state_id": "state-1", "action_instance_id": "action-1",
            "candidate_id": "source-candidate-1",
            "prediction_source": "probe_transition", "undo_source": "probe_point_id",
            "history_source": "trajectory", "effect_status": "CHANGED",
            "recovery_status": "RECOVERED", "normative_risk": False,
            "decision": "EXECUTE", "risky_raw_action": "click('12')",
            "risky_action": {"name": "click", "bid": "12"},
            "post_signal_diff": {"pre_signal": {"x": 0},
                                 "post_signal": {"x": 1}},
            "undo_actions": ["click('9')"], "undo_cost_steps": 1,
            "candidate_snapshot_hash": snapshot_sha256(OBS),
            "policy_input_observation_hash": hashlib.sha256(
                OBS.encode("utf-8")).hexdigest(),
            "evidence": {"measurement": "fixture"},
        },
    }


def _candidate():
    return candidate_from_proposal(
        {"bid": "13", "canonical_action": "click:link:back",
         "category": CATEGORY_SAFE_ALTERNATIVE},
        state_id="state-1", axtree_txt=OBS, source=SOURCE_A11Y,
        proposer_model="fixture", proposer_version="v1")


def _negative(source: str = "legal_candidate", **updates) -> PreferenceNegative:
    values = dict(
        schema_version=PREFERENCE_NEGATIVE_SCHEMA_VERSION, error_id="error-1",
        sample_id="sample-1", negative_source=source,
        candidate_id=_candidate().candidate_id,
        rejected_completion=_completion("click('13')", "AVOID"),
        preference_source="human_review", policy_model_version="",
        reviewer="reviewer-1", timestamp="2026-07-13T00:00:00+00:00",
        evidence={"reason": "fails the requested goal"})
    values.update(updates)
    return PreferenceNegative(**values)


def _trace(*, candidate_id=None, completion=None, legal=True,
           action_kind="legal", errors=None):
    source = _sft()
    completion = completion or _completion("click('13')", "AVOID")
    return SimpleNamespace(
        trace_id="trace-1", source_sample_id="sample-1",
        input_messages=source["messages"][:-1],
        input_messages_sha256="a" * 64,
        raw_completion=completion,
        raw_completion_sha256=hashlib.sha256(completion.encode()).hexdigest(),
        candidate_id=candidate_id, error_types=errors or ["wrong_decision"],
        eligible_as_negative=True, action_kind=action_kind,
        action_legal=legal, parsed_action={"name": "click", "args": ["13"],
                                           "bid": "13"},
        policy_provenance={"provider": "fixture", "model": "model-v1"},
        policy_provenance_sha256="b" * 64, model_returned="model-v1",
        rollout_run_id="rollout-1", evaluation_case_id="case-1",
        probe_point_id="point-1", truth_source="fixture-policy",
    )


def _on_policy_negative(trace):
    return _negative(
        "on_policy", candidate_id=trace.candidate_id or "",
        rejected_completion=trace.raw_completion,
        policy_model_version="fixture:model-v1",
        evidence={
            "trace_id": trace.trace_id,
            "input_messages_sha256": trace.input_messages_sha256,
            "raw_completion_sha256": trace.raw_completion_sha256,
            "error_types": list(trace.error_types),
            "action_legal": trace.action_legal,
            "action_kind": trace.action_kind,
            "completion_construction": "raw_model_output",
            "observed_policy_error": True,
        })


def test_legal_candidate_materializes_a_formal_deployment_pair():
    candidate = _candidate()
    rows = build_formal_dpo_pairs(
        [_sft()], {candidate.candidate_id: candidate}, {"error-1": _negative()})
    assert len(rows) == 1
    meta = rows[0]["meta"]
    assert meta["negative_source"] == "legal_candidate"
    assert meta["pair_type"] == "reviewed_legal_counterfactual"
    assert meta["candidate_id"] == "source-candidate-1"
    assert meta["negative_candidate_id"] == candidate.candidate_id
    assert meta["negative_candidate_snapshot_hash"] == candidate.snapshot_hash
    report = validate_rows(rows, formal=True)
    assert report["n_problems"] == 0
    assert report["deployment_negative_ratio"] == 1.0


def test_on_policy_negative_requires_model_version():
    with pytest.raises(PreferenceMaterializationError, match="policy_model_version"):
        _negative("on_policy").validate()
    with pytest.raises(PreferenceMaterializationError, match="trace evidence"):
        _negative(
            "on_policy", policy_model_version="iris-sft-step100").validate()
    candidate = _candidate()
    trace = _trace(candidate_id=candidate.candidate_id)
    record = _on_policy_negative(trace)
    rows = build_formal_dpo_pairs(
        [_sft()], {candidate.candidate_id: candidate}, {record.error_id: record},
        on_policy_traces={trace.trace_id: trace})
    assert rows[0]["meta"]["policy_model_version"] == "fixture:model-v1"
    assert rows[0]["meta"]["pair_type"] == "observed_policy_error"
    assert rows[0]["meta"]["on_policy_trace_verified"] is True


def test_trace_backed_illegal_on_policy_action_never_claims_candidate():
    completion = _completion("click('99')", "AVOID")
    trace = _trace(
        candidate_id=None, completion=completion, legal=False,
        action_kind="illegal", errors=["illegal_action"])
    trace.parsed_action = {"name": "click", "args": ["99"], "bid": "99"}
    record = _on_policy_negative(trace)
    rows = build_formal_dpo_pairs(
        [_sft()], {_candidate().candidate_id: _candidate()},
        {record.error_id: record}, on_policy_traces={trace.trace_id: trace})
    meta = rows[0]["meta"]
    assert meta["negative_candidate_id"] == ""
    assert meta["legal_at_snapshot"] is False
    assert meta["on_policy_action_kind"] == "illegal"
    assert validate_rows(rows, formal=True)["n_problems"] == 0


def test_rejected_action_must_match_legal_snapshot_candidate():
    candidate = _candidate()
    bad = _negative(rejected_completion=_completion("click('99')", "AVOID"))
    with pytest.raises(PreferenceMaterializationError, match="primitive/bid"):
        build_formal_dpo_pairs(
            [_sft()], {candidate.candidate_id: candidate}, {bad.error_id: bad})


def test_legal_counterfactual_authoring_is_not_mislabeled_observed_error():
    source = _sft()
    source["meta"].update({
        "variant": "constraint", "decision": "AVOID",
        "violates_constraint": True, "policy_constraint_truth": True,
        "action_legal": True, "legal_at_snapshot": True,
        "candidate_id": _candidate().candidate_id,
        "evaluation_case_id": "case-1", "normative_policy_id": "policy-1",
        "normative_policy_version": "v1",
    })
    source["messages"][-1]["content"] = _completion("go_back()", "AVOID")
    rejected = _completion("click('13')", "EXECUTE")
    ablation = [{
        "pair_id": "sample-1__goal_violation", "rejected": rejected,
        "meta": {"source_sample_id": "sample-1",
                 "pair_type": "goal_violation",
                 "negative_source": "synthetic_flip"},
    }]
    records = build_reviewed_legal_counterfactual_negatives(
        [source], {_candidate().candidate_id: _candidate()}, ablation,
        reviewer="reviewer-1", timestamp="2026-07-14T00:00:00+00:00")
    assert len(records) == 1
    record = records[0]
    assert record.negative_source == "legal_candidate"
    assert record.evidence["observed_policy_error"] is False
    assert record.evidence["completion_construction"] == "counterfactual_template"
    pair = build_formal_dpo_pairs(
        [source], {_candidate().candidate_id: _candidate()},
        {record.error_id: record})[0]
    assert pair["meta"]["pair_type"] == "reviewed_legal_counterfactual"


def test_materialize_cli_fails_on_empty_formal_source(tmp_path):
    negatives = tmp_path / "negatives.jsonl"
    negatives.write_text(json.dumps(_negative().to_dict()) + "\n")
    assert main(["materialize-dpo", "--negatives", str(negatives),
                 "--data-root", str(tmp_path)]) == 1
    assert not (tmp_path / "train" / "formal" /
                config.FORMAL_DPO_PATH.name).exists()
