"""Formal DPO comes from legal/on-policy errors, never implicit label flips."""
from __future__ import annotations

import hashlib
import json

import pytest

from revact.cli import main
from revact.data.candidates import (CATEGORY_SAFE_ALTERNATIVE, SOURCE_A11Y,
                                    candidate_from_proposal, snapshot_sha256)
from revact.data.preferences import (
    PREFERENCE_NEGATIVE_SCHEMA_VERSION,
    PreferenceMaterializationError,
    PreferenceNegative,
    build_formal_dpo_pairs,
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


def test_legal_candidate_materializes_a_formal_deployment_pair():
    candidate = _candidate()
    rows = build_formal_dpo_pairs(
        [_sft()], {candidate.candidate_id: candidate}, {"error-1": _negative()})
    assert len(rows) == 1
    meta = rows[0]["meta"]
    assert meta["negative_source"] == "legal_candidate"
    assert meta["candidate_id"] == "source-candidate-1"
    assert meta["negative_candidate_id"] == candidate.candidate_id
    assert meta["negative_candidate_snapshot_hash"] == candidate.snapshot_hash
    report = validate_rows(rows, formal=True)
    assert report["n_problems"] == 0
    assert report["deployment_negative_ratio"] == 1.0


def test_on_policy_negative_requires_model_version():
    with pytest.raises(PreferenceMaterializationError, match="policy_model_version"):
        _negative("on_policy").validate()
    candidate = _candidate()
    record = _negative("on_policy", policy_model_version="iris-sft-step100")
    rows = build_formal_dpo_pairs(
        [_sft()], {candidate.candidate_id: candidate}, {record.error_id: record})
    assert rows[0]["meta"]["policy_model_version"] == "iris-sft-step100"


def test_rejected_action_must_match_legal_snapshot_candidate():
    candidate = _candidate()
    bad = _negative(rejected_completion=_completion("click('99')", "AVOID"))
    with pytest.raises(PreferenceMaterializationError, match="primitive/bid"):
        build_formal_dpo_pairs(
            [_sft()], {candidate.candidate_id: candidate}, {bad.error_id: bad})


def test_materialize_cli_fails_on_empty_formal_source(tmp_path):
    negatives = tmp_path / "negatives.jsonl"
    negatives.write_text(json.dumps(_negative().to_dict()) + "\n")
    assert main(["materialize-dpo", "--negatives", str(negatives),
                 "--data-root", str(tmp_path)]) == 1
    assert not (tmp_path / "train" / "formal" /
                "iris_dpo_point_v1.jsonl").exists()
