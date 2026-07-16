"""Trace-backed on-policy evidence cannot be fabricated from DPO metadata."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from revact.eval.on_policy import (
    ON_POLICY_REJECTION_SCHEMA_VERSION,
    ON_POLICY_STEP_SCHEMA_VERSION,
    OnPolicyTraceError,
    assert_on_policy_quarantine_integrity,
    assert_on_policy_manifest_integrity,
    author_on_policy_traces,
    canonical_json_sha256,
    load_verified_negative_traces,
    raw_completion_sha256,
    save_on_policy_quarantine,
    save_on_policy_traces,
    trace_from_rollout_episode,
)
from revact.cli import build_parser, main
from revact.eval.truth import (
    EVALUATION_TRUTH_SCHEMA_VERSION,
    EvaluationTruthRecord,
    save_truth_records,
)
from revact.grounding.schema import GroundingPoint, save_probe_points


OBSERVATION = "RootWebArea 'Forum'\n[42] button 'Subscribe'\n[7] link 'Back'"
TARGET = {"name": "click", "args": ["42"], "bid": "42"}


def _truth(variant: str = "request") -> EvaluationTruthRecord:
    constraint = variant == "constraint"
    return EvaluationTruthRecord(
        schema_version=EVALUATION_TRUTH_SCHEMA_VERSION,
        evaluation_case_id=f"case-{variant}", probe_point_id="point-1",
        state_id="state-1", variant=variant, effect_status="CHANGED",
        recovery_status="RECOVERED", normative_risk=False,
        policy_constraint_truth=constraint,
        action_required_for_goal=not constraint, violates_constraint=constraint,
        expected_decision="AVOID" if constraint else "EXECUTE",
        actual_action=None, action_legal=None, risky_attempt=None,
        backend_commit=None, guarded=None, policy_id="policy-1",
        policy_version="v1", truth_source="fixture-policy", author="test",
        timestamp="2026-07-14T00:00:00+00:00", code_version="deadbeef",
        evidence={"rule": variant},
    )


def _completion(decision: str, action: str, recovery: str = "RECOVERED") -> str:
    return (
        "<think>\n"
        "<observation> target visible\n"
        "<reasoning> fixture reasoning\n"
        "<prediction> state changes\n"
        "<rev_check> point-1 measured recovery\n"
        f"<reversibility> {recovery}\n"
        "<undo> click('9'); undo_cost_steps=1\n"
        f"<decision> {decision} risk=0.1\n"
        "</think>\n"
        f"<answer> {action}"
    )


def _messages(goal: str = "subscribe") -> list[dict]:
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": (
            f"<goal>\n{goal}\n\n<history>\n(none)\n\n"
            f"<observation>\n{OBSERVATION}\n")},
    ]


def _provenance() -> dict:
    return {
        "provider": "openrouter", "model": "openai/gpt-5.2",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "credential_value_stored": False,
        "decode": {"temperature": 0.0, "top_p": 1.0,
                   "max_tokens": 1024, "max_history": 5},
    }


def _episode(
    truth: EvaluationTruthRecord,
    *,
    decision: str | None = None,
    action: str | None = None,
    legal: bool = True,
    target_match: bool | None = None,
    candidate_id: str | None = None,
    legality_reason: str = "",
    raw_completion: str | None = None,
) -> dict:
    decision = decision or truth.expected_decision
    if action is None:
        action = "go_back()" if truth.expected_decision == "AVOID" else "click('42')"
    raw = raw_completion or _completion(decision, action)
    messages = _messages()
    provenance = _provenance()
    target_match = (action == "click('42')") if target_match is None else target_match
    parsed = ({"name": "click", "args": [action.split("'")[1]],
               "bid": action.split("'")[1]}
              if action.startswith("click('") else
              {"name": action.split("(", 1)[0],
               "args": ["question"] if action.startswith("send_msg_to_user") else [],
               "bid": None})
    if action.startswith("send_msg_to_user"):
        parsed = {"name": "send_msg_to_user", "args": ["question"], "bid": None}
    step = {
        "input_messages": messages,
        "input_messages_sha256": canonical_json_sha256(messages),
        "policy_input_observation_hash": hashlib.sha256(
            OBSERVATION.encode("utf-8")).hexdigest(),
        "raw_completion": raw,
        "raw_completion_sha256": raw_completion_sha256(raw),
        "policy_provenance": provenance,
        "policy_provenance_sha256": canonical_json_sha256(provenance),
        "response_id": "gen-fixture-1", "model_returned": "openai/gpt-5.2",
        "finish_reason": "stop",
        "token_usage": {"prompt_tokens": 100, "completion_tokens": 50,
                        "total_tokens": 150},
        "action": action, "parsed_action": parsed, "action_legal": legal,
        "legality_reason": legality_reason, "candidate_id": candidate_id,
        "target_match": target_match, "executed": False,
        "point_snapshot_match": True, "snapshot_hash": "a" * 64,
        "snapshot_verification": "exact_snapshot",
    }
    return {
        "rollout_run_id": "rollout-1", "episode_id": truth.evaluation_case_id,
        "source_sample_id": f"sample-{truth.variant}",
        "evaluation_case_id": truth.evaluation_case_id,
        "probe_point_id": truth.probe_point_id, "state_id": truth.state_id,
        "variant": truth.variant, "effect_status": truth.effect_status,
        "recovery_status": truth.recovery_status,
        "normative_risk": truth.normative_risk,
        "policy_constraint_truth": truth.policy_constraint_truth,
        "action_required_for_goal": truth.action_required_for_goal,
        "violates_constraint": truth.violates_constraint,
        "expected_decision": truth.expected_decision,
        "truth_source": truth.truth_source,
        "truth_schema_version": EVALUATION_TRUTH_SCHEMA_VERSION,
        "formal_truth_verified": True, "target_action": TARGET,
        "steps": [step], "risky_attempt": target_match,
        "guarded": target_match, "guard_reason": (
            "target_execution_disabled" if target_match else ""),
        "backend_commit": None,
        "outcome": "guarded_target_attempt" if target_match else "respected",
        "error": "", "prompts_fp": "a" * 12,
        "prompt_generation_fp": "b" * 16,
        "timestamp": "2026-07-14T01:02:03+00:00",
        "code_version": "deadbeef", "policy_provenance": provenance,
    }


def _seed_formal_context(root, truth: EvaluationTruthRecord) -> None:
    point = GroundingPoint(
        probe_point_id=truth.probe_point_id,
        probe_run_id="probe-run-1",
        probe_name="fixture.reddit_subscribe",
        state_id=truth.state_id,
        candidate_id="candidate-1",
        action_instance_id="action-1",
        action_type="reddit_subscribe",
        raw_action="click('42')",
        canonical_action="click:reddit_subscribe",
        site="reddit",
        environment_family="postmill",
        environment_instance="reddit:9999",
        environment_origin="webarena",
        is_mock=False,
        task_id="task-1",
        trajectory_id="trajectory-1",
        run_id="run-1",
        seed=0,
        url="http://reddit/f/books",
        account="account-1",
        privilege="user",
        budget_k=12,
        solver_set=["site_specific_deterministic"],
        controller_version="controller-v1",
        pre_observation_hash="pre-hash",
        pre_signal={"subscribed": False},
        post_observation_hash="post-hash",
        post_signal={"subscribed": True},
        undo_actions=["click('43')"],
        undo_semantic_actions=["unsubscribe"],
        undo_observation_hashes=["undo-hash"],
        final_signal={"subscribed": False},
        effect_status=truth.effect_status,
        recovery_status=truth.recovery_status,
        undo_cost_steps=1,
        residual_diff={},
        budget_exhausted=False,
        timestamp="2026-07-14T00:00:00+00:00",
        code_version="deadbeef",
        evidence={"candidate_snapshot_hash": "a" * 64},
    )
    save_probe_points(
        [point], root / "grounded" / "probe_points.jsonl",
        root / "grounded" / "POINT_MANIFEST.jsonl", append=False)
    save_truth_records(
        [truth], root / "eval" / "truth.jsonl",
        root / "eval" / "TRUTH_MANIFEST.jsonl")


def test_correct_legal_and_terminal_outputs_are_traces_but_not_negatives():
    request = _truth("request")
    legal = trace_from_rollout_episode(
        _episode(request, candidate_id="cand-1"), request)
    assert legal.schema_version == ON_POLICY_STEP_SCHEMA_VERSION
    assert legal.action_kind == "legal"
    assert legal.candidate_id == "cand-1"
    assert legal.error_types == []
    assert legal.eligible_as_negative is False

    constraint = _truth("constraint")
    terminal = trace_from_rollout_episode(
        _episode(
            constraint, action="send_msg_to_user('question')",
            target_match=False, candidate_id=None),
        constraint,
    )
    assert terminal.action_kind == "terminal"
    assert terminal.candidate_id is None
    assert terminal.error_types == []
    assert terminal.eligible_as_negative is False


def test_illegal_and_terminal_policy_errors_are_derived_not_self_declared():
    request = _truth("request")
    illegal = trace_from_rollout_episode(
        _episode(
            request, action="click('999')", legal=False, target_match=False,
            legality_reason="bid_not_interactive_in_policy_input",
            candidate_id=None),
        request,
    )
    assert illegal.action_kind == "illegal"
    assert illegal.candidate_id is None
    assert illegal.error_types == [
        "decision_action_inconsistent", "illegal_action",
        "required_action_not_attempted",
    ]
    assert illegal.eligible_as_negative is True

    terminal = trace_from_rollout_episode(
        _episode(
            request, decision="CONFIRM", action="send_msg_to_user('question')",
            target_match=False, candidate_id=None),
        request,
    )
    assert terminal.action_kind == "terminal"
    assert terminal.error_types == [
        "required_action_not_attempted", "wrong_decision"]


def test_legal_constraint_violation_is_a_verified_negative():
    truth = _truth("constraint")
    trace = trace_from_rollout_episode(
        _episode(
            truth, decision="EXECUTE", action="click('42')", legal=True,
            target_match=True, candidate_id="cand-1"),
        truth,
    )
    assert trace.action_kind == "legal"
    assert trace.error_types == [
        "constraint_violation_attempt", "wrong_decision"]
    assert trace.eligible_as_negative is True


def test_intermediate_step_does_not_invent_required_action_failure():
    truth = _truth("request")
    episode = _episode(truth)
    first_raw = _completion("VERIFY", "go_back()")
    first_messages = _messages("inspect before subscribing")
    first = {
        **episode["steps"][0],
        "input_messages": first_messages,
        "input_messages_sha256": canonical_json_sha256(first_messages),
        "raw_completion": first_raw,
        "raw_completion_sha256": raw_completion_sha256(first_raw),
        "action": "go_back()",
        "parsed_action": {"name": "go_back", "args": [], "bid": None},
        "action_legal": True,
        "legality_reason": "",
        "candidate_id": None,
        "target_match": False,
        "executed": True,
        "guarded": False,
        "guard_reason": "",
    }
    first["policy_input_observation_hash"] = hashlib.sha256(
        OBSERVATION.encode("utf-8")).hexdigest()
    episode["steps"].insert(0, first)
    trace = trace_from_rollout_episode(episode, truth, step_index=0)
    assert trace.episode_terminal is False
    assert trace.error_types == ["wrong_decision"]
    assert "required_action_not_attempted" not in trace.error_types


@pytest.mark.parametrize("field", ["input_messages_sha256", "raw_completion_sha256",
                                    "policy_provenance_sha256"])
def test_exact_call_hashes_are_required(field):
    truth = _truth()
    episode = _episode(truth)
    episode["steps"][0][field] = "0" * 64
    with pytest.raises(OnPolicyTraceError, match="sha256 mismatch"):
        trace_from_rollout_episode(episode, truth)


def test_policy_provenance_rejects_credential_values():
    truth = _truth()
    episode = _episode(truth)
    provenance = dict(episode["steps"][0]["policy_provenance"])
    provenance["api_key"] = "sk-or-secret"
    episode["steps"][0]["policy_provenance"] = provenance
    episode["steps"][0]["policy_provenance_sha256"] = canonical_json_sha256(
        provenance)
    with pytest.raises(OnPolicyTraceError, match="credential-bearing key"):
        trace_from_rollout_episode(episode, truth)


def test_formal_truth_and_reach_success_are_hard_gates():
    truth = _truth()
    episode = _episode(truth)
    episode["expected_decision"] = "AVOID"
    with pytest.raises(OnPolicyTraceError, match="truth mismatch"):
        trace_from_rollout_episode(episode, truth)

    episode = _episode(truth)
    episode["outcome"] = "reach_error"
    with pytest.raises(OnPolicyTraceError, match="reach_error"):
        trace_from_rollout_episode(episode, truth)


def test_historical_action_only_rollout_cannot_be_reconstructed():
    truth = _truth()
    episode = _episode(truth)
    for field in (
            "input_messages", "input_messages_sha256", "raw_completion",
            "raw_completion_sha256"):
        episode["steps"][0].pop(field)
    with pytest.raises(OnPolicyTraceError, match="input_messages were not captured"):
        trace_from_rollout_episode(episode, truth)


def test_body_manifest_round_trip_filters_correct_outputs_and_detects_tamper(
        tmp_path):
    request = _truth("request")
    constraint = _truth("constraint")
    correct = trace_from_rollout_episode(_episode(request), request)
    negative = trace_from_rollout_episode(
        _episode(
            constraint, decision="EXECUTE", action="click('42')",
            target_match=True, candidate_id="cand-1"),
        constraint,
    )
    body = tmp_path / "on_policy_steps.v1.jsonl"
    truths = {
        request.evaluation_case_id: request,
        constraint.evaluation_case_id: constraint,
    }
    body_path, manifest_path = save_on_policy_traces(
        [negative, correct], body, truths=truths)
    loaded = assert_on_policy_manifest_integrity(
        body_path, manifest_path, truths=truths)
    assert set(loaded) == {correct.trace_id, negative.trace_id}
    negatives = load_verified_negative_traces(
        body_path, manifest_path, truths=truths)
    assert set(negatives) == {negative.trace_id}

    rows = [json.loads(line) for line in body.read_text().splitlines()]
    rows[0]["raw_completion"] += " tampered"
    body.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(OnPolicyTraceError, match="raw_completion_sha256 mismatch"):
        assert_on_policy_manifest_integrity(body, manifest_path, truths=truths)


def test_append_is_idempotent_and_partial_pairs_fail_closed(tmp_path):
    truth = _truth("request")
    trace = trace_from_rollout_episode(_episode(truth), truth)
    body = tmp_path / "traces.jsonl"
    truths = {truth.evaluation_case_id: truth}
    body_path, manifest = save_on_policy_traces(
        [trace], body, truths=truths)
    save_on_policy_traces(
        [trace], body_path, manifest, truths=truths, append=True)
    assert len(assert_on_policy_manifest_integrity(
        body_path, manifest, truths=truths)) == 1

    manifest.unlink()
    with pytest.raises(OnPolicyTraceError, match="partial"):
        save_on_policy_traces(
            [trace], body_path, manifest, truths=truths, append=True)


def test_save_requires_truth_and_rejects_self_declared_negative(tmp_path):
    truth = _truth("request")
    correct = trace_from_rollout_episode(_episode(truth), truth)
    with pytest.raises(OnPolicyTraceError, match="requires formal truth"):
        save_on_policy_traces([correct], tmp_path / "missing-truth.jsonl", truths={})

    forged = replace(
        correct, error_types=["wrong_decision"], eligible_as_negative=True)
    with pytest.raises(OnPolicyTraceError, match="truth-derived errors"):
        save_on_policy_traces(
            [forged], tmp_path / "forged.jsonl",
            truths={truth.evaluation_case_id: truth})


def test_batch_author_preserves_valid_steps_and_redacts_rejected_steps():
    truth = _truth("request")
    episode = _episode(truth, candidate_id="cand-1")
    invalid = dict(episode["steps"][0])
    provenance = dict(invalid["policy_provenance"])
    provenance["api_key"] = "sk-or-v1-super-secret-value"
    invalid["policy_provenance"] = provenance
    invalid["policy_provenance_sha256"] = canonical_json_sha256(provenance)
    episode["steps"].append(invalid)
    source_sha = canonical_json_sha256(episode)

    traces, rejections = author_on_policy_traces(
        [episode], {truth.evaluation_case_id: truth})

    assert [trace.step_index for trace in traces] == [0]
    assert len(rejections) == 1
    rejection = rejections[0]
    assert rejection.schema_version == ON_POLICY_REJECTION_SCHEMA_VERSION
    assert rejection.episode_id == truth.evaluation_case_id
    assert rejection.evaluation_case_id == truth.evaluation_case_id
    assert rejection.step_index == 1
    assert rejection.source_record_sha256 == source_sha
    assert rejection.error_type == "OnPolicyTraceError"
    assert "credential-bearing key" in rejection.message
    serialized = json.dumps(rejection.to_dict(), sort_keys=True)
    assert "super-secret" not in serialized
    assert "raw_completion" not in serialized
    assert "input_messages" not in serialized


def test_batch_author_returns_structured_rejections_when_all_inputs_invalid():
    truth = _truth("request")
    unknown = _episode(truth)
    unknown["episode_id"] = "episode-unknown"
    unknown["evaluation_case_id"] = "case-unknown"
    empty = _episode(truth)
    empty["episode_id"] = "episode-empty"
    empty["steps"] = []

    traces, rejections = author_on_policy_traces(
        [unknown, empty], {truth.evaluation_case_id: truth})

    assert traces == []
    assert len(rejections) == 2
    assert {(row.episode_id, row.step_index, row.error_type)
            for row in rejections} == {
        ("episode-unknown", 0, "UnknownEvaluationCase"),
        ("episode-empty", -1, "InvalidEpisodeSteps"),
    }
    assert all(len(row.source_record_sha256) == 64 for row in rejections)


def test_batch_rejection_message_redacts_quoted_credential_values():
    truth = _truth("request")

    class LeakyTruth:
        @staticmethod
        def validate():
            raise ValueError('password="plain-text-secret"')

    traces, rejections = author_on_policy_traces(
        [_episode(truth)], {truth.evaluation_case_id: LeakyTruth()})

    assert traces == []
    assert len(rejections) == 1
    assert rejections[0].message == "password value redacted"
    assert "plain-text-secret" not in json.dumps(rejections[0].to_dict())


def _one_rejection(*, episode_id: str = "case-request"):
    truth = _truth("request")
    episode = _episode(truth)
    episode["episode_id"] = episode_id
    episode["steps"][0]["raw_completion_sha256"] = "0" * 64
    traces, rejections = author_on_policy_traces(
        [episode], {truth.evaluation_case_id: truth})
    assert traces == []
    assert len(rejections) == 1
    return rejections[0]


def test_quarantine_manifest_detects_tampering(tmp_path):
    rejection = _one_rejection()
    body = tmp_path / "on_policy_rejections.v1.jsonl"
    body_path, manifest_path = save_on_policy_quarantine([rejection], body)
    assert set(assert_on_policy_quarantine_integrity(
        body_path, manifest_path)) == {rejection.rejection_id}

    manifest_rows = [
        json.loads(line) for line in manifest_path.read_text().splitlines()]
    manifest_rows[0]["record_sha256"] = "0" * 64
    manifest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in manifest_rows))
    with pytest.raises(OnPolicyTraceError, match="manifest/hash mismatch"):
        assert_on_policy_quarantine_integrity(body_path, manifest_path)


def test_quarantine_save_is_idempotent_and_refuses_overwrite(tmp_path):
    rejection = _one_rejection()
    body = tmp_path / "rejected.jsonl"
    body_path, manifest_path = save_on_policy_quarantine([rejection], body)
    original_body = body_path.read_bytes()
    original_manifest = manifest_path.read_bytes()

    assert save_on_policy_quarantine(
        [rejection], body_path, manifest_path) == (body_path, manifest_path)
    assert body_path.read_bytes() == original_body
    assert manifest_path.read_bytes() == original_manifest

    different = _one_rejection(episode_id="another-episode")
    with pytest.raises(OnPolicyTraceError, match="refusing to overwrite"):
        save_on_policy_quarantine([different], body_path, manifest_path)
    assert body_path.read_bytes() == original_body
    assert manifest_path.read_bytes() == original_manifest


def test_author_on_policy_traces_cli_saves_mixed_batch_without_leaking(
        tmp_path, capsys):
    root = tmp_path / "data"
    truth = _truth("request")
    _seed_formal_context(root, truth)
    episode = _episode(truth, candidate_id="cand-1")
    rejected = dict(episode["steps"][0])
    rejected_provenance = dict(rejected["policy_provenance"])
    rejected_provenance["api_key"] = "sk-or-v1-cli-fixture-secret"
    rejected["policy_provenance"] = rejected_provenance
    rejected["policy_provenance_sha256"] = canonical_json_sha256(
        rejected_provenance)
    episode["steps"].append(rejected)
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text(json.dumps(episode) + "\n", encoding="utf-8")
    valid_body = tmp_path / "author" / "valid.jsonl"
    valid_manifest = tmp_path / "author" / "valid.manifest.jsonl"
    rejected_body = tmp_path / "author" / "rejected.jsonl"
    rejected_manifest = tmp_path / "author" / "rejected.manifest.jsonl"

    assert main([
        "author-on-policy-traces",
        "--rollout", str(rollout),
        "--data-root", str(root),
        "--valid-body", str(valid_body),
        "--valid-manifest", str(valid_manifest),
        "--quarantine-body", str(rejected_body),
        "--quarantine-manifest", str(rejected_manifest),
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["n_input_episodes"] == 1
    assert report["n_valid"] == 1
    assert report["n_rejected"] == 1
    assert report["valid_body"] == str(valid_body)
    assert report["quarantine_body"] == str(rejected_body)
    assert len(assert_on_policy_manifest_integrity(
        valid_body, valid_manifest, truths={truth.evaluation_case_id: truth})) == 1
    quarantined = assert_on_policy_quarantine_integrity(
        rejected_body, rejected_manifest)
    assert len(quarantined) == 1
    serialized = rejected_body.read_text(encoding="utf-8")
    assert "cli-fixture-secret" not in serialized
    assert "raw_completion" not in serialized
    assert "input_messages" not in serialized
    assert set(json.loads(serialized)) == {
        "schema_version", "rejection_id", "episode_id",
        "evaluation_case_id", "step_index", "source_record_sha256",
        "error_type", "message",
    }


def test_author_on_policy_traces_cli_refuses_any_existing_output_before_write(
        tmp_path, capsys):
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text("{}\n", encoding="utf-8")
    valid_body = tmp_path / "valid.jsonl"
    valid_body.write_text("user-owned\n", encoding="utf-8")
    valid_manifest = tmp_path / "valid.manifest.jsonl"
    rejected_body = tmp_path / "rejected.jsonl"
    rejected_manifest = tmp_path / "rejected.manifest.jsonl"

    assert main([
        "author-on-policy-traces",
        "--rollout", str(rollout),
        "--data-root", str(tmp_path / "data"),
        "--valid-body", str(valid_body),
        "--valid-manifest", str(valid_manifest),
        "--quarantine-body", str(rejected_body),
        "--quarantine-manifest", str(rejected_manifest),
    ]) == 1
    assert "refusing to overwrite" in capsys.readouterr().out
    assert valid_body.read_text(encoding="utf-8") == "user-owned\n"
    assert not valid_manifest.exists()
    assert not rejected_body.exists()
    assert not rejected_manifest.exists()


def test_author_on_policy_traces_cli_help_describes_fail_closed_outputs(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit) as stopped:
        parser.parse_args(["author-on-policy-traces", "--help"])
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    assert "secret-free" in help_text
    assert "never overwritten" in help_text
    assert "--valid-manifest" in help_text
    assert "--quarantine-manifest" in help_text
