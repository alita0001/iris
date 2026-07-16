"""Regression tests for deployment-shaped episode history evidence."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from revact import config, prompts
from revact.cli import build_parser
from revact.data.collect import collect_trajectory
from revact.data.episode_traces import (
    EPISODE_TRACE_SCHEMA_VERSION,
    MESSAGE_TOPOLOGY,
    POLICY_BUILDER_ID,
    EpisodeObservation,
    EpisodeStep,
    EpisodeTraceError,
    StatelessEpisodeTrace,
    _point_snapshot_hash,
    assert_episode_manifest_integrity,
    audit_episode_trace_evidence,
    canonical_json_sha256,
    episode_trace_id_for,
    episode_trace_paths,
    import_episode_trace_file,
    observation_sha256,
    save_episode_traces,
)
from revact.envs.obs_utils import history_entry
from revact.envs.harness import RevActEnv
from revact.envs.mock_env import MockShoppingEnv
from revact.eval.truth import (EVALUATION_TRUTH_SCHEMA_VERSION,
                               EvaluationTruthRecord, save_truth_records)
from revact.grounding.schema import (EFFECT_CHANGED, RECOVERY_RECOVERED,
                                     GroundingPoint, save_probe_points)


def _full_iris(action: str) -> str:
    return (
        "<think>\n"
        "<observation>the selected control is visible\n"
        "<reasoning>the goal calls for this step\n"
        "<prediction>the observed page will update\n"
        "<rev_check>point evidence is consulted when state changes\n"
        "<reversibility>RECOVERED\n"
        "<undo>use the recorded recovery action\n"
        "<decision>EXECUTE risk=0.1\n"
        "</think>\n"
        f"<answer> {action}"
    )


def _observation(index: int, signal: int) -> EpisodeObservation:
    text = (
        f"RootWebArea 'step {index}'\n"
        f"  [{index}] button 'Do step {index}'"
    )
    return EpisodeObservation(
        url=f"http://example.test/work/{index}",
        title=f"step {index}",
        axtree_txt=text,
        axtree_sha256=observation_sha256(text),
        persistent_signal={"counter": signal},
    )


def _truth(point: GroundingPoint, case_id: str = "case-final") \
        -> EvaluationTruthRecord:
    truth = EvaluationTruthRecord(
        schema_version=EVALUATION_TRUTH_SCHEMA_VERSION,
        evaluation_case_id=case_id,
        probe_point_id=point.probe_point_id,
        state_id=point.state_id,
        variant="request",
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
        truth_source="fixture-authoring",
        author="fixture-reviewer",
        timestamp="2026-07-15T00:00:00Z",
        code_version="deadbeef",
        evidence={"policy_clause": "fixture request permits the action"},
    )
    truth.validate(point)
    return truth


def _write_release_dependencies(root, point: GroundingPoint,
                                formal_rows: list[dict],
                                truth: EvaluationTruthRecord) -> None:
    point_body = root / "grounded" / "probe_points.jsonl"
    point_manifest = root / "grounded" / "POINT_MANIFEST.jsonl"
    save_probe_points(
        [point], point_body, point_manifest, append=False)
    truth_body = root / "eval" / "truth.jsonl"
    truth_manifest = root / "eval" / "TRUTH_MANIFEST.jsonl"
    save_truth_records(
        [truth], truth_body, truth_manifest, append=False)
    formal = root / "train" / "formal" / config.FORMAL_SFT_PATH.name
    formal.parent.mkdir(parents=True, exist_ok=True)
    formal.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in formal_rows), encoding="utf-8")


def _fixture_trace(*, supervised: bool = True
                   ) -> tuple[StatelessEpisodeTrace, GroundingPoint, list[dict]]:
    history = []
    steps = []
    formal_rows = []
    signal = 0
    state_step_observations = None
    for index in range(9):
        pre = _observation(index, signal)
        post_signal = signal + 1 if index == 3 else signal
        # The post-state of one action is byte-for-byte the next policy call's
        # pre-state.  This prevents a fixture from accidentally legitimising
        # stitched, non-continuous "episodes".
        post = _observation(index + 1, post_signal)
        action = f"click('{index}')"
        messages = prompts.build_policy_messages(
            "complete the fixture workflow",
            pre.axtree_txt,
            history,
            system_prompt="fixture system",
            max_history=config.POLICY_HISTORY_STEPS,
            max_axtree_chars=config.MAX_AXTREE_CHARS_POLICY,
        )
        completion = _full_iris(action)
        turn_type = ("decision" if index == 8 else
                     "state_changing" if index == 3 else "routine")
        entry = history_entry(action, pre.to_view(), post.to_view())
        sample_id = f"sample-{index}" if supervised else ""
        point_linked = index in {3, 8}
        point_id = "point-state-change" if point_linked else ""
        state_id = "state-state-change" if point_linked else ""
        evaluation_case_id = (
            "case-final" if index == 8 else
            "case-state-change" if index == 3 else "")
        step = EpisodeStep(
            step_index=index,
            pre_observation=pre.to_dict(),
            input_messages=messages,
            input_messages_sha256=canonical_json_sha256(messages),
            assistant_completion=completion,
            raw_action=action,
            turn_type=turn_type,
            action_legal=True,
            executed=True,
            guarded=False,
            post_observation=post.to_dict(),
            history_entry=entry,
            history_steps_total=index,
            history_steps_kept=min(index, config.POLICY_HISTORY_STEPS),
            probe_point_id=point_id,
            state_id=state_id,
            supervised_sample_id=sample_id,
            evaluation_case_id=evaluation_case_id,
        )
        steps.append(step.to_dict())
        if supervised:
            formal_rows.append({
                "sample_id": sample_id,
                "messages": messages + [{
                    "role": "assistant", "content": completion}],
                "meta": {
                    "formal_dataset": True,
                    "message_topology": MESSAGE_TOPOLOGY,
                    "history_source": "trajectory",
                    "history_steps_total": index,
                    "history_steps_kept": min(
                        index, config.POLICY_HISTORY_STEPS),
                    "probe_point_id": point_id,
                    "state_id": state_id,
                    "evaluation_case_id": evaluation_case_id,
                    "assistant_turn_types": [turn_type],
                    "prompts_fp": "prompt-content-fixture",
                    "prompt_generation_fp": "prompt-generation-fixture",
                },
            })
        if index == 3:
            state_step_observations = (pre, post)
        history.append(entry)
        signal = post_signal

    assert state_step_observations is not None
    state_pre, state_post = state_step_observations
    point = GroundingPoint(
        probe_point_id="point-state-change",
        probe_run_id="probe-run-1",
        probe_name="fixture.state_change",
        state_id="state-state-change",
        candidate_id="candidate-state-change",
        action_instance_id="action-state-change",
        action_type="fixture_change",
        raw_action="click('3')",
        canonical_action="click:fixture:3",
        site="fixture",
        environment_family="fixture-family",
        environment_instance="fixture:1",
        environment_origin="fixture-test",
        is_mock=False,
        task_id="task-1",
        trajectory_id="trajectory-1",
        run_id="run-1",
        seed=7,
        url=state_pre.url,
        account="fixture-account",
        privilege="user",
        budget_k=9,
        solver_set=["fixture-controller"],
        controller_version="fixture-controller-v1",
        pre_observation_hash=_point_snapshot_hash(
            state_pre, "click('3')"),
        pre_signal={"counter": 0},
        post_observation_hash=_point_snapshot_hash(
            state_post, "click('3')"),
        post_signal={"counter": 1},
        undo_actions=["click('undo')"],
        undo_semantic_actions=["decrement_counter"],
        undo_observation_hashes=["undo-hash"],
        final_signal={"counter": 0},
        effect_status=EFFECT_CHANGED,
        recovery_status=RECOVERY_RECOVERED,
        undo_cost_steps=1,
        residual_diff={},
        budget_exhausted=False,
        timestamp="2026-07-15T00:00:00Z",
        code_version="deadbeef",
        evidence={
            "signal_channels": ["persistent_fixture"],
            "candidate_snapshot_hash": _point_snapshot_hash(
                state_pre, "click('3')"),
        },
    )
    trace = StatelessEpisodeTrace(
        schema_version=EPISODE_TRACE_SCHEMA_VERSION,
        episode_trace_id="",
        rollout_run_id="rollout-run-1",
        episode_id="episode-1",
        task_id="task-1",
        trajectory_id="trajectory-1",
        goal="complete the fixture workflow",
        system_prompt="fixture system",
        prompts_fp="prompt-content-fixture",
        prompt_generation_fp="prompt-generation-fixture",
        policy_builder=POLICY_BUILDER_ID,
        message_topology=MESSAGE_TOPOLOGY,
        max_history_steps=config.POLICY_HISTORY_STEPS,
        max_axtree_chars=config.MAX_AXTREE_CHARS_POLICY,
        environment_family="fixture-family",
        environment_instance="fixture:1",
        environment_origin="fixture-test",
        is_mock=False,
        collector_success=True,
        seed=7,
        steps=steps,
        timestamp="2026-07-15T00:00:00Z",
        code_version="deadbeef",
    )
    trace = replace(trace, episode_trace_id=episode_trace_id_for(trace))
    trace.validate()
    return trace, point, formal_rows


def test_round_trip_and_audit_prove_stateless_history_and_intermediate(tmp_path):
    trace, point, formal_rows = _fixture_trace()
    truth = _truth(point)
    body, manifest = episode_trace_paths(tmp_path)
    save_episode_traces([trace], body, manifest)

    loaded = assert_episode_manifest_integrity(body, manifest)
    assert loaded == {trace.episode_trace_id: trace}
    report = audit_episode_trace_evidence(
        tmp_path, {point.probe_point_id: point}, formal_rows,
        {truth.evaluation_case_id: truth})
    assert report["integrity"] is True
    assert report["n_episodes"] == 1
    assert report["n_steps"] == 9
    assert report["n_supervised_verified_steps"] == 9
    assert report["history_buckets"] == {"1_3": 3, "4_6": 3, "ge_7": 2}
    assert report["configuration_mismatches"] == []
    assert report["intermediate_state_changing"]["verified_count"] == 1
    assert report["intermediate_state_changing"]["passed"] is True
    assert report["truth_integrity"] is True
    assert report["final_decision_truth_join"]["verified_count"] == 1
    assert report["final_decision_truth_join"]["passed"] is True


def test_raw_trace_without_exact_sft_join_is_not_supervision(tmp_path):
    trace, point, _ = _fixture_trace(supervised=False)
    truth = _truth(point)
    body, manifest = episode_trace_paths(tmp_path)
    save_episode_traces([trace], body, manifest)

    report = audit_episode_trace_evidence(
        tmp_path, {point.probe_point_id: point}, [],
        {truth.evaluation_case_id: truth})
    assert report["integrity"] is True
    assert report["n_steps"] == 9
    assert report["n_supervised_verified_steps"] == 0
    assert report["history_buckets"] == {"1_3": 0, "4_6": 0, "ge_7": 0}
    assert report["intermediate_state_changing"]["passed"] is False
    assert "missing supervised_sample_id" in next(iter(
        report["supervision_rejections"].values()))


def test_intermediate_gate_rejects_a_nonexact_sft_derivation(tmp_path):
    trace, point, formal_rows = _fixture_trace()
    truth = _truth(point)
    body, manifest = episode_trace_paths(tmp_path)
    save_episode_traces([trace], body, manifest)
    formal_rows[3]["messages"][-1]["content"] = _full_iris("click('99')")

    report = audit_episode_trace_evidence(
        tmp_path, {point.probe_point_id: point}, formal_rows,
        {truth.evaluation_case_id: truth})
    intermediate = report["intermediate_state_changing"]
    assert intermediate["verified_count"] == 0
    assert intermediate["passed"] is False
    assert "formal SFT messages differ from trace call/completion" in next(iter(
        intermediate["rejected_turns"].values()))


def test_absent_asset_fails_closed_instead_of_counting_formal_metadata(tmp_path):
    _trace, point, formal_rows = _fixture_trace()
    report = audit_episode_trace_evidence(
        tmp_path, {point.probe_point_id: point}, formal_rows)
    assert report["integrity"] is False
    assert report["n_supervised_verified_steps"] == 0
    assert report["history_buckets"] == {"1_3": 0, "4_6": 0, "ge_7": 0}
    assert "artifact is absent" in report["error"]


def test_manifest_corruption_and_noncanonical_builder_fail_closed(tmp_path):
    trace, _point, _rows = _fixture_trace()
    body, manifest = episode_trace_paths(tmp_path)
    save_episode_traces([trace], body, manifest)
    pin = json.loads(manifest.read_text().splitlines()[0])
    pin["record_sha256"] = "0" * 64
    manifest.write_text(json.dumps(pin) + "\n")
    with pytest.raises(EpisodeTraceError, match="manifest/hash mismatch"):
        assert_episode_manifest_integrity(body, manifest)

    bad = replace(trace, policy_builder="ad_hoc.serializer")
    bad = replace(bad, episode_trace_id=episode_trace_id_for(bad))
    with pytest.raises(EpisodeTraceError, match="non-canonical policy builder"):
        bad.validate()


def test_duplicate_episode_identity_cannot_inflate_coverage(tmp_path):
    trace, _point, _rows = _fixture_trace()
    replay = replace(trace, timestamp="2026-07-15T00:00:01Z")
    replay = replace(replay, episode_trace_id=episode_trace_id_for(replay))
    body, manifest = episode_trace_paths(tmp_path)
    with pytest.raises(EpisodeTraceError, match="duplicate rollout_run_id"):
        save_episode_traces([trace, replay], body, manifest)


def test_duplicate_final_truth_case_cannot_inflate_episode_coverage(tmp_path):
    trace, _point, _rows = _fixture_trace()
    replay = replace(
        trace,
        rollout_run_id="rollout-run-2",
        episode_id="episode-2",
        trajectory_id="trajectory-2",
        timestamp="2026-07-15T00:00:01Z",
    )
    replay = replace(replay, episode_trace_id=episode_trace_id_for(replay))
    body, manifest = episode_trace_paths(tmp_path)
    with pytest.raises(EpisodeTraceError, match="duplicate final evaluation_case_id"):
        save_episode_traces([trace, replay], body, manifest)


def test_trace_rejects_fabricated_history_entry():
    trace, _point, _rows = _fixture_trace()
    first = dict(trace.steps[0])
    first["history_entry"] = {
        "action": "click('0')", "flag": "state-change", "delta": "invented"}
    broken = replace(trace, steps=[first, *trace.steps[1:]])
    broken = replace(broken, episode_trace_id=episode_trace_id_for(broken))
    with pytest.raises(EpisodeTraceError, match="observed-delta derived"):
        broken.validate()


def test_trace_rejects_stitched_noncontinuous_steps():
    trace, _point, _rows = _fixture_trace()
    second = dict(trace.steps[1])
    alien = _observation(88, 0)
    second["pre_observation"] = alien.to_dict()
    # Keep the per-step message self-consistent: continuity, rather than the
    # builder check, must be what rejects the stitched trace.
    second["input_messages"] = prompts.build_policy_messages(
        trace.goal, alien.axtree_txt, [trace.steps[0]["history_entry"]],
        system_prompt=trace.system_prompt,
        max_history=trace.max_history_steps,
        max_axtree_chars=trace.max_axtree_chars,
    )
    second["input_messages_sha256"] = canonical_json_sha256(
        second["input_messages"])
    broken = replace(trace, steps=[trace.steps[0], second, *trace.steps[2:]])
    broken = replace(broken, episode_trace_id=episode_trace_id_for(broken))
    with pytest.raises(EpisodeTraceError, match="previous post observation"):
        broken.validate()


def test_replay_trajectory_may_differ_from_point_collection_trajectory(tmp_path):
    trace, point, formal_rows = _fixture_trace()
    truth = _truth(point)
    trace = replace(trace, trajectory_id="fresh-live-replay-trajectory")
    trace = replace(trace, episode_trace_id=episode_trace_id_for(trace))
    body, manifest = episode_trace_paths(tmp_path)
    save_episode_traces([trace], body, manifest)
    report = audit_episode_trace_evidence(
        tmp_path, {point.probe_point_id: point}, formal_rows,
        {truth.evaluation_case_id: truth})
    assert report["intermediate_state_changing"]["passed"] is True


def test_final_truth_join_rejects_sft_state_metadata_mismatch(tmp_path):
    trace, point, formal_rows = _fixture_trace()
    truth = _truth(point)
    body, manifest = episode_trace_paths(tmp_path)
    save_episode_traces([trace], body, manifest)
    formal_rows[-1]["meta"]["state_id"] = "state-from-another-case"

    report = audit_episode_trace_evidence(
        tmp_path, {point.probe_point_id: point}, formal_rows,
        {truth.evaluation_case_id: truth})
    final_join = report["final_decision_truth_join"]
    assert final_join["verified_count"] == 0
    assert final_join["passed"] is False
    reasons = next(iter(final_join["rejected_turns"].values()))
    assert "joined SFT state_id mismatch" in reasons
    assert any(reason.startswith("final truth/trace/SFT meta mismatch")
               for reason in reasons)


def test_strict_import_writes_canonical_body_and_rejects_legacy_raw(tmp_path):
    trace, point, formal_rows = _fixture_trace()
    truth = _truth(point)
    source = tmp_path / "reviewed.jsonl"
    source.write_text(json.dumps(trace.to_dict()) + "\n")
    root = tmp_path / "release"
    _write_release_dependencies(root, point, formal_rows, truth)
    report = import_episode_trace_file(source, root)
    assert report["n_imported"] == report["n_stored"] == 1
    assert report["n_exact_sft_joins"] == 9
    assert report["n_intermediate_state_changes"] == 1
    body, manifest = episode_trace_paths(root)
    assert_episode_manifest_integrity(body, manifest)

    legacy = tmp_path / "legacy-step-record.jsonl"
    legacy.write_text(json.dumps({
        "task_id": "webarena.1", "step_id": 1,
        "action": "click('42')", "obs_after_axtree": "[43] link 'Next'",
    }) + "\n")
    with pytest.raises(EpisodeTraceError, match="unknown stateless episode trace"):
        import_episode_trace_file(legacy, tmp_path / "legacy-release")

    unlinked, _point, _rows = _fixture_trace(supervised=False)
    unlinked_path = tmp_path / "unlinked.jsonl"
    unlinked_path.write_text(json.dumps(unlinked.to_dict()) + "\n")
    with pytest.raises(EpisodeTraceError, match="requires supervised_sample_id"):
        import_episode_trace_file(
            unlinked_path, tmp_path / "unlinked-release",
            points={point.probe_point_id: point}, formal_rows=[],
            truths={truth.evaluation_case_id: truth})


def test_import_rejects_corrupt_evaluation_truth_manifest_before_write(tmp_path):
    trace, point, formal_rows = _fixture_trace()
    truth = _truth(point)
    root = tmp_path / "release"
    _write_release_dependencies(root, point, formal_rows, truth)
    truth_manifest = root / "eval" / "TRUTH_MANIFEST.jsonl"
    pin = json.loads(truth_manifest.read_text())
    pin["record_sha256"] = "0" * 64
    truth_manifest.write_text(json.dumps(pin) + "\n")
    source = tmp_path / "reviewed-corrupt-truth.jsonl"
    source.write_text(json.dumps(trace.to_dict()) + "\n")

    with pytest.raises(
            EpisodeTraceError, match="canonical evaluation truth is unavailable"):
        import_episode_trace_file(source, root)
    body, manifest = episode_trace_paths(root)
    assert not body.exists()
    assert not manifest.exists()


def test_import_episode_trace_cli_is_explicit_and_offline():
    args = build_parser().parse_args([
        "import-episode-traces", "--input", "reviewed.jsonl",
        "--data-root", "release", "--append",
    ])
    assert args.fn.__name__ == "cmd_import_episode_traces"
    assert args.append is True
    collect = build_parser().parse_args([
        "collect", "--task-ids", "webarena.117", "--policy", "iris",
        "--provider", "openrouter", "--read-only-live",
        "--code-version", "worktree:test",
    ])
    assert collect.policy == "iris"
    assert collect.provider == "openrouter"
    assert collect.read_only_live is True


def test_collector_preserves_exact_episode_source_without_promoting_it():
    class _OneCallPolicy:
        def __init__(self):
            self.last_request_messages = []
            self.last_raw_response = ""
            self.last_finish_reason = "stop"

        def reset(self):
            self.last_request_messages = []
            self.last_raw_response = ""

        def act(self, view, goal="", history=None):
            self.last_request_messages = prompts.build_policy_messages(
                goal, view["axtree_txt"], history or [],
                system_prompt="fixture collector")
            self.last_raw_response = "<answer> click('11')"
            return "click('11')"

    renv = RevActEnv(
        MockShoppingEnv(goal="browse the fixture"), task_id="mock.fixture")
    _states, summary = collect_trajectory(
        renv, _OneCallPolicy(), seed=0, trajectory_id="trace-source",
        run_id="run-source", max_steps=1, code_version="worktree:test")
    initial, action = renv.logger.records
    assert initial.policy_call_captured is False
    assert action.policy_call_captured is True
    assert action.policy_input_messages_sha256 == canonical_json_sha256(
        action.policy_input_messages)
    assert action.assistant_completion_sha256 == observation_sha256(
        action.assistant_completion)
    assert action.pre_observation["axtree_txt"]
    assert action.post_observation["axtree_txt"]
    assert action.observed_history_entry["action"] == "click('11')"
    assert action.code_version == "worktree:test"
    assert summary["episode_source_capture"] == {
        "schema_version": "iris.raw_episode_source.v1",
        "n_action_steps": 1,
        "n_exact_policy_calls": 1,
        "counts_as_formal_supervision": False,
    }
    assert summary["policy_attempt_source_capture"] == {
        "schema_version": "iris.policy-attempt.v1",
        "n_policy_attempts": 1,
        "n_unexecuted_policy_attempts": 0,
        "counts_as_environment_transitions": False,
        "counts_as_formal_supervision": False,
    }
    attempt = renv.policy_attempt_logger.records[0]
    assert attempt.execution_status == "EXECUTED"
    assert attempt.executed_action == "click('11')"
    assert attempt.policy_input_messages_sha256 == canonical_json_sha256(
        attempt.policy_input_messages)


def test_collector_captures_policy_call_that_yields_no_action():
    class _MalformedPolicy:
        provider = "openrouter"
        model = "fixture/model"

        def reset(self):
            self.last_request_messages = []
            self.last_raw_response = ""
            self.last_finish_reason = ""

        def act(self, view, goal="", history=None):
            self.last_request_messages = prompts.build_policy_messages(
                goal, view["axtree_txt"], history or [],
                system_prompt="fixture collector")
            self.last_raw_response = "<think>truncated before answer"
            self.last_finish_reason = "length"
            return None

    renv = RevActEnv(
        MockShoppingEnv(goal="browse the fixture"), task_id="mock.fixture")
    _states, summary = collect_trajectory(
        renv, _MalformedPolicy(), seed=0, trajectory_id="trace-no-action",
        run_id="run-no-action", max_steps=1, code_version="worktree:test")

    assert renv.step_id == 0
    assert len(renv.logger.records) == 1  # initial observation only
    assert summary["episode_source_capture"]["n_action_steps"] == 0
    assert summary["episode_source_capture"]["n_exact_policy_calls"] == 0
    assert summary["policy_attempt_source_capture"]["n_policy_attempts"] == 1
    assert summary["policy_attempt_source_capture"][
        "n_unexecuted_policy_attempts"] == 1
    attempt = renv.policy_attempt_logger.records[0]
    assert attempt.execution_status == "NO_ACTION"
    assert attempt.executed_action == ""
    assert attempt.executed_completion == ""
    assert attempt.proposed_completion == "<think>truncated before answer"
    assert attempt.proposed_completion_sha256 == observation_sha256(
        attempt.proposed_completion)
    assert attempt.finish_reason == "length"
    assert attempt.provider == "openrouter"
    assert attempt.model == "fixture/model"
