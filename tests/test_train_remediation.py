"""Training gates added by the point-level remediation."""
from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from revact import config
from revact.train import dpo, grpo, sft
from revact.train.distill import (distill_sample, evidence_token, qc_check,
                                  qc_full_sample, run as run_distill)
from revact.train.sft import completion_encoding_report, validate_rows
from revact.train.validators import (actions_match, decision_answer_consistent,
                                     parse_action)


class _CharChatTokenizer:
    """Tiny prefix-preserving tokenizer fixture; no model/network dependency."""

    def apply_chat_template(self, messages, *, add_generation_prompt=False,
                            tokenize=True, return_dict=True):
        text = ""
        for message in messages:
            text += f"<{message['role']}>" + message["content"] + "</turn>"
        if add_generation_prompt:
            text += "<assistant>"
        return {"input_ids": [ord(char) for char in text]}


def _assistant(answer="click('12')", decision="EXECUTE"):
    return (
        "<think>\n<observation> o\n<reasoning> r\n"
        "<prediction> measured from {\"x\": 0} to {\"x\": 1}\n"
        "<rev_check> pp-1 recorded undo\n<reversibility> RECOVERED\n"
        "<undo> click('3'); undo_cost_steps=1\n"
        f"<decision> {decision} risk=0.1\n</think>\n<answer> {answer}")


def _sft_row(sample_id="s", assistant=None, formal=False):
    return {
        "sample_id": sample_id,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "<goal>\ng\n\n<observation>\n[12] button 'Do'"},
            {"role": "assistant", "content": assistant or _assistant()},
        ],
        "meta": ({"formal_dataset": True, "is_mock": False,
                  "collector_success": True, "turn_type": "decision",
                  "probe_point_id": "pp-1", "state_id": "state-1",
                  "action_instance_id": "action-1",
                  "prediction_source": "probe_transition",
                  "undo_source": "probe_point_id", "effect_status": "CHANGED",
                  "recovery_status": "RECOVERED",
                  "history_source": "trajectory", "decision": "EXECUTE",
                  "risky_action": {"name": "click", "bid": "12"},
                  "risky_raw_action": "click('12')",
                  "post_signal_diff": {"pre_signal": {"x": 0},
                                       "post_signal": {"x": 1}},
                  "undo_actions": ["click('3')"], "undo_cost_steps": 1,
                  "policy_input_observation_hash": hashlib.sha256(
                      "[12] button 'Do'".encode()).hexdigest(),
                  "evidence": {"fixture": True}}
                 if formal else {}),
    }


def test_completion_only_token_report_exposes_and_rejects_drop_inputs():
    short = _sft_row("short")
    long = _sft_row("long", _assistant() + "x" * 500)
    report = completion_encoding_report(
        [short, long], _CharChatTokenizer(), max_len=400,
        include_examples=True)
    assert report["n_dropped"] == 1
    assert report["dropped_sample_ids"] == ["long"]
    assert report["token_p50"] > 0 and report["token_p95"] > report["token_p50"]
    assert report["completion_only"] is True
    labels = report["examples"][0]["labels"]
    assert -100 in labels and any(label != -100 for label in labels)


def test_dpo_and_rlvr_token_reports_expose_truncation_by_id():
    pair = _pair("pair-long", "synthetic_flip")
    pair["rejected"] += "x" * 500
    dpo_report = dpo.pair_encoding_report(
        [pair], _CharChatTokenizer(), max_len=400)
    assert dpo_report["n_dropped"] == 1
    assert dpo_report["dropped_pair_ids"] == ["pair-long"]

    prompt_row = {"sample_id": "prompt-long", "prompt": [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "x" * 500},
    ]}
    prompt_report = grpo.prompt_encoding_report(
        [prompt_row], _CharChatTokenizer(), max_prompt_len=400)
    assert prompt_report["n_dropped"] == 1
    assert prompt_report["dropped_prompt_ids"] == ["prompt-long"]




def test_each_trainer_dry_run_fails_on_nonzero_token_drop(
        tmp_path, monkeypatch, capsys):
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: _CharChatTokenizer()))
    original_train = sft.config.TRAIN

    sft_row = _sft_row("sft-drop", formal=True)
    sft_path = tmp_path / "sft.jsonl"
    sft_path.write_text(json.dumps(sft_row) + "\n")
    monkeypatch.setattr(sft, "formal_point_join_problems",
                        lambda *_args, **_kwargs: [])
    assert sft.run(sft_path, dry_run=True, max_len=100) == 1
    assert "drop=1" in capsys.readouterr().out

    pair = _pair(
        "dpo-drop", "legal_candidate", candidate_id="source-candidate",
        negative_candidate_id="negative-candidate", legal_at_snapshot=True,
        negative_candidate_snapshot_hash="sha256:fixture")
    dpo_path = tmp_path / "dpo.jsonl"
    dpo_path.write_text(json.dumps(pair) + "\n")
    monkeypatch.setattr(dpo, "formal_point_join_problems",
                        lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        dpo.config, "TRAIN", dataclasses.replace(original_train, max_len=100))
    assert dpo.run(dpo_path, dry_run=True) == 1
    assert "drop=1" in capsys.readouterr().out

    rlvr_row = _sft_row("rlvr-drop", formal=True)
    rlvr_row["messages"][1]["content"] += " long" * 100
    rlvr_row["meta"].update({
        "variant": "request", "normative_risk": False,
        "violates_constraint": False, "action_legal": True,
        "backend_commit": False,
    })
    rlvr_path = tmp_path / "rlvr.jsonl"
    rlvr_path.write_text(json.dumps(rlvr_row) + "\n")
    monkeypatch.setattr(grpo, "formal_point_join_problems",
                        lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        grpo.config, "TRAIN", dataclasses.replace(original_train, max_len=400))
    assert grpo.run([rlvr_path], dry_run=True) == 1
    assert "drop=1" in capsys.readouterr().out

def test_formal_state_changing_turn_requires_explicit_type_and_full_block():
    row = _sft_row(formal=True)
    row["meta"].pop("turn_type")
    assert "explicitly type" in validate_rows([row])["problems"][0]
    row["meta"]["turn_type"] = "state_changing"
    row["messages"][-1]["content"] = "<answer> click('12')"
    report = validate_rows([row])
    assert report["n_problems"] > 0
    assert any("tag <think>" in problem for problem in report["problems"])


def _pair(pair_id, source, **extra):
    meta = {
        "formal_dataset": True, "is_mock": False,
        "collector_success": True, "negative_source": source,
        "pair_type": "false_safe", "probe_point_id": "pp-1",
        "state_id": "state-1", "action_instance_id": "action-1",
        "effect_status": "CHANGED", "recovery_status": "RECOVERED",
        "prediction_source": "probe_transition",
        "undo_source": "probe_point_id", "history_source": "trajectory",
        "normative_risk": False, "decision": "EXECUTE",
        "risky_action": {"name": "click", "bid": "12"},
        "risky_raw_action": "click('12')",
        "post_signal_diff": {"pre_signal": {"x": 0},
                             "post_signal": {"x": 1}},
        "undo_actions": ["click('3')"], "undo_cost_steps": 1,
        "policy_input_observation_hash": hashlib.sha256(
            "[12] button 'Do'".encode()).hexdigest(),
        "evidence": {"fixture": True},
        **extra,
    }
    return {
        "pair_id": pair_id,
        "prompt": [{"role": "system", "content": "s"},
                   {"role": "user", "content": "<observation>\n[12] button 'Do'"}],
        "chosen": _assistant(),
        "rejected": _assistant(answer="go_back()", decision="AVOID"),
        "meta": meta,
    }


def test_dpo_formal_negative_source_gate_and_legacy_escape_hatch():
    rows = [
        _pair("synthetic", "synthetic_flip"),
        _pair("legal", "legal_candidate", candidate_id="source-c1",
              negative_candidate_id="negative-c1", legal_at_snapshot=True,
              negative_candidate_snapshot_hash="sha256:s1"),
    ]
    report = dpo.validate_rows(rows)
    assert report["n_problems"] == 0
    assert report["deployment_negative_ratio"] == .5
    only_synthetic = dpo.validate_rows([rows[0]])
    assert only_synthetic["n_problems"] == 1
    assert "ratio" in only_synthetic["problems"][0]
    legacy = [{**rows[0], "meta": {"pair_type": "false_safe"}}]
    assert dpo.validate_rows(legacy)["n_problems"] == 0


def test_train_entrypoints_fail_closed_on_legacy_even_when_real_run_requested(
        tmp_path, capsys):
    sft_path = tmp_path / "legacy_sft.jsonl"
    sft_path.write_text(__import__("json").dumps(_sft_row()) + "\n")
    assert sft.run(sft_path, dry_run=True) == 1
    assert "--allow-legacy" in capsys.readouterr().out
    # The escape hatch is audit-only; it cannot authorise a real trainer.
    assert sft.run(sft_path, dry_run=False, allow_legacy=True) == 1

    pair = _pair("legacy", "synthetic_flip")
    pair["meta"] = {"pair_type": "false_safe"}
    dpo_path = tmp_path / "legacy_dpo.jsonl"
    dpo_path.write_text(__import__("json").dumps(pair) + "\n")
    assert dpo.run(dpo_path, dry_run=True) == 1
    assert dpo.run(dpo_path, dry_run=False, allow_legacy=True) == 1

    legacy_rlvr = _sft_row()
    legacy_rlvr["meta"] = {
        "decision": "EXECUTE", "reversibility": "RECOVERED",
        "variant": "request", "risky_raw_action": "click('12')",
    }
    rlvr_path = tmp_path / "legacy_rlvr.jsonl"
    rlvr_path.write_text(__import__("json").dumps(legacy_rlvr) + "\n")
    assert grpo.run([rlvr_path], dry_run=True) == 1
    assert grpo.run([rlvr_path], dry_run=False, allow_legacy=True) == 1


def test_empty_training_assets_never_report_vacuous_success(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert sft.run(empty, dry_run=True) == 1
    assert dpo.run(empty, dry_run=True) == 1
    assert grpo.run([empty], dry_run=True) == 1


def test_structured_action_parser_ignores_prose_and_code():
    click = parse_action("click('12')")
    assert click and click.bid == "12"
    assert actions_match(click, {"name": "click", "bid": "12"})
    assert parse_action("I might click('12')") is None
    assert parse_action("obj.click('12')") is None
    assert parse_action("click(f())") is None
    assert not decision_answer_consistent("AVOID", "click('12')", "click('12')")


def test_offline_rlvr_adversarial_behavior_truth_table():
    def completion(decision, answer, prose=""):
        return ("<think>\n<observation> o " + prose +
                "\n<reasoning> r\n<prediction> p\n<rev_check> c\n"
                "<reversibility> NOT_RECOVERED_WITHIN_BUDGET\n<undo> none\n"
                f"<decision> {decision}\n</think>\n<answer> {answer}")

    cases = [
        completion("AVOID", "go_back()", "mentions click('12') only"),
        completion("AVOID", "click('12')"),
        completion("CONFIRM", "click('12')"),
        completion("VERIFY", "click('12')"),
        completion("EXECUTE", "click('12')"),
        completion("EXECUTE", "click('99')"),
    ]
    target = [{"name": "click", "bid": "12"}] * len(cases)
    safety = grpo.reward_safety(
        cases, variant=["constraint"] * len(cases), risky_action=target,
        action_legal=[True] * len(cases),
        backend_commit=[False, False, True, False, True, False])
    assert safety == [0.0, -8.0, -12.0, -8.0, -4.0, 0.0]
    consistency = grpo.reward_decision_answer_consistency(
        cases, risky_action=target)
    assert consistency == [1.0, -2.0, -2.0, -2.0, 1.0, -2.0]
    # An illegal action string is not counted as a legal attempt/commit.
    assert grpo.reward_safety(
        [cases[1]], variant=["constraint"], risky_action=[target[0]],
        action_legal=[False], backend_commit=[True]) == [0.0]
    # Normative risk is independent of the request/constraint text variant.
    assert grpo.reward_safety(
        [cases[4]], variant=["request"], risky_action=[target[0]],
        action_legal=[True], backend_commit=[False], normative_risk=[True],
        violates_constraint=[False]) == [-4.0]
    duplicated = cases[0] + "\n<answer> click('12')"
    assert grpo.reward_format([duplicated]) == [0.0]


def test_formal_rlvr_requires_explicit_normative_legality_and_commit_truth():
    row = {
        "prompt": [{"role": "system", "content": "s"},
                   {"role": "user", "content": "u"}],
        "gold_decision": "EXECUTE", "gold_reversibility": "RECOVERED",
        "gold_recovery_status": "RECOVERED", "variant": "request",
        "risky_raw_action": "click('12')",
        "risky_action": {"name": "click", "bid": "12"},
        "action_legal": True, "backend_commit": False,
        "normative_risk": False, "violates_constraint": False,
        "formal_dataset": True, "probe_point_id": "pp-1",
        "state_id": "s-1", "action_instance_id": "a-1",
        "effect_status": "CHANGED", "prediction_source": "probe_transition",
        "undo_source": "probe_point_id", "history_source": "trajectory",
        "is_mock": False, "collector_success": True,
    }
    assert grpo.validate_rows([row])["n_problems"] == 0
    for key in ("normative_risk", "violates_constraint", "backend_commit"):
        broken = dict(row)
        broken[key] = None
        assert any(key in problem for problem in
                   grpo.validate_rows([broken])["problems"])


def _formal_distill_row():
    row = _sft_row(formal=True)
    row["messages"][-1]["content"] = row["messages"][-1]["content"].replace(
        'measured from {"x": 0} to {"x": 1}',
        "measured " + evidence_token("post_diff", {"cart_count": [0, 1]}))
    row["meta"].update({
        "action_type": "add_to_cart",
        "reversibility": "RECOVERED",
        "recovery_status": "RECOVERED",
        "decision": "EXECUTE",
        "probe_point_id": "pp-1",
        "prediction_source": "probe_transition",
        "undo_source": "probe_point_id",
        "undo_source_probe_point_id": "pp-1",
        "post_signal_diff": {"cart_count": [0, 1]},
        "undo_actions": ["click('3')"],
        "undo_cost_steps": 1,
        "risky_action": {"name": "click", "bid": "12"},
        "risky_raw_action": "click('12')",
    })
    return row


def test_formal_teacher_qc_requires_evidence_and_checks_full_output():
    input_ref = evidence_token("input", "[12] button 'Do'")
    post_ref = evidence_token("post_diff", {"cart_count": [0, 1]})
    undo_ref = evidence_token(
        "undo", {"actions": ["click('3')"], "undo_cost_steps": 1})
    teacher = (
        f"<observation> The target is visible. {input_ref}\n"
        "<reasoning> The measured transition and recovery support proceeding.\n"
        f"<prediction> Cart count changes from zero to one. {post_ref}\n"
        f"<rev_check> The recorded undo click restored it. {undo_ref}")
    row = _formal_distill_row()
    report = distill_sample(row, lambda _: teacher)
    assert report == {"ok": True, "attempts": 1}
    assert row["meta"]["prose_source"] == "teacher"
    assert row["meta"]["rev_check_source"] == "teacher"
    assert qc_full_sample(row) is None
    assert "post_diff evidence citation" in qc_check(
        teacher.replace(f" {post_ref}", ""), "RECOVERED", "EXECUTE",
        formal=True, input_observation="[12] button 'Do'",
        post_signal_diff={"cart_count": [0, 1]},
        undo_actions=["click('3')"], undo_cost_steps=1)
    hallucinated = teacher.replace(
        "The recorded undo click restored it.",
        "A Remove control exists and restores it.")
    assert "absent from input" in qc_check(
        hallucinated, "RECOVERED", "EXECUTE", formal=True,
        input_observation="[12] button 'Do'",
        post_signal_diff={"cart_count": [0, 1]},
        undo_actions=["click('3')"], undo_cost_steps=1)




@pytest.mark.parametrize(("old", "new", "expected"), [
    ("<undo> click('3');", "<undo> click('4');",
     "completion_undo_action_mismatch"),
    ("undo_cost_steps=1", "undo_cost_steps=2",
     "completion_undo_cost_mismatch"),
    ("<decision> EXECUTE", "<decision> AVOID",
     "completion_decision_mismatch"),
    ("<answer> click('12')", "<answer> go_back()",
     "completion_decision_answer_inconsistent"),
])
def test_formal_teacher_full_sample_rejects_pinned_output_tampering(
        old, new, expected):
    input_ref = evidence_token("input", "[12] button 'Do'")
    post_ref = evidence_token("post_diff", {"cart_count": [0, 1]})
    undo_ref = evidence_token(
        "undo", {"actions": ["click('3')"], "undo_cost_steps": 1})
    teacher = (
        f"<observation> The target is visible. {input_ref}\n"
        "<reasoning> The measured transition and recovery support proceeding.\n"
        f"<prediction> Cart count changes from zero to one. {post_ref}\n"
        f"<rev_check> The recorded undo click restored it. {undo_ref}")
    row = _formal_distill_row()
    assert distill_sample(row, lambda _prompt: teacher)["ok"] is True
    row["messages"][-1]["content"] = row["messages"][-1]["content"].replace(
        old, new, 1)
    assert qc_full_sample(row) == expected

def test_teacher_source_failure_is_explicit_template_fallback_without_call():
    row = _formal_distill_row()
    row["meta"].pop("probe_point_id")
    called = []
    report = distill_sample(row, lambda _: called.append(True) or "")
    assert not report["ok"] and report["attempts"] == 0
    assert not called
    assert row["meta"]["prose_source"] == "template"
    assert row["meta"]["rev_check_source"] == "template"
    assert row["meta"]["teacher_qc_status"] == "source_rejected"


def test_formal_distill_binds_opaque_evidence_tokens_in_pipeline():
    row = _formal_distill_row()
    teacher_without_hashes = (
        "<observation> The page shows the task target.\n"
        "<reasoning> The recorded transition supports the requested action.\n"
        "<prediction> The measured cart count changes.\n"
        "<rev_check> The recorded undo restored the prior state.")
    report = distill_sample(row, lambda _prompt: teacher_without_hashes)
    assert report["ok"] is True
    assistant = row["messages"][-1]["content"]
    assert evidence_token("input", "[12] button 'Do'") in assistant
    assert evidence_token("post_diff", {"cart_count": [0, 1]}) in assistant
    assert evidence_token(
        "undo", {"actions": ["click('3')"], "undo_cost_steps": 1}) in assistant
    assert qc_full_sample(row) is None


def test_distill_run_records_separate_teacher_generation_fingerprint(tmp_path):
    import json
    from revact.prompt_store import load_generation_bundle

    class _InvalidTeacher:
        model = "fixture-teacher"
        base_url = "http://fixture.invalid/v1"

        @staticmethod
        def complete(_prompt):
            return "invalid teacher prose"

    source = tmp_path / "formal.jsonl"
    source.write_text(json.dumps(_formal_distill_row()) + "\n")
    output = tmp_path / "iris_sft_distilled_point_v1.jsonl"
    # Invalid prose is quarantined and the coverage gate correctly fails, but
    # the attempted generation remains exactly reproducible.
    assert run_distill(
        source, output, limit=1, client=_InvalidTeacher(),
        provenance_root=tmp_path, min_formal_teacher_coverage=.95) == 1
    fallback = output.with_name(
        output.stem + ".template_fallback.jsonl")
    row = json.loads(fallback.read_text())
    meta = row["meta"]
    assert meta["prose_source"] == "template"
    assert meta["rev_check_source"] == "template"
    assert meta["teacher_prompts_fp"]
    generation = load_generation_bundle(
        meta["teacher_prompt_generation_fp"], root=tmp_path)
    assert generation["prompts_fp"] == meta["teacher_prompts_fp"]
    assert generation["model"]["name"] == "fixture-teacher"
    assert generation["decode_config"]["temperature"] == .7


def test_distill_all_materializes_independent_single_and_multiturn_artifacts(
        tmp_path, monkeypatch):
    class _FixtureTeacher:
        model = "fixture-teacher"
        base_url = "http://fixture.invalid/v1"

        def __init__(self):
            self.calls = 0

        def complete(self, _prompt):
            self.calls += 1
            return (
                "<observation> The target is visible. " +
                evidence_token("input", "[12] button 'Do'") + "\n" +
                "<reasoning> Measured transition and recovery support proceeding.\n" +
                "<prediction> The cart count changes. " +
                evidence_token("post_diff", {"cart_count": [0, 1]}) + "\n" +
                "<rev_check> The recorded undo restored the state. " +
                evidence_token(
                    "undo", {"actions": ["click('3')"],
                             "undo_cost_steps": 1}))

    formal = tmp_path / "train" / "formal"
    single_source = formal / "iris_sft_point_v1.jsonl"
    multi_source = formal / "iris_sft_multiturn_point_v1.jsonl"
    single_output = formal / "iris_sft_distilled_point_v1.jsonl"
    multi_output = formal / "iris_sft_multiturn_distilled_point_v1.jsonl"
    single = _formal_distill_row()
    single["sample_id"] = "single-1"
    multi = _formal_distill_row()
    multi["sample_id"] = "multiturn-1"
    formal.mkdir(parents=True)
    single_source.write_text(json.dumps(single) + "\n")
    multi_source.write_text(json.dumps(multi) + "\n")
    monkeypatch.setattr(config, "FORMAL_SFT_PATH", single_source)
    monkeypatch.setattr(config, "FORMAL_MULTITURN_SFT_PATH", multi_source)
    monkeypatch.setattr(config, "FORMAL_DISTILLED_SFT_PATH", single_output)
    monkeypatch.setattr(
        config, "FORMAL_MULTITURN_DISTILLED_SFT_PATH", multi_output)

    client = _FixtureTeacher()
    assert run_distill(
        family="all", limit=1, client=client, provenance_root=tmp_path) == 0
    assert client.calls == 2
    outputs = [json.loads(single_output.read_text()),
               json.loads(multi_output.read_text())]
    assert [row["sample_id"] for row in outputs] == [
        "single-1", "multiturn-1"]
    assert all((row["meta"]["prose_source"],
                row["meta"]["rev_check_source"],
                row["meta"]["teacher_qc_status"]) ==
               ("teacher", "teacher", "passed") for row in outputs)


def test_distill_cli_selects_family_and_custom_paths():
    from revact.cli import build_parser

    args = build_parser().parse_args([
        "distill", "--family", "multiturn", "--input", "in.jsonl",
        "--output", "out.jsonl", "--limit", "7", "--overwrite"])
    assert args.family == "multiturn"
    assert args.input == "in.jsonl" and args.output == "out.jsonl"
    assert args.limit == 7 and args.overwrite is True
