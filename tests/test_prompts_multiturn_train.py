"""P0/P1 offline tests: shared prompt format, multi-turn assembly, trainer
validation, and GRPO verifiable rewards. No torch, no network, no live env."""
import json

import pytest

from revact import prompts
from revact.data.multiturn import assemble_multiturn, build_conversation
from revact.train import dpo as dpo_train
from revact.train import grpo as grpo_train
from revact.train.sft import validate_rows as sft_validate


# ------------------------------------------------------------------ prompts -- #
def test_render_and_parse_roundtrip():
    hist = [{"action": "goto('http://x/p')", "obs": "http://x/p"},   # legacy entry
            {"action": "click('12')", "delta": "cart items 0 -> 1",  # P2 entry
             "flag": "state-change"}]
    user = prompts.render_user("Do not buy.", "RootWebArea 'P'", hist)
    assert user.startswith("<goal>\nDo not buy.\n\n<history>\n1. goto(")
    assert "2. click('12') -> [state-change] cart items 0 -> 1" in user
    p = prompts.parse_user(user)
    assert p["goal"] == "Do not buy."
    assert "click('12')" in p["history"]
    assert p["obs"] == "RootWebArea 'P'"


def test_history_no_effect_breaks_input_identity():
    """Two consecutive no-effect steps must yield DIFFERENT user inputs (the
    'model loops on identical (goal, obs)' failure the P2 format fixes)."""
    h1 = [{"action": "click('5')", "delta": "no visible change", "flag": "no-effect"}]
    h2 = h1 + [{"action": "click('5')", "delta": "no visible change",
                "flag": "no-effect"}]
    u1 = prompts.render_user("G", "OBS", h1)
    u2 = prompts.render_user("G", "OBS", h2)
    assert u1 != u2 and "[no-effect]" in u2


def test_parse_user_accepts_pre_p0_format():
    old = "<goal>\nG\n\n<observation>\nOBS\n"
    p = prompts.parse_user(old)
    assert p == {"goal": "G", "history": "", "obs": "OBS"}


def test_empty_history_renders_none():
    p = prompts.parse_user(prompts.render_user("G", "OBS", []))
    assert p["history"] == ""


def test_state_history_plan_vs_canonical():
    ents, src = prompts.state_history(
        {"reach_plan": [["goto", "http://x"], ["click_text", "add to cart"]]})
    assert src == "plan" and len(ents) == 2
    ents, src = prompts.state_history(
        {"reach_plan": [], "action_type": "place_order", "url": "http://x/checkout"})
    assert src == "canonical"
    assert any("checkout" in e["action"] for e in ents)


def test_policy_and_assemble_share_format():
    """The rollout policy's user message must parse with the same parser the
    training data uses — the P0 guarantee."""
    from revact.policies import LLMActionPolicy

    pol = LLMActionPolicy.__new__(LLMActionPolicy)   # no key needed
    pol.max_history = 6
    pol.system_prompt = prompts.SYSTEM_COLLECTOR
    msgs = LLMActionPolicy._build_messages(
        pol, {"axtree_txt": "RootWebArea 'X'"}, "G",
        [{"action": "click('1')", "obs": "o"}])
    p = prompts.parse_user(msgs[1]["content"])
    assert p["goal"] == "G" and "click('1')" in p["history"]


# ---------------------------------------------------------------- multiturn -- #
def _steps():
    return [
        {"step_id": 0, "action": None, "url_after": "http://x/",
         "obs_after_axtree": "RootWebArea 'Home'\n[11] link 'Shop'"},
        {"step_id": 1, "action": "click('11')", "url_after": "http://x/shop",
         "obs_after_axtree": "RootWebArea 'Shop'\n[20] link 'Item'"},
        {"step_id": 2, "action": "click('20')", "url_after": "http://x/item",
         "obs_after_axtree": "RootWebArea 'Item'\n[30] button 'Add to Cart'"},
    ]


def test_build_conversation_shapes():
    msgs = build_conversation(_steps(), k=2, goal="G")
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assert "<answer> click('11')" == msgs[2]["content"]
    assert msgs[-1]["content"].startswith("<observation>")
    # trajectories without a step-0 record are rejected, not mangled
    assert build_conversation(_steps()[1:], k=2, goal="G") is None


def test_assemble_multiturn_end_to_end(tmp_path):
    traj_dir = tmp_path / "traj"
    traj_dir.mkdir()
    with (traj_dir / "t1.jsonl").open("w") as f:
        for s in _steps():
            f.write(json.dumps(s) + "\n")
    ks = tmp_path / "ks.jsonl"
    ks.write_text(json.dumps({
        "trajectory_id": "t1", "step_id": 2, "site": "shopping",
        "afforded_action_types": ["add_to_cart"], "url": "http://x/item"}))
    res = assemble_multiturn(traj_dir, ks, {"add_to_cart": "REVERSIBLE"}, tmp_path)
    assert res["n_sft"] == 2 and res["n_dpo"] > 0          # both goal variants
    rows = [json.loads(ln) for ln in
            (tmp_path / "train" / "sft" / "revact_sft_multiturn.jsonl").open()]
    assert sft_validate(rows)["n_problems"] == 0
    r = rows[0]
    assert r["meta"]["kind"] == "multiturn"
    assert r["meta"]["risky_raw_action"] == "click('30')"  # real bid from obs
    assert r["messages"][-1]["content"].count("<reversibility>") == 1
    pairs = [json.loads(ln) for ln in
             (tmp_path / "train" / "dpo" / "revact_dpo_multiturn.jsonl").open()]
    assert dpo_train.validate_rows(pairs)["n_problems"] == 0
    assert all(p["prompt"][-1]["role"] == "user" for p in pairs)


def test_multiturn_skips_when_control_missing(tmp_path):
    traj_dir = tmp_path / "traj"
    traj_dir.mkdir()
    steps = _steps()
    steps[2]["obs_after_axtree"] = "RootWebArea 'Item'"     # no add-to-cart control
    with (traj_dir / "t1.jsonl").open("w") as f:
        for s in steps:
            f.write(json.dumps(s) + "\n")
    ks = tmp_path / "ks.jsonl"
    ks.write_text(json.dumps({
        "trajectory_id": "t1", "step_id": 2, "site": "shopping",
        "afforded_action_types": ["add_to_cart"], "url": "http://x/item"}))
    res = assemble_multiturn(traj_dir, ks, {"add_to_cart": "REVERSIBLE"}, tmp_path)
    assert res["n_sft"] == 0 and res["n_skipped"] == 1


# ----------------------------------------------------------------- trainers -- #
def test_sft_validate_role_patterns():
    good_single = [{"sample_id": "a", "messages": [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content":
         "<think>\n<observation> o\n<reasoning> r\n<prediction> p\n"
         "<rev_check> c\n<reversibility> REVERSIBLE\n<undo> u\n"
         "<decision> EXECUTE risk=0.1\n</think>\n"
         "<answer> click('1')"}], "meta": {}}]
    assert sft_validate(good_single)["n_problems"] == 0
    bad = [{"sample_id": "b", "messages": [
        {"role": "system", "content": "s"},
        {"role": "assistant", "content": "x"}], "meta": {}}]
    assert sft_validate(bad)["n_problems"] == 1


def test_grpo_rewards_are_verifiable():
    good = ("<think>\n<observation> o\n<reasoning> r\n<prediction> p\n"
            "<rev_check> c\n<reversibility> IRREVERSIBLE\n<undo> none available\n"
            "<decision> AVOID risk=0.9\n</think>\n"
            "<answer> go_back()")
    bad = ("<answer> click('99')\n<decision> EXECUTE")      # tags out of order
    comps = [[{"role": "assistant", "content": good}],
             [{"role": "assistant", "content": bad}]]
    assert grpo_train.reward_format(comps) == [1.0, 0.0]
    assert grpo_train.reward_decision(comps, gold_decision=["AVOID", "AVOID"]) \
        == [2.0, 0.0]
    assert grpo_train.reward_reversibility(
        comps, gold_reversibility=["IRREVERSIBLE", "IRREVERSIBLE"]) == [2.0, 0.0]
    assert grpo_train.reward_safety(
        comps, variant=["constraint", "constraint"],
        risky_raw_action=["click('99')", "click('99')"]) == [0.0, -4.0]


def test_grpo_load_prompts_strips_answer(tmp_path):
    p = tmp_path / "sft.jsonl"
    p.write_text(json.dumps({
        "sample_id": "a",
        "messages": [{"role": "system", "content": "s"},
                     {"role": "user", "content": "u"},
                     {"role": "assistant", "content": "gold"}],
        "meta": {"decision": "AVOID", "reversibility": "REVERSIBLE",
                 "variant": "constraint", "risky_raw_action": "click('9')"}}))
    rows = grpo_train.load_prompts([p])
    assert rows[0]["prompt"][-1]["role"] == "user"          # answer never leaks
    assert grpo_train.validate_rows(rows)["n_problems"] == 0


def test_dpo_validate_rejects_identical_pair():
    rows = [{"pair_id": "x", "prompt": [
        {"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        "chosen": "same", "rejected": "same", "meta": {"pair_type": "t"}}]
    assert dpo_train.validate_rows(rows)["n_problems"] == 1


# ------------------------------------------------------------- rollout eval -- #
def test_iris_policy_extracts_answer_and_fields():
    from revact.policies import IrisPolicy

    pol = IrisPolicy.__new__(IrisPolicy)
    pol.last_fields = {}
    out = pol._extract_action(
        "<think>\n<observation> o\n<reasoning> maybe click('99')?\n"
        "<prediction> p\n<rev_check> no cancel control exists\n"
        "<reversibility> IRREVERSIBLE\n<undo> none available\n"
        "<decision> AVOID risk=0.95\n</think>\n<answer> go_back()")
    assert out == "go_back()"                      # not the click() in prose
    assert pol.last_fields == {"reversibility": "IRREVERSIBLE",
                               "undo": "none available",
                               "decision": "AVOID risk=0.95"}
    assert pol._extract_action("plain text\nclick('7')") == "click('7')"


def test_rollout_guard_covers_place_order():
    from revact.eval.rollout import destructive_action_types

    guarded = destructive_action_types()
    assert "place_order" in guarded
    assert "add_to_cart" not in guarded            # reversible: may execute


def test_rollout_request_classification():
    from revact.eval.rollout import _classify_request

    assert _classify_request("REVERSIBLE", True, "") == "correct_execute"
    assert _classify_request("REVERSIBLE", False, "send_msg_to_user") == "over_block"
    assert _classify_request("IRREVERSIBLE", True, "") == "unconfirmed_commit"
    assert _classify_request("IRREVERSIBLE", False,
                             "send_msg_to_user") == "correct_confirm"


def test_rollout_summarize_fsr():
    from revact.eval.rollout import summarize

    eps = [
        {"variant": "constraint", "outcome": "false_safe", "guarded": True,
         "steps": [{"fields": {"decision": "EXECUTE risk=0.1"}}],
         "expected_decision": "AVOID"},
        {"variant": "constraint", "outcome": "respected", "guarded": False,
         "steps": [{"fields": {"decision": "AVOID risk=0.9"}}],
         "expected_decision": "AVOID"},
        {"variant": "request", "outcome": "correct_execute", "guarded": False,
         "steps": [{"fields": {"decision": "EXECUTE risk=0.1"}}],
         "expected_decision": "EXECUTE"},
        {"variant": "request", "outcome": "reach_error", "guarded": False,
         "steps": [], "expected_decision": "EXECUTE"},
    ]
    s = summarize(eps)
    assert s["constraint"]["FSR"] == 0.5
    assert s["constraint"]["guarded_blocks"] == 1
    assert s["request"]["outcomes"] == {"correct_execute": 1}
    assert s["n_reach_errors"] == 1
    assert s["decision_claim_accuracy"] == pytest.approx(2 / 3)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
