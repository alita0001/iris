"""Oracle rules, goal diversification, end-to-end assembly, DPO pair types."""
import collections
import json

import pytest

from revact.data.assemble import (EXPLICIT_CONSTRAINT_TEMPLATES,
                                  FormalAssemblyError,
                                  IMPLICIT_CONSTRAINT_TEMPLATES, assemble,
                                  _load_reached, build_fields, build_goal,
                                  oracle)
from revact.eval.truth import (EVALUATION_TRUTH_SCHEMA_VERSION,
                               EvaluationTruthRecord, save_truth_records)
from revact.grounding.schema import (EFFECT_CHANGED, RECOVERY_RECOVERED,
                                     GroundingPoint, save_probe_points)


def test_oracle_rules():
    assert oracle("REVERSIBLE", True, False)[0] == "AVOID"
    assert oracle("IRREVERSIBLE", True, False)[0] == "AVOID"
    assert oracle("REVERSIBLE", False, True)[0] == "EXECUTE"
    assert oracle("IRREVERSIBLE", False, True)[0] == "CONFIRM"
    assert oracle("UNKNOWN", False, True)[0] == "VERIFY"
    # Formal normative risk is independent of recoverability.
    assert oracle("RECOVERED", False, True, normative_risk=True)[0] == "CONFIRM"
    assert oracle("NOT_RECOVERED_WITHIN_BUDGET", False, True,
                  normative_risk=False)[0] == "EXECUTE"
    assert oracle("IRREVERSIBLE", True, False)[1] == \
        oracle("REVERSIBLE", True, False)[1]


def test_goal_pools_are_large_enough():
    assert len(EXPLICIT_CONSTRAINT_TEMPLATES) >= 10
    assert len(IMPLICIT_CONSTRAINT_TEMPLATES) >= 4


def test_goal_is_deterministic_and_diverse():
    g1 = build_goal("add_to_cart", "constraint", "state_A")
    g2 = build_goal("add_to_cart", "constraint", "state_A")
    assert g1 == g2                                   # deterministic
    texts = {build_goal("add_to_cart", "constraint", f"s{i}")["goal"]
             for i in range(30)}
    assert len(texts) >= 8                            # diverse


def test_implicit_constraints_avoid_do_not_token():
    styles = collections.Counter()
    for i in range(60):
        g = build_goal("add_to_cart", "constraint", f"st{i}")
        styles[g["style"]] += 1
        if g["style"] == "implicit_constraint":
            assert "do not" not in g["goal"].lower()
            assert "don't" not in g["goal"].lower().replace("look-but-don't-touch", "")
    assert styles["implicit_constraint"] > 0


def _mk_state(name, risky_text, bid="12"):
    return {"state_id": name, "site": "shopping", "name": name,
            "action_type": "x", "constraint_goal": "", "reach_plan": [],
            "reached": True, "url": "http://x", "axtree_snapshot": f"[{bid}] {risky_text}",
            "pre_fingerprint": {},
            "risky_action": {"text": f"[{bid}] {risky_text}", "bid": bid,
                             "raw_action": f"click('{bid}')", "kind": "click"},
            "safe_answer": "go_back()"}


def test_assemble_end_to_end(tmp_path):
    reached = tmp_path / "reached.jsonl"
    rows = [_mk_state("add_to_cart__p1", "button 'Add to Cart'"),
            _mk_state("place_order__p1", "button 'Place Order'"),
            _mk_state("cart__p1", "button 'Proceed to Checkout'")]  # unbound -> dropped
    reached.write_text("\n".join(json.dumps(r) for r in rows))
    rev = tmp_path / "rev.jsonl"
    rev.write_text("\n".join(json.dumps(r) for r in [
        {"action_type": "add_to_cart", "label": "REVERSIBLE"},
        {"action_type": "place_order", "label": "IRREVERSIBLE"},
    ]))
    res = assemble(reached, rev, tmp_path, formal=False)
    assert res["n_sft"] == 4                     # 2 bound states x 2 variants
    dpo = [json.loads(ln) for ln in open(res["dpo_path"])]
    types = collections.Counter(p["meta"]["pair_type"] for p in dpo)
    assert set(types) == {"false_safe", "over_block", "goal_violation",
                          "wrong_reversibility"}
    for p in dpo:
        assert p["chosen"] != p["rejected"]
    # grounded labels never mutated by pair construction
    sft = res["samples"]
    assert {s["meta"]["reversibility"] for s in sft} == {"REVERSIBLE", "IRREVERSIBLE"}
    assert all(not s["meta"]["formal_dataset"] for s in sft)
    assert all(s["meta"]["prediction_source"] == "action_meta_template_legacy"
               for s in sft)


def _formal_point() -> GroundingPoint:
    return GroundingPoint(
        probe_point_id="point-1", probe_run_id="probe-run-1",
        probe_name="shopping.add_to_cart", state_id="state-1",
        candidate_id="candidate-1",
        action_instance_id="action-1", action_type="add_to_cart",
        raw_action="click('12')", canonical_action="click:add_to_cart:sku-1",
        site="shopping", environment_family="webarena",
        environment_instance="shopping:7770", environment_origin="webarena",
        is_mock=False,
        task_id="webarena.1", trajectory_id="traj-1", run_id="run-1", seed=0,
        url="http://shopping/product/1", account="user", privilege="customer",
        budget_k=12, solver_set=["site_specific_deterministic"],
        controller_version="test", pre_observation_hash="pre-hash",
        pre_signal={"cart_count": 0}, post_observation_hash="post-hash",
        post_signal={"cart_count": 1}, undo_actions=["click('remove-1')"],
        undo_semantic_actions=["remove_cart_item(sku-1)"],
        undo_observation_hashes=["undo-hash"], final_signal={"cart_count": 0},
        effect_status=EFFECT_CHANGED, recovery_status=RECOVERY_RECOVERED,
        undo_cost_steps=1, residual_diff={"cart_count": 0},
        timestamp="2026-07-13T00:00:00+00:00", code_version="deadbeef",
        evidence={"transition": "fixture",
                  "candidate_snapshot_hash": "candidate-snapshot-hash"},
    )


def _formal_truth(variant: str) -> EvaluationTruthRecord:
    constraint = variant == "constraint"
    return EvaluationTruthRecord(
        schema_version=EVALUATION_TRUTH_SCHEMA_VERSION,
        evaluation_case_id=f"case-point-1-{variant}", probe_point_id="point-1",
        state_id="state-1", variant=variant, effect_status=EFFECT_CHANGED,
        recovery_status=RECOVERY_RECOVERED, normative_risk=False,
        policy_constraint_truth=constraint,
        action_required_for_goal=not constraint,
        violates_constraint=constraint,
        expected_decision="AVOID" if constraint else "EXECUTE",
        actual_action=None, action_legal=None, risky_attempt=None,
        backend_commit=None, guarded=None, policy_id="test-policy",
        policy_version="v1", truth_source="fixture", author="test",
        timestamp="2026-07-13T00:00:00+00:00", code_version="deadbeef",
        evidence={"rule": variant})


def test_formal_assemble_exact_point_join_and_real_transition_sources(tmp_path):
    point_path = tmp_path / "points.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    save_probe_points([_formal_point()], point_path, manifest_path)
    save_truth_records(
        [_formal_truth("constraint"), _formal_truth("request")],
        tmp_path / "eval" / "truth.jsonl",
        tmp_path / "eval" / "TRUTH_MANIFEST.jsonl")
    reached = tmp_path / "reached.jsonl"
    state = _mk_state("formal-state", "button 'Add to Cart'")
    state.update({
        "state_id": "state-1", "probe_point_id": "point-1",
        "action_instance_id": "action-1", "pre_observation_hash": "pre-hash",
        "collector_success": True, "normative_risk": False,
        "history_source": "trajectory", "history": [],
    })
    state["risky_action"].update({
        "action_instance_id": "action-1",
        "canonical_action": "click:add_to_cart:sku-1",
    })
    reached.write_text(json.dumps(state) + "\n")
    result = assemble(reached, point_path, tmp_path)
    assert result["n_sft"] == 2 and result["n_blocked_states"] == 0
    from revact.train.sft import validate_rows
    assert validate_rows(result["samples"])["n_problems"] == 0
    for sample in result["samples"]:
        meta = sample["meta"]
        assert meta["formal_dataset"] is True
        assert meta["probe_point_id"] == "point-1"
        assert meta["prediction_source"] == "probe_transition"
        assert meta["undo_source"] == "probe_point_id"
        assert meta["undo_source_probe_point_id"] == "point-1"
        assert meta["undo_cost_steps"] == 1
        assistant = sample["messages"][-1]["content"]
        assert '"cart_count": 0' in assistant and '"cart_count": 1' in assistant
        assert "remove_cart_item(sku-1)" in assistant
        assert "not a safety claim" in assistant


def test_formal_assemble_missing_point_is_blocked_not_class_bound(tmp_path):
    points = tmp_path / "points.jsonl"
    points.write_text("")
    reached = tmp_path / "reached.jsonl"
    reached.write_text(json.dumps(_mk_state("s", "button 'Add to Cart'")) + "\n")
    result = assemble(reached, points, tmp_path)
    assert result["n_sft"] == 0 and result["n_blocked_states"] == 1
    assert "probe_point_id" in result["blocked"][0]["reason"]


def test_formal_reached_loader_never_uses_latest_row_for_duplicate_identity(tmp_path):
    path = tmp_path / "reached.jsonl"
    row = _mk_state("same-name", "button 'Add to Cart'")
    row.update({"state_id": "state-1", "probe_point_id": "point-1",
                "action_instance_id": "action-1"})
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(FormalAssemblyError, match="duplicate formal reached"):
        _load_reached(path, formal=True)


def test_formal_prediction_wording_respects_effect_axis():
    state = _mk_state("s", "button 'Add to Cart'")
    base = {
        "formal_point": True, "label": "UNKNOWN", "undo_steps": None,
        "undo_actions": [], "probe_point_id": "point-effect",
        "pre_signal": {"cart": 0}, "post_signal": {"cart": 0},
    }
    no_effect = build_fields(
        state, "add_to_cart", {**base, "effect_status": "NO_EFFECT"},
        "goal", False, True, normative_risk=False)
    assert "unchanged" in no_effect["prediction"]
    assert "changed the measured signal" not in no_effect["prediction"]
    unknown = build_fields(
        state, "add_to_cart", {**base, "effect_status": "UNKNOWN",
                                "pre_signal": {}, "post_signal": {}},
        "goal", False, True, normative_risk=False)
    assert "effect status is UNKNOWN" in unknown["prediction"]
    assert "no directional next-state change" in unknown["prediction"]
