"""Oracle rules, goal diversification, end-to-end assembly, DPO pair types."""
import collections
import json

from revact.data.assemble import (EXPLICIT_CONSTRAINT_TEMPLATES,
                                  IMPLICIT_CONSTRAINT_TEMPLATES, assemble,
                                  build_goal, oracle)


def test_oracle_rules():
    assert oracle("REVERSIBLE", True, False)[0] == "AVOID"
    assert oracle("IRREVERSIBLE", True, False)[0] == "AVOID"
    assert oracle("REVERSIBLE", False, True)[0] == "EXECUTE"
    assert oracle("IRREVERSIBLE", False, True)[0] == "CONFIRM"
    assert oracle("UNKNOWN", False, True)[0] == "VERIFY"
    # risk score must scale with reversibility under a constraint
    assert oracle("IRREVERSIBLE", True, False)[1] > oracle("REVERSIBLE", True, False)[1]


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
    res = assemble(reached, rev, tmp_path)
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
