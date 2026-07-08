"""P2 offline tests: prompt registry (override/fingerprint/validation),
observed-delta history entries, and the <rev_check>/<undo> output fields."""
import json

import pytest

from revact import prompts
from revact.data.assemble import build_fields, build_rev_fields, render_assistant
from revact.envs.obs_utils import history_entry, obs_delta


# ----------------------------------------------------------------- registry -- #
def test_defaults_and_module_attrs():
    assert prompts.get("agent_system") == prompts.SYSTEM
    assert prompts.get("collector_system") == prompts.SYSTEM_COLLECTOR
    assert "<rev_check>" in prompts.SYSTEM and "<undo>" in prompts.SYSTEM
    assert len(prompts.get_list("explicit_constraint_templates")) >= 10


def test_override_roundtrip_and_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setenv("REVACT_PROMPTS_FILE", str(tmp_path / "p.json"))
    fp0 = prompts.fingerprint()
    prompts.set_override("collector_system", "Collector v2 rules.")
    assert prompts.get("collector_system") == "Collector v2 rules."
    assert prompts.SYSTEM_COLLECTOR == "Collector v2 rules."   # attr stays live
    assert prompts.fingerprint() != fp0
    prompts.clear_override("collector_system")
    assert prompts.fingerprint() == fp0
    # list-kind override drives build_goal through the registry
    prompts.set_override("request_templates", ["Kindly {verb} right away."])
    from revact.data.assemble import build_goal
    g = build_goal("add_to_cart", "request", "any-state")
    assert g["goal"] == "Kindly add this item to the cart right away."


def test_override_validation():
    assert prompts.validate_override("nope", "x") is not None
    assert prompts.validate_override("agent_system", "") is not None
    assert prompts.validate_override("teacher_distill", "no placeholders") is not None
    assert prompts.validate_override("request_templates", []) is not None
    assert prompts.validate_override("request_templates", ["do {verb}"]) is None
    with pytest.raises(ValueError):
        prompts.set_override("agent_system", "")


def test_corrupt_overrides_file_falls_back(tmp_path, monkeypatch):
    f = tmp_path / "p.json"
    f.write_text("{not json")
    monkeypatch.setenv("REVACT_PROMPTS_FILE", str(f))
    assert prompts.get("agent_system").startswith("You are a safe web agent")


# ------------------------------------------------------------ delta history -- #
def _view(url, tree, backend=None):
    v = {"url": url, "axtree_txt": tree}
    if backend is not None:
        v["backend_state"] = backend
    return v


def test_obs_delta_flags():
    home = _view("http://x/", "RootWebArea 'Home'\n[1] link 'Shop'")
    shop = _view("http://x/shop", "RootWebArea 'Shop'\n[2] link 'Item'")
    assert obs_delta(home, shop)["flag"] == "nav"
    assert "Shop" in obs_delta(home, shop)["delta"]
    # nothing changed -> no-effect (the anti-looping signal)
    assert obs_delta(home, home) == {"flag": "no-effect",
                                     "delta": "no visible change"}
    # same URL, different content -> update
    menu = _view("http://x/", "RootWebArea 'Home'\n[1] link 'Shop'\n[9] menu 'open'")
    assert obs_delta(home, menu)["flag"] == "update"
    # backend signal moved -> state-change with the entity delta
    pre = _view("http://x/p", "RootWebArea 'P'", backend={"cart": [], "orders": []})
    post = _view("http://x/p", "RootWebArea 'P'", backend={"cart": ["i1"], "orders": []})
    d = obs_delta(pre, post)
    assert d["flag"] == "state-change" and "cart items 0 -> 1" in d["delta"]


def test_history_entry_and_render():
    pre = _view("http://x/p", "RootWebArea 'P'")
    e = history_entry("click('5')", pre, pre)
    assert e == {"action": "click('5')", "flag": "no-effect",
                 "delta": "no visible change"}
    block = prompts.history_block([e])
    assert block == "1. click('5') -> [no-effect] no visible change"


# ----------------------------------------------------------- rev_check/undo -- #
def _state():
    return {"name": "s", "risky_action": {"text": "[3] button 'Add to Cart'",
                                          "raw_action": "click('3')"},
            "safe_answer": "go_back()"}


def test_rev_fields_consume_grounded_evidence():
    info = {"label": "REVERSIBLE", "undo_steps": 2,
            "undo_actions": ["click('a')", "click('b')"],
            "residual_diff": {"count_delta": 0}}
    rev_check, undo = build_rev_fields("add_to_cart", info)
    assert "2 steps" in undo and "remove the item from the cart" in undo
    assert "2 steps" in rev_check
    _, undo_irrev = build_rev_fields("place_order", {"label": "IRREVERSIBLE"})
    assert undo_irrev == "none available"
    rc_unknown, undo_unknown = build_rev_fields("address_add", {"label": "UNKNOWN"})
    assert undo_unknown == "unverified" and "cannot verify" in rc_unknown


def test_assistant_has_reasoning_before_label():
    f = build_fields(_state(), "add_to_cart",
                     {"label": "REVERSIBLE", "undo_steps": 1,
                      "undo_actions": ["click('45')"], "residual_diff": None},
                     "Please add this item to the cart.", False, True)
    asst = render_assistant(f)
    # inverse world model: mechanism reasoning precedes the discrete label,
    # and the constructed plan follows it
    assert asst.index("<rev_check>") < asst.index("<reversibility>") \
        < asst.index("<undo>") < asst.index("<decision>")
    assert "(1 step)" in asst
    # legacy callers may still pass a bare label
    f2 = build_fields(_state(), "add_to_cart", "REVERSIBLE", "goal", True, False)
    assert f2["undo_steps"] is None and "<rev_check>" in render_assistant(f2)


def test_assemble_meta_carries_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("REVACT_PROMPTS_FILE", str(tmp_path / "p.json"))
    from revact.data.assemble import assemble

    reached = tmp_path / "reached.jsonl"
    reached.write_text(json.dumps({
        "name": "add_to_cart__prod-a", "reached": True, "url": "http://x/p",
        "risky_action": {"text": "[3] button 'Add to Cart'",
                         "raw_action": "click('3')"},
        "safe_answer": "go_back()", "axtree_snapshot": "tree"}))
    rev = tmp_path / "rev.jsonl"
    rev.write_text(json.dumps({
        "action_type": "add_to_cart", "label": "REVERSIBLE",
        "grounding": "cart_item_count", "destructive": False,
        "evidence": {"undo_steps": 1, "undo_actions": ["click('45')"],
                     "residual_diff": {"count_delta": 0}}}))
    rep = assemble(reached, rev, tmp_path)
    rows = [json.loads(ln) for ln in
            (tmp_path / "train" / "sft" / "revact_sft.jsonl").open()]
    assert rep["n_sft"] == 2 and len(rows) == 2
    for r in rows:
        assert r["meta"]["format"] == "iris.v2"
        assert r["meta"]["prompts_fp"] == prompts.fingerprint()
        assert r["meta"]["undo_steps"] == 1
        asst = r["messages"][-1]["content"]
        assert "<rev_check>" in asst and "<undo>" in asst


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
