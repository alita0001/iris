"""P2 offline tests: prompt registry (override/fingerprint/validation),
observed-delta history entries, and the <rev_check>/<undo> output fields."""
import json

import pytest

from revact import prompts
from revact.prompt_store import (content_fingerprint, diff_bundles,
                                 load_bundle, load_generation_bundle,
                                 store_bundle, store_generation_bundle)
from revact.data.assemble import build_fields, build_rev_fields, render_assistant
from revact.envs.obs_utils import history_entry, obs_delta


# ----------------------------------------------------------------- registry -- #
def test_defaults_and_module_attrs():
    assert prompts.get("agent_system") == prompts.SYSTEM
    assert prompts.get("collector_system") == prompts.SYSTEM_COLLECTOR
    assert "give_up()" in prompts.get("undo_attacker_system")
    assert "{observation}" in prompts.get("undo_attacker_user")
    assert "effect, recovery, normative_risk" in prompts.get(
        "opinion_rater_system")
    assert all(token in prompts.get("opinion_rater_user") for token in (
        "{goal}", "{observation}", "{action}"))
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


def test_immutable_prompt_bundles_restore_full_text_and_diff(tmp_path):
    first = {"agent_system": "one", "pool": ["a"]}
    second = {"agent_system": "two", "pool": ["a"]}
    p1 = store_bundle(first, root=tmp_path, author="test")
    p2 = store_bundle(second, root=tmp_path, parent_fp=content_fingerprint(first),
                      author="test")
    # Content addressing is idempotent and never rewrites first-writer metadata.
    before = p1.read_bytes()
    assert store_bundle(first, root=tmp_path, author="someone-else") == p1
    assert p1.read_bytes() == before
    assert load_bundle(content_fingerprint(first), root=tmp_path)["prompts"] == first
    diff = diff_bundles(content_fingerprint(first), content_fingerprint(second),
                        root=tmp_path)
    assert diff["n_changed"] == 1
    assert diff["changed"]["agent_system"] == {"before": "one", "after": "two"}
    assert p1 != p2
    with pytest.raises(ValueError):
        load_bundle("../../etc/passwd", root=tmp_path)


def test_generation_bundle_separates_config_and_restores_full_prompt(tmp_path):
    prompt_values = {"agent_system": "one", "pool": ["a"]}
    prompt_path = store_bundle(prompt_values, root=tmp_path, author="test")
    common = {
        "prompts_fp": prompt_path.stem,
        "producer": "test.materializer",
        "model": {"provider": "fixture", "name": "teacher", "revision": "1"},
        "root": tmp_path,
    }
    greedy = store_generation_bundle(
        **common, decode_config={"temperature": 0.0, "top_p": 1.0})
    sampled = store_generation_bundle(
        **common, decode_config={"temperature": 0.7, "top_p": 0.9})

    # Same prompt text remains backwards-compatible while generation identity
    # changes with decoding parameters.
    assert greedy.stem != sampled.stem
    assert content_fingerprint(prompt_values) == prompt_path.stem
    restored = load_generation_bundle(sampled.stem, root=tmp_path)
    assert restored["prompts_fp"] == prompt_path.stem
    assert restored["decode_config"] == {"temperature": 0.7, "top_p": 0.9}
    assert restored["prompt_bundle"]["prompts"] == prompt_values

    tampered = json.loads(sampled.read_text(encoding="utf-8"))
    tampered["decode_config"]["temperature"] = 0.1
    sampled.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity check"):
        load_generation_bundle(sampled.stem, root=tmp_path)

    # The old API must not silently discard a model configuration.
    with pytest.raises(ValueError, match="generation provenance"):
        store_bundle(prompt_values, root=tmp_path,
                     model_config={"temperature": 1.0})


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


def test_obs_delta_does_not_treat_order_text_navigation_as_mutation():
    listing = _view(
        "http://x/history?p=3",
        "RootWebArea 'Orders'\n[10] link 'Order #178'\n[11] link 'Order #179'",
    )
    detail = _view(
        "http://x/view/order_id/178/",
        "RootWebArea 'Order #178'\nStaticText 'Order #178'",
    )
    delta = obs_delta(listing, detail)
    assert delta["flag"] == "nav"
    assert "new order" not in delta["delta"]


def test_obs_delta_accepts_explicit_persistent_probe_signal():
    pre = _view("http://x/p", "RootWebArea 'P'")
    post = _view("http://x/p", "RootWebArea 'P changed'")
    pre["persistent_signal"] = {"subscribed": False}
    post["persistent_signal"] = {"subscribed": True}
    delta = obs_delta(pre, post)
    assert delta == {
        "flag": "state-change", "delta": "persistent site signal changed"}


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
    assert undo_irrev == "not found within the recorded budget"
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
        "safe_answer": "go_back()",
        "axtree_snapshot": "[3] button 'Add to Cart'"}))
    rev = tmp_path / "rev.jsonl"
    rev.write_text(json.dumps({
        "action_type": "add_to_cart", "label": "REVERSIBLE",
        "grounding": "cart_item_count", "destructive": False,
        "evidence": {"undo_steps": 1, "undo_actions": ["click('45')"],
                     "residual_diff": {"count_delta": 0}}}))
    rep = assemble(reached, rev, tmp_path, formal=False)
    rows = [json.loads(ln) for ln in
            open(rep["sft_path"])]
    assert rep["n_sft"] == 2 and len(rows) == 2
    for r in rows:
        assert r["meta"]["format"] == "iris.v2-legacy"
        assert r["meta"]["formal_dataset"] is False
        assert r["meta"]["prompts_fp"] == prompts.fingerprint()
        assert load_bundle(r["meta"]["prompts_fp"], root=tmp_path)["prompts"] \
            == prompts.effective()
        generated_with = load_generation_bundle(
            r["meta"]["prompt_generation_fp"], root=tmp_path)
        assert generated_with["prompts_fp"] == r["meta"]["prompts_fp"]
        assert generated_with["producer"] == "revact.data.assemble"
        assert generated_with["prompt_bundle"]["prompts"] == prompts.effective()
        assert r["meta"]["undo_cost_steps"] == 1
        asst = r["messages"][-1]["content"]
        assert "<rev_check>" in asst and "<undo>" in asst


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
