"""Dataset explorer builder: payload assembly + self-contained rendering."""
import json
import re

from revact.viz.build import collect_payload, render_html


def _seed_data(root):
    (root / "grounded").mkdir(parents=True)
    (root / "grounded" / "reversibility.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"action_type": "add_to_cart", "label": "REVERSIBLE", "grounding": "cart_item_count",
         "destructive": False, "probe_id": "p1", "timestamp": "t1", "probe_name": "shopping.add_to_cart",
         "evidence": {"undo_steps": 1, "undo_actions": ["click('45')"]}},
        {"action_type": "place_order", "label": "UNKNOWN", "grounding": "order_set",
         "destructive": False, "evidence": {"reason": "dry-run"}},
    ]))
    raw = root / "raw"
    (raw / "trajectories").mkdir(parents=True)
    (raw / "trajectories" / "t1.jsonl").write_text(json.dumps(
        {"task_id": "mock.1", "trajectory_id": "t1", "step_id": 1, "action": "click('11')",
         "url_after": "http://x", "reward": 1.0, "obs_after_axtree": "RootWebArea 'X'",
         "screenshot": ""}))
    (raw / "trajectories_meta.jsonl").write_text(json.dumps(
        {"trajectory_id": "t1", "success": True, "max_reward": 1.0, "n_steps": 1}))
    (raw / "state_bank").mkdir(parents=True)
    (raw / "state_bank" / "pilot_reached_states.jsonl").write_text(json.dumps(
        {"name": "add_to_cart", "action_type": "add_to_cart", "reached": True,
         "url": "http://x/p", "risky_action": {"text": "[3] button 'Add to Cart'"},
         "safe_answer": "go_back()", "axtree_snapshot": "tree </script> injection"}))
    tr = root / "train"
    (tr / "sft").mkdir(parents=True)
    (tr / "sft" / "revact_sft.jsonl").write_text(json.dumps({
        "sample_id": "add_to_cart__p1__request",
        "messages": [{"role": "system", "content": "s"},
                     {"role": "user", "content": "<goal>\nPlease add it.\n\n<observation>\n[3] button 'Add to Cart'\n"},
                     {"role": "assistant", "content": "<think>\n<observation> x\n<reversibility> REVERSIBLE\n<decision> EXECUTE risk=0.1\n</think>\n<answer> click('3')"}],
        "meta": {"action_type": "add_to_cart", "variant": "request", "decision": "EXECUTE",
                 "reversibility": "REVERSIBLE", "constraint_style": "request"}}))
    (tr / "dpo").mkdir(parents=True)
    (tr / "dpo" / "revact_dpo.jsonl").write_text(json.dumps({
        "pair_id": "add_to_cart__p1__request__over_block",
        "prompt": [{"role": "system", "content": "s"},
                   {"role": "user", "content": "<goal>\nPlease add it.\n\n<observation>\nobs\n"}],
        "chosen": "<decision> EXECUTE", "rejected": "<decision> AVOID",
        "meta": {"pair_type": "over_block", "action_type": "add_to_cart", "variant": "request"}}))
    (tr / "splits").mkdir(parents=True)
    (tr / "splits" / "sft_test.jsonl").write_text("")


def test_payload_counts(tmp_path):
    _seed_data(tmp_path)
    p = collect_payload(data_root=tmp_path)
    o = p["overview"]
    assert o["n_sft"] == 1 and o["n_dpo"] == 1 and o["n_states"] == 1
    assert o["n_grounded_runs"] == 2 and o["n_classes"] == 2
    # dry-run UNKNOWN never displaces a grounded label
    assert o["effective_labels"]["add_to_cart"] == "REVERSIBLE"
    assert p["sft"][0]["goal"] == "Please add it."
    assert p["sft"][0]["split"] == "train"


def test_render_is_self_contained_and_parseable(tmp_path):
    _seed_data(tmp_path)
    p = collect_payload(data_root=tmp_path)
    html = render_html(p)
    assert html.startswith("<!DOCTYPE html>") and html.rstrip().endswith("</html>")
    # no external requests (self-contained; also required by the Artifact CSP)
    assert not re.search(r'(src|href)="https?://', html)
    # embedded JSON survives round-trip despite </script>-bearing content
    m = re.search(r'<script type="application/json" id="viz-data">(.*?)</script>',
                  html, re.DOTALL)
    data = json.loads(m.group(1).replace("<\\/", "</"))
    assert data["overview"]["n_sft"] == 1
    assert "</script> injection" in data["states"][0]["axtree"]
    for tab in ("tab-ov", "tab-g", "tab-t", "tab-s", "tab-q", "tab-p"):
        assert f'id="{tab}"' in html


def test_body_only_variant(tmp_path):
    _seed_data(tmp_path)
    html = render_html(collect_payload(data_root=tmp_path), full_document=False)
    assert not html.startswith("<!DOCTYPE")
    assert "<html" not in html.split("<script", 1)[0]
    assert html.startswith("<title>")
