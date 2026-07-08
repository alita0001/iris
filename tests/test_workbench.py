"""Workbench server: loaders, annotations, adapters, quality, export, HTTP.

All offline: a seeded tmp data root + monkeypatched config paths; no live env,
no key, no subprocess jobs (only in-process actions are exercised here).
"""
import json
import threading
import urllib.error
import urllib.request

import pytest

from revact import config
from revact.server import adapters, annotations
from revact.server.datasets import DataStore
from revact.server.export import export_dataset
from revact.server.quality import compute_quality

SYS = "sys"
ASST = ("<think>\n<observation> on page\n<reasoning> fits goal\n"
        "<prediction> adds item\n<rev_check> a remove control exists\n"
        "<reversibility> REVERSIBLE\n<undo> remove the item (1 step)\n"
        "<decision> EXECUTE risk=0.1\n</think>\n<answer> click('3')")


def seed(root):
    (root / "grounded").mkdir(parents=True)
    (root / "grounded" / "reversibility.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"action_type": "add_to_cart", "label": "REVERSIBLE", "grounding": "cart_item_count",
         "destructive": False, "probe_id": "p1", "timestamp": "t1",
         "probe_name": "shopping.add_to_cart",
         "evidence": {"baseline": 0, "after_action": 1, "after_undo": 0,
                      "undo_steps": 1, "undo_actions": ["click('45')"]}},
        {"action_type": "add_to_cart", "label": "UNKNOWN", "grounding": "cart_item_count",
         "destructive": False, "probe_id": "p2", "timestamp": "t2",
         "evidence": {"reason": "dry-run"}},
    ]))
    (root / "grounded" / "MANIFEST.jsonl").write_text(json.dumps(
        {"probe_id": "p1", "controller_version": "test"}))
    raw = root / "raw"
    (raw / "trajectories").mkdir(parents=True)
    (raw / "trajectories" / "t1.jsonl").write_text(json.dumps(
        {"task_id": "mock.1", "trajectory_id": "t1", "step_id": 1, "action": "click('11')",
         "url_after": "http://x", "reward": 1.0, "obs_after_axtree": "RootWebArea 'X'",
         "screenshot": ""}))
    (raw / "trajectories_meta.jsonl").write_text(json.dumps(
        {"trajectory_id": "t1", "task_id": "mock.1", "success": True,
         "max_reward": 1.0, "n_steps": 1}))
    (raw / "state_bank").mkdir(parents=True)
    (raw / "state_bank" / "shopping_key_states.jsonl").write_text(json.dumps(
        {"state_id": "t1_s0", "task_id": "mock.1", "trajectory_id": "t1", "step_id": 0,
         "goal": "g", "afforded_action_types": ["add_to_cart"], "replay_prefix": [],
         "url": "http://x", "axtree_snapshot": "tree"}))
    (raw / "state_bank" / "pilot_reached_states.jsonl").write_text(json.dumps(
        {"state_id": "s1", "name": "add_to_cart__prod-a", "action_type": "add_to_cart",
         "reached": True, "url": "http://x/p",
         "risky_action": {"text": "[3] button 'Add to Cart'", "raw_action": "click('3')"},
         "safe_answer": "go_back()", "axtree_snapshot": "tree"}))
    tr = root / "train"
    (tr / "sft").mkdir(parents=True)
    sft_rows = []
    for name, variant, dec in [("add_to_cart__prod-a", "request", "EXECUTE"),
                               ("add_to_cart__prod-b", "constraint", "AVOID")]:
        sft_rows.append({
            "sample_id": f"{name}__{variant}",
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user",
                          "content": "<goal>\nPlease add it.\n\n<observation>\nobs\n"},
                         {"role": "assistant", "content": ASST}],
            "meta": {"action_type": "add_to_cart", "variant": variant, "decision": dec,
                     "reversibility": "REVERSIBLE", "constraint_style": variant,
                     "goal_template": "request:0"}})
    (tr / "sft" / "revact_sft.jsonl").write_text(
        "\n".join(json.dumps(r) for r in sft_rows))
    distilled = json.loads(json.dumps(sft_rows[0]))
    distilled["messages"][2]["content"] = ASST.replace("on page", "teacher words")
    distilled["meta"]["prose_source"] = "teacher"
    (tr / "sft" / "revact_sft_distilled.jsonl").write_text(json.dumps(distilled))
    (tr / "dpo").mkdir(parents=True)
    (tr / "dpo" / "revact_dpo.jsonl").write_text(json.dumps({
        "pair_id": "add_to_cart__prod-a__request__over_block",
        "prompt": [{"role": "system", "content": SYS},
                   {"role": "user", "content": "<goal>\nPlease add it.\n\n<observation>\nobs\n"}],
        "chosen": ASST, "rejected": ASST.replace("EXECUTE", "AVOID"),
        "meta": {"pair_type": "over_block", "action_type": "add_to_cart",
                 "variant": "request", "reversibility": "REVERSIBLE"}}))
    (tr / "splits").mkdir(parents=True)
    (tr / "splits" / "sft_train.jsonl").write_text(json.dumps(sft_rows[0]))
    (tr / "splits" / "sft_test.jsonl").write_text(json.dumps(sft_rows[1]))
    (tr / "splits" / "dpo_train.jsonl").write_text(
        (tr / "dpo" / "revact_dpo.jsonl").read_text())


@pytest.fixture
def root(tmp_path, monkeypatch):
    seed(tmp_path)
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(adapters.MANAGER, "jobs_dir", tmp_path / "jobs")
    monkeypatch.setattr(adapters.MANAGER, "index_path", tmp_path / "jobs.jsonl")
    return tmp_path


# ---------------------------------------------------------------- loaders -- #
def test_datastore_joins(root):
    s = DataStore(root)
    assert s.summary()["n_sft"] == 2
    assert s.effective_labels()["add_to_cart"] == "REVERSIBLE"   # dry-run safe
    st = s.reached_states()[0]
    assert st["grounded_action_type"] == "add_to_cart"
    cands = s.candidates_for(st["name"])
    kinds = {c["kind"] for c in cands["candidates"]}
    assert kinds == {"expert_risky", "safe_alternative"}
    assert any(c["pair_type"] == "over_block" for c in cands["counterfactuals"])
    lin = s.lineage("add_to_cart__prod-a__request")
    assert lin["effective_label"] == "REVERSIBLE"
    assert lin["dpo_pairs"][0]["pair_type"] == "over_block"
    assert lin["distilled"]["prose_source"] == "teacher"


def test_sample_raw_and_dataset_card(root):
    s = DataStore(root)
    raw = s.sample_raw("add_to_cart__prod-a__request")
    assert raw["split"] == "train"
    assert raw["sft"]["messages"][2]["content"] == ASST     # unclipped
    assert raw["distilled"]["meta"]["prose_source"] == "teacher"
    assert [p["pair_id"] for p in raw["dpo"]] == [
        "add_to_cart__prod-a__request__over_block"]
    assert raw["n_chars"]["assistant"] == len(ASST)
    assert s.sample_raw("add_to_cart__prod-b__constraint")["split"] == "test"
    assert s.sample_raw("nope") is None
    card = s.dataset_card()
    assert card["summary"]["n_sft"] == 2
    assert card["system_prompt"].startswith("You are a safe web agent")
    assert {f[0] for f in card["dpo_schema"]} >= {"pair_id", "chosen", "rejected"}
    assert card["length_stats"]["assistant"] == {
        "n": 2, "min": len(ASST), "avg": len(ASST), "max": len(ASST)}


# ------------------------------------------------------------ annotations -- #
def test_annotation_overlay_merge(root):
    annotations.add("sample", "x", {"review_status": "needs-review"}, root=root)
    annotations.add("sample", "x", {"note": "later"}, root=root)
    eff = annotations.effective("sample", root)["x"]
    assert eff["review_status"] == "needs-review" and eff["note"] == "later"
    assert eff["_n_rows"] == 2
    with pytest.raises(ValueError):
        annotations.add("nope", "x", {}, root=root)


# --------------------------------------------------------------- adapters -- #
def test_pipeline_overview_and_gating(root, monkeypatch):
    stages = {s["id"]: s for s in adapters.pipeline_overview(DataStore(root))}
    assert stages["collect"]["status"] == "success"
    assert stages["probe"]["status"] == "success"
    assert stages["candidates"]["implemented"] == "placeholder"
    # placeholder action -> explicit extension point, not a fake run
    r = adapters.run_action("candidates", "propose")
    assert not r["ok"] and r.get("placeholder")
    # live-gated action blocks without WA_SHOPPING
    monkeypatch.setattr(config, "WA_SHOPPING", "")
    monkeypatch.setitem(adapters.RUNTIME.settings, "env",
                        {"WA_SHOPPING": "", "WA_SHOPPING_ADMIN": ""})
    r = adapters.run_action("key_states", "reach")
    assert not r["ok"] and r.get("blocked")
    # in-process preview evaluates the real assemble logic
    r = adapters.run_action("counterfactuals", "preview",
                            {"state": "add_to_cart__prod-a"})
    assert r["ok"] and r["result"]["counterfactuals"]


# ---------------------------------------------------------------- quality -- #
def test_quality_report(root):
    q = compute_quality(DataStore(root))
    assert q["volumes"]["sft_samples"] == 2
    assert q["teacher"]["pinned_label_agreement"] == 1.0
    assert q["counterfactual_coverage"]["samples_with_pairs"] == 1
    assert q["n_low_quality"] == 0


# ----------------------------------------------------------------- export -- #
def test_export_applies_overlays(root, monkeypatch):
    out_dir = root / "exports"
    monkeypatch.setattr("revact.server.export.EXPORTS_DIR", out_dir)
    annotations.add("sample", "add_to_cart__prod-b__constraint",
                    {"review_status": "rejected", "note": "bad obs"}, root=root)
    rep = export_dataset(DataStore(root), {"name": "t1", "val_frac": 0.0})
    assert rep["ok"] and rep["n_test"] == 0 and rep["n_excluded"] == 1
    exp = next(out_dir.iterdir())
    got = (exp / "sft_train.jsonl").read_text()
    assert "teacher words" in got          # distilled prose preferred
    dpo_rows = (exp / "dpo_train.jsonl").read_text().strip()
    assert "pair_id" in dpo_rows and "teacher words" not in dpo_rows
    card = (exp / "dataset_card.md").read_text()
    assert "behaviorally measured" in card
    # a human override that contradicts the pinned label EXCLUDES the sample
    annotations.add("grounded", "p1",
                    {"reversibility_override": "IRREVERSIBLE"}, root=root)
    rep2 = export_dataset(DataStore(root), {"name": "t2", "val_frac": 0.0})
    assert rep2["n_train"] == 0
    assert any(e["reason"] == "label_conflict"
               for e in json.loads("[" + ",".join(
                   ln for ln in (out_dir / sorted(p.name for p in out_dir.iterdir())[-1]
                                 / "excluded.jsonl").read_text().splitlines()) + "]"))


# ------------------------------------------------------------------- http -- #
def test_http_roundtrip(root, monkeypatch):
    from http.server import ThreadingHTTPServer
    from revact.server.app import RUNTIME, Handler

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        def get(p):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{p}") as r:
                return json.loads(r.read())

        def post(p, body):
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}{p}", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req) as r:
                return json.loads(r.read())

        assert get("/api/health")["summary"]["n_sft"] == 2
        assert len(get("/api/pipeline")["stages"]) == 10
        assert get("/api/sft")["items"][0]["decision"] == "EXECUTE"
        assert get("/api/lineage?sample=add_to_cart__prod-a__request")["ok"]
        assert get("/api/dataset_card")["card"]["summary"]["n_sft"] == 2
        raw = get("/api/sample_raw?sample=add_to_cart__prod-a__request")
        assert raw["ok"] and raw["raw"]["dpo"][0]["pair_id"].endswith("over_block")
        assert not get("/api/sample_raw?sample=nope")["ok"]
        # config: key value goes to memory, is masked in GET, absent on save
        post("/api/config", {"models": {"teacher": {"api_key": "sk-SECRET",
                                                    "api_key_env": "T_KEY"}}})
        cfg = get("/api/config")
        assert cfg["settings"]["models"]["teacher"]["api_key_set"] is True
        assert "sk-SECRET" not in json.dumps(cfg)
        assert RUNTIME.secrets["T_KEY"] == "sk-SECRET"
        local = root / "wb.json"
        monkeypatch.setattr("revact.server.app.LOCAL_CONFIG_PATH", local)
        post("/api/config/save", {})
        assert "sk-SECRET" not in local.read_text()
        # annotation via HTTP
        post("/api/annotations", {"kind": "key_state", "target_id": "t1_s0",
                                  "payload": {"review_status": "confirmed"}})
        assert get("/api/annotations/key_state")["effective"]["t1_s0"][
            "review_status"] == "confirmed"
        # prompt registry: list, override, fingerprint change, reset
        monkeypatch.setenv("REVACT_PROMPTS_FILE", str(root / "prompts.json"))
        pr = get("/api/prompts")
        assert {i["id"] for i in pr["items"]} >= {
            "agent_system", "collector_system", "teacher_distill"}
        fp0 = pr["fingerprint"]
        r = post("/api/prompts", {"id": "collector_system", "value": "New collector rules."})
        assert r["ok"] and r["fingerprint"] != fp0
        items = {i["id"]: i for i in get("/api/prompts")["items"]}
        assert items["collector_system"]["overridden"]
        assert items["collector_system"]["value"] == "New collector rules."
        # invalid override (missing required placeholder) is rejected with 400
        try:
            post("/api/prompts", {"id": "teacher_distill", "value": "no placeholders"})
            raise AssertionError("expected 400")
        except urllib.error.HTTPError as e:
            assert e.code == 400
        assert post("/api/prompts/reset", {"id": "collector_system"})["ok"]
        assert get("/api/prompts")["fingerprint"] == fp0
        # path traversal is rejected
        try:
            get("/api/screenshot?path=../../etc/passwd")
            raise AssertionError("expected 403")
        except urllib.error.HTTPError as e:
            assert e.code in (403, 404)
    finally:
        httpd.shutdown()
        RUNTIME.secrets.pop("T_KEY", None)
