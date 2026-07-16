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
from revact.data.candidates import FORMAL_CANDIDATE_BODY_NAME
from revact.grounding.schema import GroundingPoint, save_probe_points
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
         "risky_action": {"text": "[3] button 'Add to Cart'", "bid": "3",
                          "raw_action": "click('3')"},
         "safe_answer": "go_back()", "axtree_snapshot": (
             "RootWebArea 'P'\n[3] button 'Add to Cart'\n[4] link 'Back'\n"
             "[5] button 'Continue'\n[6] link 'Help'\n[7] checkbox 'Gift'\n"
             "[8] button 'Delete draft'")}))
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
    assert s.summary()["default_tier"] == "formal"
    assert s.summary()["n_sft"] == 0
    assert s.summary()["n_legacy_sft"] == 2
    assert s.effective_labels()["add_to_cart"] == "REVERSIBLE"   # dry-run safe
    st = s.reached_states()[0]
    assert st["grounded_action_type"] == "add_to_cart"
    cands = s.candidates_for(st["name"])
    assert cands["asset_tier"] == "legacy"
    assert cands["uses_class_level_label"] is True
    kinds = {c["kind"] for c in cands["candidates"]}
    assert cands["s4_status"] == "ready" and len(cands["candidates"]) == 6
    assert "expert_action" in kinds and len(kinds) >= 4
    assert all(c["legal_at_snapshot"] for c in cands["candidates"])
    assert any(c["pair_type"] == "over_block" for c in cands["counterfactuals"])
    lin = s.lineage("add_to_cart__prod-a__request")
    assert lin["asset_tier"] == "legacy"
    assert "display-only" in lin["legacy_notice"]
    assert lin["effective_label_tier"] == "legacy_class_smoke"
    assert lin["effective_label"] == "REVERSIBLE"
    assert lin["dpo_pairs"][0]["pair_type"] == "over_block"
    assert lin["distilled"]["prose_source"] == "teacher"


def test_workbench_split_membership_is_explicit(root):
    path = root / "train" / "sft" / "revact_sft.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    orphan = json.loads(json.dumps(rows[0]))
    orphan["sample_id"] = "add_to_cart__orphan__request"
    path.write_text(path.read_text() + "\n" + json.dumps(orphan) + "\n")
    viewed = {row["sample_id"]: row for row in
              DataStore(root).sft(tier="legacy")}
    assert viewed[orphan["sample_id"]]["split"] == "unassigned"

    split_dir = root / "train" / "splits"
    for side in ("train", "dev"):
        with (split_dir / f"sft_{side}.jsonl").open("a") as handle:
            handle.write("\n" + json.dumps(orphan) + "\n")
    viewed = {row["sample_id"]: row for row in
              DataStore(root).sft(tier="legacy")}
    assert viewed[orphan["sample_id"]]["split"] == "membership_error"
    assert DataStore(root).dpo(tier="legacy")[0]["split"] == "train"


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
    assert card["default_tier"] == "formal"
    assert card["summary"]["n_sft"] == 0
    assert card["legacy_assets"]["sft_single"] == 2
    assert card["system_prompt"].startswith("You are a safe web agent")
    assert {f[0] for f in card["dpo_schema"]} >= {"pair_id", "chosen", "rejected"}
    assert card["grounding_assets"]["formal_point"]["n_points"] == 0
    assert card["grounding_assets"]["legacy_class_smoke"] == {
        "n_rows": 2, "n_manifest": 1, "formal_supervision": False,
        "binding": "action_type latest non-UNKNOWN (display only)"}
    assert card["length_stats"]["assistant"] == {"n": 0}


def test_workbench_browses_formal_single_and_multiturn_without_legacy_merge(root):
    formal = root / "train" / "formal"
    formal.mkdir(parents=True, exist_ok=True)
    legacy = json.loads((root / "train" / "sft" /
                         "revact_sft.jsonl").read_text().splitlines()[0])
    single = json.loads(json.dumps(legacy))
    single["sample_id"] = "formal-single"
    single["meta"].update({"formal_dataset": True, "format": "iris.v3",
                           "probe_point_id": "point-single"})
    multi = json.loads(json.dumps(single))
    multi["sample_id"] = "formal-multiturn"
    multi["meta"].update({"kind": "multiturn",
                          "probe_point_id": "point-multiturn",
                          "history_source": "trajectory"})
    (formal / config.FORMAL_SFT_PATH.name).write_text(json.dumps(single) + "\n")
    (formal / config.FORMAL_MULTITURN_SFT_PATH.name).write_text(
        json.dumps(multi) + "\n")
    multi_distilled = json.loads(json.dumps(multi))
    multi_distilled["meta"].update({
        "prose_source": "teacher", "rev_check_source": "teacher",
        "teacher_qc_status": "passed",
    })
    (formal / config.FORMAL_MULTITURN_DISTILLED_SFT_PATH.name).write_text(
        json.dumps(multi_distilled) + "\n")
    pair = {"pair_id": "formal-single__on-policy", "prompt": single["messages"][:-1],
            "chosen": single["messages"][-1]["content"],
            "rejected": single["messages"][-1]["content"].replace(
                "EXECUTE", "AVOID"), "meta": {"pair_type": "on_policy_error"}}
    (formal / config.FORMAL_DPO_PATH.name).write_text(json.dumps(pair) + "\n")

    store = DataStore(root)
    assert [r["sample_id"] for r in store.sft(tier="formal")] == [
        "formal-single"]
    assert [r["sample_id"] for r in store.sft(
        family="multiturn", tier="formal")] == ["formal-multiturn"]
    assert len(store.sft(family="all", tier="formal")) == 2
    assert [row["sample_id"] for row in store.sft(
        distilled=True, family="all", tier="formal")] == [
            "formal-multiturn"]
    assert len(store.sft(family="all", tier="legacy")) == 2
    assert store.dpo(tier="formal")[0]["asset_tier"] == "formal"
    assert [row["asset_tier"] for row in store.sft(family="all")] == [
        "formal", "formal"]
    assert store.dpo()[0]["asset_tier"] == "formal"
    quality = compute_quality(store)
    assert quality["scope"] == quality["asset_tier"] == "formal"
    assert quality["volumes"]["sft_samples"] == 2
    assert quality["volumes"]["dpo_pairs"] == 1
    assert quality["volumes"]["distilled_samples"] == 1
    assert quality["legacy_assets"]["sft_samples"] == 2
    card = store.dataset_card()
    assert card["summary"]["n_sft"] == 2
    assert card["length_stats"]["assistant"]["n"] == 2
    raw = store.sample_raw("formal-multiturn")
    assert raw["asset_tier"] == "formal" and raw["family"] == "multiturn"
    assert raw["distilled"]["meta"]["prose_source"] == "teacher"


def test_cross_site_formal_candidates_and_complete_point_lineage(root):
    """The formal browser joins immutable point assets, never class labels."""
    state_id = "reddit-state-1"
    candidate_id = "formal-candidate-1"
    point_id = "formal-point-1"
    trajectory_id = "reddit-traj-1"
    snapshot = "RootWebArea 'Forum'\n[12] button 'Subscribe'\n[13] link 'Back'"

    reddit_key = {
        "state_id": state_id, "task_id": "webarena.2", "site": "reddit",
        "trajectory_id": trajectory_id, "step_id": 1,
        "afforded_action_types": ["reddit_subscribe"], "url": "http://reddit/f/x",
        "axtree_snapshot": snapshot, "collector_success": True,
    }
    (root / "raw" / "state_bank" / "reddit_key_states.jsonl").write_text(
        json.dumps(reddit_key) + "\n")
    reached = {
        **reddit_key, "name": "reddit-formal-state", "reached": True,
        "action_type": "reddit_subscribe", "candidate_id": candidate_id,
        "probe_point_id": point_id, "probe_run_id": "probe-run-1",
        "risky_action": {"bid": "12", "raw_action": "click('12')",
                         "text": "[12] button 'Subscribe'",
                         "candidate_id": candidate_id,
                         "probe_point_id": point_id},
    }
    (root / "raw" / "state_bank" /
     "formal_point_reached_states.jsonl").write_text(json.dumps(reached) + "\n")

    candidate_dir = root / "raw" / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate = {
        "schema_version": "iris.candidate.v2", "candidate_id": candidate_id,
        "state_id": state_id, "bid": "12",
        "canonical_action": "click:button:subscribe",
        "category": "expert_action", "source": "expert",
        "legal_at_snapshot": True, "proposer_model": "recorded-expert",
        "proposer_version": "fixture", "snapshot_hash": "snapshot-hash",
    }
    (candidate_dir / FORMAL_CANDIDATE_BODY_NAME).write_text(
        json.dumps(candidate) + "\n")

    point = GroundingPoint(
        probe_point_id=point_id, probe_run_id="probe-run-1",
        probe_name="reddit.subscribe.fixture", state_id=state_id,
        candidate_id=candidate_id, action_instance_id="action-1",
        action_type="reddit_subscribe", raw_action="click('12')",
        canonical_action="click:button:subscribe", site="reddit",
        environment_family="webarena", environment_instance="reddit:9999",
        environment_origin="webarena", task_id="webarena.2",
        trajectory_id=trajectory_id, run_id="run-1", seed=0,
        url="http://reddit/f/x", account="user", privilege="member",
        budget_k=12, solver_set=["deterministic"],
        controller_version="fixture", pre_observation_hash="pre",
        pre_signal={"subscribed": False}, post_observation_hash="post",
        post_signal={"subscribed": True}, undo_actions=["click('12')"],
        undo_semantic_actions=["unsubscribe"],
        undo_observation_hashes=["undo"], final_signal={"subscribed": False},
        effect_status="CHANGED", recovery_status="RECOVERED",
        undo_cost_steps=1, residual_diff={},
        timestamp="2026-07-14T00:00:00+00:00", code_version="fixture",
        evidence={"candidate_snapshot_hash": "snapshot-hash"},
    )
    save_probe_points([point], root / "grounded" / "probe_points.jsonl",
                      root / "grounded" / "POINT_MANIFEST.jsonl", append=False)

    legacy = json.loads((root / "train" / "sft" /
                         "revact_sft.jsonl").read_text().splitlines()[0])
    sample_id = "reddit-formal-state__request"
    formal_row = json.loads(json.dumps(legacy))
    formal_row["sample_id"] = sample_id
    formal_row["meta"].update({
        "formal_dataset": True, "format": "iris.v3", "site": "reddit",
        "action_type": "reddit_subscribe", "state_id": state_id,
        "candidate_id": candidate_id, "probe_point_id": point_id,
        "probe_run_id": "probe-run-1", "trajectory_id": trajectory_id,
        "run_id": "run-1", "effect_status": "CHANGED",
        "recovery_status": "RECOVERED", "undo_cost_steps": 1,
    })
    formal_dir = root / "train" / "formal"
    formal_dir.mkdir(parents=True, exist_ok=True)
    (formal_dir / config.FORMAL_SFT_PATH.name).write_text(
        json.dumps(formal_row) + "\n")
    teacher = json.loads(json.dumps(formal_row))
    teacher["meta"]["prose_source"] = "teacher"
    (formal_dir / config.FORMAL_DISTILLED_SFT_PATH.name).write_text(
        json.dumps(teacher) + "\n")
    split_dir = formal_dir / "splits"
    split_dir.mkdir(exist_ok=True)
    (split_dir / "sft_train.jsonl").write_text(json.dumps(formal_row) + "\n")

    store = DataStore(root)
    assert {row["site"] for row in store.key_states()} == {"shopping", "reddit"}
    formal_states = store.reached_states(tier="formal")
    assert [(row["name"], row["asset_tier"]) for row in formal_states] == [
        ("reddit-formal-state", "formal")]
    formal_candidates = store.candidates_for("reddit-formal-state", tier="formal")
    assert formal_candidates["source_artifact"].endswith(
        FORMAL_CANDIDATE_BODY_NAME)
    assert formal_candidates["uses_class_level_label"] is False
    assert formal_candidates["counterfactuals"] == []
    assert formal_candidates["candidates"][0]["probe_point_id"] == point_id

    lineage = store.lineage(sample_id)
    assert lineage["asset_tier"] == "formal"
    assert all(lineage[key] is not None for key in (
        "state", "candidate", "transition", "probe", "label", "teacher"))
    assert lineage["split"] == "train"
    assert lineage["lineage_integrity"]["complete"] is True
    assert lineage["lineage_integrity"]["split_assigned"] is True
    assert lineage["effective_label_tier"] == "formal_point"
    assert lineage["candidate"]["candidate_id"] == candidate_id
    assert lineage["label"]["recovery_status"] == "RECOVERED"
    assert lineage["legacy_display_label_tier"] == "legacy_class_smoke"


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
    assert stages["candidates"]["implemented"] == "real"
    r = adapters.run_action("candidates", "propose",
                            {"state": "add_to_cart__prod-a"})
    assert r["ok"] and r["result"]["n"] == 6
    assert (root / "raw" / "candidates" / "iris_candidates.v3.jsonl").exists()
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


def test_workbench_keeps_provider_choice_and_routes_judge_configuration():
    cfg = adapters.RuntimeConfig()
    assert cfg.settings["models"]["opinion"]["provider"] == "custom"
    assert cfg.settings["models"]["opinion"]["base_url"] == ""
    cfg.settings["models"].update({
        "policy": {
            "provider": "openrouter", "base_url": "https://router.test/api/v1",
            "model": "vendor/policy", "api_key_env": "ROUTER_KEY",
        },
        "judge": {
            "provider": "openrouter", "base_url": "https://router.test/api/v1",
            "model": "vendor/judge", "api_key_env": "ROUTER_KEY",
            "mode": "openrouter",
        },
        "opinion": {
            "provider": "custom", "base_url": "https://opinion.test/v1",
            "model": "vendor/opinion", "api_key_env": "OPINION_KEY",
            "temperature": 0.0, "top_p": 0.9, "max_tokens": 240,
        },
    })
    cfg.secrets["ROUTER_KEY"] = "fixture-secret"
    cfg.secrets["OPINION_KEY"] = "fixture-opinion-secret"
    env = cfg.merged_env("policy", "judge", "opinion")
    assert env["REVACT_LLM_BASE_URL"] == "https://router.test/api/v1"
    assert env["REVACT_LLM_MODEL"] == "vendor/policy"
    assert env["REVACT_WA_JUDGE_BASE_URL"] == "https://router.test/api/v1"
    assert env["REVACT_WA_JUDGE_MODEL"] == "vendor/judge"
    assert env["REVACT_WA_JUDGE_API_KEY_ENV"] == "ROUTER_KEY"
    assert env["REVACT_WA_JUDGE"] == "openrouter"
    assert env["REVACT_LLM_PROVIDER"] == "openrouter"
    assert env["REVACT_WA_JUDGE_PROVIDER"] == "openrouter"
    assert env["REVACT_OPINION_PROVIDER"] == "custom"
    assert env["REVACT_OPINION_BASE_URL"] == "https://opinion.test/v1"
    assert env["REVACT_OPINION_MODEL"] == "vendor/opinion"
    assert env["REVACT_OPINION_KEY_ENV"] == "OPINION_KEY"
    assert env["REVACT_OPINION_TEMPERATURE"] == "0.0"
    assert env["REVACT_OPINION_TOP_P"] == "0.9"
    assert env["REVACT_OPINION_MAX_TOKENS"] == "240"
    assert env["OPINION_KEY"] == "fixture-opinion-secret"
    assert env["ROUTER_KEY"] == "fixture-secret"
    # Selecting OpenRouter is one option, not a process-wide hard-coded policy.
    cfg.settings["models"]["policy"]["provider"] = "custom"
    assert cfg.settings["models"]["policy"]["provider"] == "custom"


# ---------------------------------------------------------------- quality -- #
def test_quality_report(root):
    q = compute_quality(DataStore(root))
    assert q["scope"] == q["asset_tier"] == "formal"
    assert q["volumes"]["sft_samples"] == 0
    assert q["teacher"]["pinned_label_agreement"] is None
    assert q["counterfactual_coverage"]["samples_with_pairs"] == 0
    assert q["legacy_assets"]["sft_samples"] == 2
    assert q["legacy_assets"]["dpo_pairs"] == 1
    assert q["legacy_assets"]["distilled_samples"] == 1
    assert q["n_low_quality"] == 0


# ----------------------------------------------------------------- export -- #
def test_export_applies_overlays(root, monkeypatch):
    out_dir = root / "exports"
    monkeypatch.setattr("revact.server.export.EXPORTS_DIR", out_dir)
    annotations.add("sample", "add_to_cart__prod-b__constraint",
                    {"review_status": "rejected", "note": "bad obs"}, root=root)
    # These historical fixtures deliberately lack point-level provenance; the
    # overlay behavior is exercised through an explicitly non-formal export.
    rep = export_dataset(DataStore(root), {"name": "t1", "val_frac": 0.0,
                                           "formal": False})
    assert rep["ok"] and rep["n_test"] == 0 and rep["n_excluded"] == 1
    exp = next(out_dir.iterdir())
    got = (exp / "sft_train.jsonl").read_text()
    assert "teacher words" in got          # distilled prose preferred
    dpo_rows = (exp / "dpo_train.jsonl").read_text().strip()
    assert "pair_id" in dpo_rows and "teacher words" not in dpo_rows
    card = (exp / "dataset_card.md").read_text()
    assert "Operational-recoverability" in card
    assert (exp / "provenance.json").exists()
    assert list((exp / "prompt_bundles").glob("*.json"))
    # a human override that contradicts the pinned label EXCLUDES the sample
    annotations.add("grounded", "p1",
                    {"reversibility_override": "IRREVERSIBLE"}, root=root)
    rep2 = export_dataset(DataStore(root), {"name": "t2", "val_frac": 0.0,
                                            "formal": False})
    assert rep2["n_train"] == 0
    assert any(e["reason"] == "label_conflict"
               for e in json.loads("[" + ",".join(
                   ln for ln in (out_dir / sorted(p.name for p in out_dir.iterdir())[-1]
                                   / "excluded.jsonl").read_text().splitlines()) + "]"))


def test_multiturn_browser_and_formal_export_gates(root, monkeypatch):
    sft_dir = root / "train" / "sft"
    dpo_dir = root / "train" / "dpo"
    split_dir = root / "train" / "formal" / "splits"
    split_dir.mkdir(parents=True)

    def mt(sid, tid, origin, is_mock, success):
        return {"sample_id": sid,
                "messages": [{"role": "system", "content": SYS},
                             {"role": "user", "content":
                              "<goal>\nG\n\n<history>\n1. click('1') -> [nav] P\n\n"
                              "<observation>\n[3] button 'Add to Cart'\n"},
                             {"role": "assistant", "content": ASST}],
                "meta": {"kind": "multiturn", "action_type": "add_to_cart",
                         "variant": "request", "decision": "EXECUTE",
                         "reversibility": "REVERSIBLE", "constraint_style": "request",
                         "goal_template": "request:0", "history_source": "trajectory",
                         "trajectory_id": tid, "state_id": tid + "_s1",
                         "canonical_entity_id": tid, "environment_origin": origin,
                         "is_mock": is_mock, "collector_success": success}}

    good = mt("mt__web__s1__add__request", "webarena.1_seed0", "webarena", False, True)
    mock = mt("mt__mock__s1__add__request", "mock.add_seed0", "mock", True, True)
    failed = mt("mt__fail__s1__add__request", "webarena.2_seed0", "webarena", False, False)
    rows = [good, mock, failed]
    (sft_dir / "revact_sft_multiturn.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows))
    (split_dir / "sft_train_multiturn.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows))
    (split_dir / "sft_test_multiturn.jsonl").write_text("")
    pair = {"pair_id": good["sample_id"] + "__wrong", "prompt": good["messages"][:2],
            "chosen": ASST, "rejected": ASST.replace("EXECUTE", "AVOID"),
            "meta": dict(good["meta"], pair_type="wrong_reversibility")}
    (dpo_dir / "revact_dpo_multiturn.jsonl").write_text(json.dumps(pair))
    (split_dir / "dpo_train_multiturn.jsonl").write_text(json.dumps(pair))

    store = DataStore(root)
    assert len(store.sft(family="all", tier="legacy")) == 5
    raw = store.sample_raw(good["sample_id"])
    assert raw["family"] == "multiturn" and len(raw["sft"]["messages"]) == 3

    out_dir = root / "exports-multi"
    monkeypatch.setattr("revact.server.export.EXPORTS_DIR", out_dir)
    rep = export_dataset(store, {"name": "formal", "val_frac": 0.0,
                                 "formal": True, "prefer_distilled": False})
    # A publication export now fails before writing any release when the joint
    # train/dev/test split report is absent or one side is empty.
    assert rep["ok"] is False
    assert "split" in rep["note"]
    assert not out_dir.exists()


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

        health = get("/api/health")["summary"]
        assert health["default_tier"] == "formal"
        assert health["n_sft"] == 0 and health["n_legacy_sft"] == 2
        assert len(get("/api/pipeline")["stages"]) == 10
        grounded = get("/api/grounded")
        assert grounded["formal_point"]["n_points"] == 0
        assert grounded["legacy_class_smoke"]["formal_supervision"] is False
        assert grounded["canonical_schema"]["recovery_status"] == [
            "RECOVERED", "PARTIALLY_RECOVERED",
            "NOT_RECOVERED_WITHIN_BUDGET", "UNKNOWN"]
        default_sft = get("/api/sft")
        assert default_sft["asset_tier"] == "formal"
        assert default_sft["items"] == []
        default_states = get("/api/states")
        assert default_states["asset_tier"] == "formal"
        assert default_states["legacy_display_only"] is False
        assert default_states["items"] == []
        legacy_states = get("/api/states?tier=legacy")
        assert legacy_states["legacy_display_only"] is True
        assert legacy_states["items"][0]["asset_tier"] == "legacy"
        assert not get("/api/candidates?state=add_to_cart__prod-a")["ok"]
        legacy_candidates = get(
            "/api/candidates?state=add_to_cart__prod-a&tier=legacy")
        assert legacy_candidates["ok"]
        assert legacy_candidates["candidates"]["uses_class_level_label"] is True
        assert legacy_candidates["legacy_display_only"] is True
        assert get("/api/sft?tier=legacy")["items"][0]["decision"] == "EXECUTE"
        assert get("/api/dpo")["asset_tier"] == "formal"
        assert get("/api/dpo?tier=legacy")["items"][0]["pair_type"] == "over_block"
        assert get("/api/lineage?sample=add_to_cart__prod-a__request")["ok"]
        card = get("/api/dataset_card")["card"]
        assert card["default_tier"] == "formal"
        assert card["summary"]["n_sft"] == 0
        assert card["legacy_assets"]["sft_single"] == 2
        grounding = get("/api/grounded")
        assert grounding["asset_tier"] == "formal" and grounding["items"] == []
        assert grounding["formal_point"]["n_points"] == 0
        assert grounding["legacy_class_smoke"]["formal_supervision"] is False
        raw = get("/api/sample_raw?sample=add_to_cart__prod-a__request")
        assert raw["ok"] and raw["raw"]["dpo"][0]["pair_id"].endswith("over_block")
        assert not get("/api/sample_raw?sample=nope")["ok"]
        authored = post("/api/probe-specs", {"proposal": {
            "name": "shopping.fixture", "site": "shopping",
            "action_type": "add_to_cart",
            "candidate_id": "candidate-1", "state_id": "state-1",
            "action_instance_id": "action-1", "raw_action": "click('3')",
            "canonical_action": "click:button:add-to-cart",
            "signal_channels": ["ui_structural", "api"],
            "undo_sequences": [["click('9')"]],
            "solver_set": ["site_specific_deterministic"], "budget_k": 12,
            "safety_level": "self_recovering", "author": "test"}})
        assert authored["ok"] and authored["spec"]["fixture_status"] == "PENDING"
        assert authored["artifact"] == str(
            root / "grounded" / "specs" / "authored_specs.jsonl")
        assert not (root / "grounded" / "probe_specs").exists()
        specs = get("/api/probe-specs")
        assert specs["label_entry_supported"] is False and len(specs["items"]) == 1
        assert specs["artifact"] == authored["artifact"]
        try:
            post("/api/probe-specs", {"label": "RECOVERED", "proposal": {}})
            raise AssertionError("expected label input to be rejected")
        except urllib.error.HTTPError as e:
            assert e.code == 400
        assert len(get("/api/probe-specs")["items"]) == 1
        # config: key value goes to memory, is masked in GET, absent on save
        post("/api/config", {"models": {"teacher": {"api_key": "sk-SECRET",
                                                    "api_key_env": "T_KEY"}}})
        post("/api/config", {"models": {"opinion": {
            "provider": "custom", "base_url": "https://opinion.test/v1",
            "model": "vendor/opinion", "api_key": "fixture-opinion-key",
            "api_key_env": "OPINION_KEY"}}})
        cfg = get("/api/config")
        assert cfg["settings"]["models"]["teacher"]["api_key_set"] is True
        assert cfg["settings"]["models"]["opinion"]["api_key_set"] is True
        assert "sk-SECRET" not in json.dumps(cfg)
        assert "fixture-opinion-key" not in json.dumps(cfg)
        assert RUNTIME.secrets["T_KEY"] == "sk-SECRET"
        assert RUNTIME.secrets["OPINION_KEY"] == "fixture-opinion-key"
        local = root / "wb.json"
        monkeypatch.setattr("revact.server.app.LOCAL_CONFIG_PATH", local)
        post("/api/config/save", {})
        assert "sk-SECRET" not in local.read_text()
        assert "fixture-opinion-key" not in local.read_text()
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
        changed_fp = r["fingerprint"]
        bundle = get("/api/prompts/bundle?fp=" + changed_fp)
        assert bundle["bundle"]["prompts"]["collector_system"] == "New collector rules."
        diff = get(f"/api/prompts/diff?left={fp0}&right={changed_fp}")
        assert diff["changed"]["collector_system"]["before"] != \
            diff["changed"]["collector_system"]["after"]
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
        RUNTIME.secrets.pop("OPINION_KEY", None)
