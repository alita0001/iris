"""Mock end-to-end: collect -> key states -> replay -> reach -> assemble."""
import json

import pytest

from revact.data.assemble import assemble
from revact import config
from revact.data.collect import _site_for_url, collect_trajectory, run_collection
from revact.data.governance import audit_collection_lineage
from revact.data.reach import reach_and_record
from revact.envs.fingerprint import StateFingerprint, is_restored
from revact.envs.harness import RevActEnv, replay_to_state
from revact.envs.mock_env import MockRedditEnv, MockShoppingEnv
from revact.policies import ScriptedShoppingPolicy

BASE = "http://mock-shopping"
PRODUCT = "http://mock-shopping/product/20"
FLOWS = ["add_to_cart", "place_order", "delete_address"]


def test_collection_site_identity_comes_from_observed_origin(monkeypatch):
    monkeypatch.setenv("WA_SHOPPING", "http://shop.example:7770")
    monkeypatch.setenv("WA_SHOPPING_ADMIN", "http://shop.example:7780/admin")
    monkeypatch.setenv("WA_REDDIT", "http://reddit.example:9999")
    assert _site_for_url(
        "http://reddit.example:9999/f/books", "shopping") == "reddit"
    assert _site_for_url(
        "http://shop.example:7780/admin/dashboard", "shopping") == \
        "shopping_admin"
    assert _site_for_url("http://unknown.example/", "shopping") == "shopping"
    assert config.site_environment_family("shopping") == "magento"
    assert config.site_environment_family("shopping_admin") == "magento"
    assert config.site_environment_family("reddit") == "postmill"


def test_flows_reach_key_states_and_replay():
    for flow in FLOWS:
        renv = RevActEnv(MockShoppingEnv(goal=flow), task_id=f"mock_{flow}")
        key_states, _ = collect_trajectory(renv, ScriptedShoppingPolicy(flow),
                                           seed=0, trajectory_id=f"{flow}_s0",
                                           max_steps=20)
        types = {t for k in key_states for t in k.afforded_action_types}
        assert flow in types
        ks = key_states[-1]
        res = replay_to_state(MockShoppingEnv(goal="replay"), seed=ks.seed,
                              replay_prefix=ks.replay_prefix,
                              expected_fp=StateFingerprint.from_dict(ks.pre_fingerprint))
        assert res.ok


def test_mutation_is_not_restored():
    renv = RevActEnv(MockShoppingEnv(goal="mut"), task_id="mut")
    renv.reset(seed=0)
    for a in ["click('11')", "click('20')", "click('30')", "click('40')"]:
        renv.step(a)
    before = renv.current_fingerprint()
    renv.step("click('50')")             # place order (mock)
    after = renv.current_fingerprint()
    assert not is_restored(before, after)


def test_run_collection_success_and_immutable_attempt_lineage(tmp_path):
    class _Answer:
        last_raw_response = "added"
        last_finish_reason = "stop"

        def __init__(self):
            self._plan = ["click('11')", "click('20')", "click('30')",
                          "send_msg_to_user('done')"]
            self._i = 0

        def reset(self):
            self._i = 0

        def act(self, obs_view, goal="", history=None):
            if self._i >= len(self._plan):
                return None
            a = self._plan[self._i]
            self._i += 1
            return a

    def env_factory(tid):
        return MockShoppingEnv(goal="Add a laptop.",
                               reward_fn=lambda b, p, g: 1.0 if b["cart"] else 0.0)

    res1 = run_collection(env_factory, lambda t, s: _Answer(),
                          ["mock.ok"], [0], out_dir=tmp_path, max_steps=15)
    ks1 = list((tmp_path / "state_bank" / "shopping_key_states.jsonl").open())
    res2 = run_collection(env_factory, lambda t, s: _Answer(),
                          ["mock.ok"], [0], out_dir=tmp_path, max_steps=15)
    ks2 = list((tmp_path / "state_bank" / "shopping_key_states.jsonl").open())
    assert res1["n_success"] == 1 and res2["n_success"] == 1
    assert res1["run_id"] != res2["run_id"]
    assert len(ks2) == 2 * len(ks1)      # attempts remain independently auditable
    rows = [json.loads(ln) for ln in ks2]
    assert {r["run_id"] for r in rows} == {res1["run_id"], res2["run_id"]}
    assert len({r["trajectory_id"] for r in rows}) == 2
    metas = [json.loads(ln) for ln in
             (tmp_path / "trajectories_meta.jsonl").open()]
    assert {m["run_id"] for m in metas} == {res1["run_id"], res2["run_id"]}
    assert all((tmp_path / "trajectories" /
                f"{m['trajectory_id']}.jsonl").exists() for m in metas)
    assert all((tmp_path / m["policy_attempt_artifact"]).exists()
               for m in metas)
    attempt_rows = [
        json.loads(line)
        for meta in metas
        for line in (tmp_path / meta["policy_attempt_artifact"]).open()
        if line.strip()
    ]
    assert attempt_rows
    assert {row["execution_status"] for row in attempt_rows} <= {
        "EXECUTED", "GUARDED", "NO_ACTION"}
    assert all(m["policy_provenance"]["credential_value_stored"] is False
               for m in metas)
    assert all(m["policy_provenance"]["provider"] == "local_or_scripted"
               for m in metas)
    assert all(m["judge_provenance"]["credential_value_stored"] is False
               for m in metas)
    audit = audit_collection_lineage(tmp_path)
    assert audit["ok"] and audit["n_problems"] == 0
    assert audit["missing_policy_attempt_artifacts"] == []
    assert audit["invalid_policy_attempt_artifacts"] == []
    with pytest.raises(FileExistsError):
        run_collection(env_factory, lambda t, s: _Answer(),
                       ["mock.ok"], [0], out_dir=tmp_path, max_steps=15,
                       run_id=res1["run_id"])


def test_collection_lineage_rejects_policy_attempt_path_traversal(tmp_path):
    class _NoAction:
        last_raw_response = "malformed"
        last_finish_reason = "stop"

        def reset(self):
            pass

        def act(self, obs_view, goal="", history=None):
            return None

    result = run_collection(
        lambda _tid: MockShoppingEnv(goal="fixture"),
        lambda _tid, _seed: _NoAction(), ["mock.path"], [0],
        out_dir=tmp_path, max_steps=1)
    meta_path = tmp_path / "trajectories_meta.jsonl"
    meta = json.loads(meta_path.read_text())
    meta["policy_attempt_artifact"] = "../outside.jsonl"
    meta_path.write_text(json.dumps(meta) + "\n")

    audit = audit_collection_lineage(tmp_path)
    assert audit["ok"] is False
    assert audit["invalid_policy_attempt_artifacts"] == [
        result["summaries"][0]["trajectory_id"]]


def test_run_collection_routes_mixed_sites_to_separate_state_banks(
        tmp_path, monkeypatch):
    monkeypatch.setenv("WA_SHOPPING", "http://mock-shopping")
    monkeypatch.setenv("WA_REDDIT", "http://mock-reddit")

    class _Terminal:
        last_raw_response = "done"
        last_finish_reason = "stop"

        def reset(self):
            return None

        def act(self, *_args, **_kwargs):
            return "send_msg_to_user('done')"

    def reward(*_args):
        return 1.0

    def env_factory(task_id):
        if task_id == "mock.reddit":
            return MockRedditEnv(goal="read", reward_fn=reward)
        return MockShoppingEnv(goal="read", reward_fn=reward)

    result = run_collection(
        env_factory, lambda *_args: _Terminal(),
        ["mock.shopping", "mock.reddit"], [0], out_dir=tmp_path,
        run_id="mixed-sites", max_steps=1)
    assert set(result["state_banks"]) == {"shopping", "reddit"}
    assert result["state_bank"] == ""
    for site, family in (("shopping", "magento"),
                         ("reddit", "postmill")):
        rows = [json.loads(line) for line in (
            tmp_path / "state_bank" / f"{site}_key_states.jsonl"
        ).open() if line.strip()]
        assert rows
        assert {row["site"] for row in rows} == {site}
        assert {row["environment_family"] for row in rows} == {family}
    manifest = json.loads((
        tmp_path / "manifests" / "collection_runs" /
        "mixed-sites.json").read_text())
    assert set(manifest["state_banks"]) == {"shopping", "reddit"}
    assert manifest["state_bank"] == ""


def test_failed_collection_closes_auditable_failed_transaction(tmp_path):
    class _BrokenPolicy:
        def act(self, *_args, **_kwargs):
            raise RuntimeError("fixture collector crash")

    with pytest.raises(RuntimeError, match="collector crash"):
        run_collection(
            lambda _tid: MockShoppingEnv(goal="fail"),
            lambda _tid, _seed: _BrokenPolicy(), ["mock.fail"], [0],
            out_dir=tmp_path, run_id="failed-run", max_steps=2)
    manifest = json.loads((tmp_path / "manifests" / "collection_runs" /
                           "failed-run.json").read_text())
    assert manifest["status"] == "FAILED"
    assert manifest["completed_at"]
    assert manifest["failure"] == {
        "stage": "trajectory:mock.fail",
        "error_type": "RuntimeError",
        "message": "fixture collector crash",
    }
    audit = audit_collection_lineage(tmp_path)
    assert audit["incomplete_collection_manifests"] == ["failed-run"]
    assert not audit["ok"]


def test_reach_then_assemble(tmp_path):
    renv = RevActEnv(MockShoppingEnv(goal="reach"), task_id="reach")
    records = reach_and_record(renv, base=BASE, product_url=PRODUCT, seed=0)
    by_name = {r.name: r for r in records}
    assert by_name["add_to_cart"].reached and by_name["place_order"].reached

    reached = tmp_path / "reached.jsonl"
    from dataclasses import asdict
    reached.write_text("\n".join(json.dumps(asdict(r)) for r in records))
    rev = tmp_path / "rev.jsonl"
    rev.write_text("\n".join(json.dumps(r) for r in [
        {"action_type": "add_to_cart", "label": "REVERSIBLE"},
        {"action_type": "place_order", "label": "IRREVERSIBLE"},
    ]))
    res = assemble(reached, rev, tmp_path, formal=False)
    assert res["n_sft"] >= 4
    for s in res["samples"]:
        asst = s["messages"][2]["content"]
        for tag in ("<observation>", "<prediction>", "<rev_check>",
                    "<reversibility>", "<undo>", "<decision>", "<answer>"):
            assert tag in asst
        # inverse world model reasons before it labels (P2)
        assert asst.index("<rev_check>") < asst.index("<reversibility>")
