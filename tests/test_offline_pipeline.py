"""Mock end-to-end: collect -> key states -> replay -> reach -> assemble."""
import json

from revact.data.assemble import assemble
from revact.data.collect import collect_trajectory, run_collection
from revact.data.reach import reach_and_record
from revact.envs.fingerprint import StateFingerprint, is_restored
from revact.envs.harness import RevActEnv, replay_to_state
from revact.envs.mock_env import MockShoppingEnv
from revact.policies import ScriptedShoppingPolicy

BASE = "http://mock-shopping"
PRODUCT = "http://mock-shopping/product/20"
FLOWS = ["add_to_cart", "place_order", "delete_address"]


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


def test_run_collection_success_and_dedup(tmp_path):
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
    assert len(ks1) == len(ks2)          # cross-run de-dup


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
    res = assemble(reached, rev, tmp_path)
    assert res["n_sft"] >= 4
    for s in res["samples"]:
        asst = s["messages"][2]["content"]
        for tag in ("<observation>", "<prediction>", "<rev_check>",
                    "<reversibility>", "<undo>", "<decision>", "<answer>"):
            assert tag in asst
        # inverse world model reasons before it labels (P2)
        assert asst.index("<rev_check>") < asst.index("<reversibility>")
