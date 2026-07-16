"""Offline fixtures for the live point producer's safety-critical gates."""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json

import pytest

from revact.data.point_collect import (PointReachSpec, _reach_one,
                                       collect_point_states,
                                       version_point_reach_specs)
from revact import config
from revact.envs.harness import RevActEnv
from revact.envs.mock_env import MockRedditEnv
from revact.envs.mock_env import MockShoppingEnv
from revact.envs.obs_utils import prune_axtree_txt
from revact.grounding import signals
from revact.grounding.authoring import ProbeExecutionSpec
from revact.grounding.batch_prepare import (PointBatchPreparationError,
                                            prepare_point_probe_batch)
from revact.grounding.point_runner import (PointExecutionError,
                                            _attacker_action_error,
                                            _check_review_gates,
                                            _replay_to_state,
                                            load_key_states,
                                            run_point_spec, run_point_specs)


PRODUCT = """RootWebArea 'Product'
[42] button 'Add to Cart'
[43] link 'Help'
[44] link 'My Account'
[45] button 'Continue'
"""


def test_site_registry_uses_admin_only_session_bootstrap():
    assert config.SITES["shopping"].session_task == "webarena.21"
    assert config.SITES["shopping_admin"].session_task == "webarena.0"
    assert config.SITES["shopping_admin"].session_task != \
        config.SITES["shopping"].session_task


class _CartRenv:
    """Tiny persistent-state browser fixture; reset does not reset cart state."""

    def __init__(self, *, count: int = 0):
        self.count = count
        self.actions: list[str] = []
        self._last_obs_view = {"url": "http://shop/", "axtree_txt": "RootWebArea"}

    def _cart(self):
        tree = "RootWebArea 'Cart'\n"
        if self.count:
            tree += "[55] link 'Remove item'\n"
        self._last_obs_view = {
            "url": "http://shop/checkout/cart/", "axtree_txt": tree}

    def reset(self, **_kwargs):
        self._last_obs_view = {"url": "http://shop/", "axtree_txt": "RootWebArea"}

    def step(self, action: str):
        self.actions.append(action)
        if action.startswith("goto(") and "checkout/cart" in action:
            self._cart()
        elif action.startswith("goto("):
            self._last_obs_view = {
                "url": "http://shop/product.html", "axtree_txt": PRODUCT}
        elif action == "click('42')":
            self.count += 1
            self._last_obs_view = {
                "url": "http://shop/product.html", "axtree_txt": PRODUCT}
        elif action == "click('55')":
            self.count -= 1
            self._cart()
        else:
            raise AssertionError(f"unexpected fixture action {action}")
        return None, 0.0, False, False, {}, self._last_obs_view


def _spec(*, budget: int = 12) -> ProbeExecutionSpec:
    return ProbeExecutionSpec(
        schema_version="iris.probe_execution_spec.v1",
        authored_spec_id="draft-1", probe_name="shopping.add_to_cart.fixture",
        site="shopping", action_type="add_to_cart",
        probe_point_id="point-1", probe_run_id="probe-run-1",
        state_id="state-1", candidate_id="candidate-1",
        candidate_snapshot_hash="snapshot-hash", action_instance_id="action-1",
        raw_action="click('42')", canonical_action="click:button:add-to-cart",
        environment_family="webarena", environment_instance="http://shop",
        environment_origin="webarena", is_mock=False, task_id="webarena.1",
        trajectory_id="trajectory-1", run_id="run-1", seed=0,
        url="http://shop/product.html", account="fixture",
        privilege="customer", signal_channels=("ui_structural",),
        undo_sequences=(
            ("navigate_to_signal_page", "remove_cart_line"),
            ("navigate_to_signal_page", "remove_cart_line", "confirm_modal_ok"),
        ), solver_set=("site_specific_deterministic.live-1",), budget_k=budget,
        safety_level="self_recovering", controller_version="fixture-v1",
        code_version="deadbeef", fixture_status="PASSED",
        code_review_status="APPROVED", reviewer="test",
        review_timestamp="2026-07-13T00:00:00+00:00",
        requires_commit_flag=False, requires_destructive_env=False,
        product_url="http://shop/product.html")


def _key_state():
    return {"axtree_snapshot": PRODUCT,
            "replay_prefix": ["goto('http://shop/product.html')"]}


REDDIT_BASE = "http://mock-reddit"


def _reddit_spec(point_id: str = "reddit-point-1",
                 state_id: str = "reddit-state-1") -> ProbeExecutionSpec:
    return ProbeExecutionSpec(
        schema_version="iris.probe_execution_spec.v1",
        authored_spec_id=f"draft-{point_id}",
        probe_name="reddit.vote.fixture", site="reddit",
        action_type="reddit_vote", probe_point_id=point_id,
        probe_run_id="reddit-probe-run", state_id=state_id,
        candidate_id=f"candidate-{point_id}",
        candidate_snapshot_hash="snapshot-hash",
        action_instance_id=f"action-{point_id}", raw_action="click('220')",
        canonical_action="click:button:upvote", environment_family="webarena",
        environment_instance=REDDIT_BASE, environment_origin="webarena",
        is_mock=False, task_id="webarena.27",
        trajectory_id=f"trajectory-{state_id}", run_id="run-reddit", seed=0,
        url=f"{REDDIT_BASE}/f/AskReddit/1", account="reddit-fixture",
        privilege="user", signal_channels=("ui_structural",),
        undo_sequences=(("navigate_to_signal_page", "toggle_reddit_upvote"),),
        solver_set=("site_specific_deterministic.live-1",), budget_k=4,
        safety_level="self_recovering", controller_version="fixture-v1",
        code_version="deadbeef", fixture_status="PASSED",
        code_review_status="APPROVED", reviewer="test",
        review_timestamp="2026-07-13T00:00:00+00:00",
        requires_commit_flag=False, requires_destructive_env=False,
        submission_url=f"{REDDIT_BASE}/f/AskReddit/1")


def _reddit_tree() -> str:
    env = MockRedditEnv()
    env.reset(seed=0)
    obs, *_ = env.step(f"goto('{REDDIT_BASE}/f/AskReddit/1')")
    return prune_axtree_txt(obs["axtree_txt"])


class _PersistentRedditEnv(MockRedditEnv):
    """Live-like fixture: browser reset preserves Postmill account state."""

    def __init__(self, *, fail_post_read: bool = False,
                 fail_cleanup: bool = False):
        super().__init__()
        self.fail_post_read = fail_post_read
        self.fail_cleanup = fail_cleanup
        self.raise_next_navigation = False
        self.vote_clicks = 0

    def reset(self, seed: int = 0):
        if self._backend is None:
            return super().reset(seed=seed)
        backend = copy.deepcopy(self._backend)
        obs, info = super().reset(seed=seed)
        self._backend = backend
        return self._obs(), info

    def step(self, action: str):
        if self.raise_next_navigation and action.startswith("goto("):
            self.raise_next_navigation = False
            raise RuntimeError("injected post-action signal read failure")
        if action == "click('220')":
            self.vote_clicks += 1
            if self.fail_cleanup and self.vote_clicks >= 2:
                raise RuntimeError("injected recovery control failure")
        result = super().step(action)
        if action == "click('220')" and self.fail_post_read:
            self.raise_next_navigation = True
        return result


def _write_reddit_states(root, state_ids: list[str]) -> None:
    bank = root / "raw" / "state_bank"
    bank.mkdir(parents=True)
    tree = _reddit_tree()
    rows = []
    for state_id in state_ids:
        rows.append({
            "state_id": state_id,
            "axtree_snapshot": tree,
            "replay_prefix": [f"goto('{REDDIT_BASE}/f/AskReddit/1')"],
        })
    (bank / "reddit_key_states.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_clean_point_records_navigation_and_exact_undo_cost():
    renv = _CartRenv(count=0)
    result = run_point_spec(renv, _spec(), _key_state(), "http://shop")
    _check_review_gates(_spec(), result)
    assert result.effect_status == "CHANGED"
    assert result.recovery_status == "RECOVERED"
    assert result.undo_cost_steps == 2
    assert result.solver_set == ["site_specific_deterministic.live-1"]
    assert result.evidence["undo_actions"] == [
        "goto('http://shop/checkout/cart/')", "click('55')"]
    transition = result._probe_transition
    assert transition.pre_observation.url == "http://shop/product.html"
    assert transition.post_observation.url == "http://shop/product.html"
    assert transition.post_signal_observation.url == \
        "http://shop/checkout/cart/"
    assert [step.raw_action for step in transition.recovery_steps] == \
        result.evidence["undo_actions"]
    assert transition.post_observation.policy_axtree_sha256 == \
        result.evidence["post_observation_hash"]
    assert result.evidence["transition_ref"]["record_sha256"]
    assert renv.count == 0


def test_nonempty_shared_account_fails_before_candidate_or_cleanup_click():
    renv = _CartRenv(count=1)
    with pytest.raises(PointExecutionError, match="isolated clean baseline"):
        run_point_spec(renv, _spec(), _key_state(), "http://shop")
    assert renv.count == 1
    assert not any(action.startswith("click") for action in renv.actions)


def test_budget_includes_signal_page_navigation_and_cannot_overrun():
    renv = _CartRenv(count=0)
    result = run_point_spec(renv, _spec(budget=1), _key_state(), "http://shop")
    assert result.recovery_status == "PARTIALLY_RECOVERED"
    assert result.undo_cost_steps == 1
    assert result.budget_exhausted is True
    assert result.evidence["undo_actions"] == [
        "goto('http://shop/checkout/cart/')"]


def test_llm_attacker_has_hard_primitive_legality_and_origin_gate():
    view = {"axtree_txt": (
        "[1] button 'Cancel order'\n"
        "[2] button 'Place Order'\n")}
    assert _attacker_action_error("click('1')", view, "http://shop/orders") is None
    assert _attacker_action_error(
        "click('2')", view, "http://shop/orders") == \
        "commit_like_control_forbidden"
    assert _attacker_action_error(
        "goto('http://shop/account')", view, "http://shop/orders") is None
    assert _attacker_action_error(
        "goto('https://evil.invalid/')", view, "http://shop/orders") == \
        "cross_origin_goto_forbidden"
    assert _attacker_action_error(
        "fill('1', 'x')", view, "http://shop/orders").startswith(
            "primitive_forbidden")


def test_point_collection_rejects_existing_logical_state_before_browser(tmp_path):
    bank = tmp_path / "raw" / "state_bank"
    bank.mkdir(parents=True)
    (bank / "shopping_key_states.jsonl").write_text(
        json.dumps({"state_id": "state-1"}) + "\n")
    called = False

    def factory(_task_id):
        nonlocal called
        called = True
        raise AssertionError("browser must not be opened")

    with pytest.raises(ValueError, match="already_exists"):
        collect_point_states([
            PointReachSpec("state-1", "add_to_cart", [], ["add to cart"])
        ], out_dir=tmp_path, env_factory=factory, code_version="deadbeef")
    assert called is False
    assert not (tmp_path / "manifests" / "collection_runs").exists()


def test_point_collection_closes_manifest_when_live_reset_fails(tmp_path):
    class _ResetFailure:
        def reset(self, **_kwargs):
            raise RuntimeError("missing WA_SHOPPING sk-secretmustberemoved")

        def close(self):
            pass

    with pytest.raises(RuntimeError, match="missing WA_SHOPPING"):
        collect_point_states([
            PointReachSpec(
                "state-reset-failure", "add_to_cart", [], ["add to cart"])
        ], out_dir=tmp_path, env_factory=lambda _task: _ResetFailure(),
            run_id="failed-point-run", code_version="worktree:test")

    manifest = json.loads((
        tmp_path / "manifests" / "collection_runs" /
        "failed-point-run.json").read_text())
    assert manifest["status"] == "FAILED"
    assert manifest["completed_at"]
    assert manifest["failure"]["stage"] == "point_reach"
    assert manifest["failure"]["error_type"] == "RuntimeError"
    assert "sk-secretmustberemoved" not in manifest["failure"]["message"]
    assert "<redacted>" in manifest["failure"]["message"]


def test_point_collection_pins_code_version_across_raw_state_meta_manifest(
        tmp_path):
    code_version = "code-snapshot:fixture-123"
    result = collect_point_states([
        PointReachSpec(
            "state-code-version", "add_to_cart",
            ["goto('http://mock-shopping/product/item.html')"],
            ["add to cart"])
    ], out_dir=tmp_path,
        env_factory=lambda _task: MockShoppingEnv(),
        run_id="code-version-run", code_version=code_version)
    assert result["n_reached"] == 1

    trajectory_id = result["summaries"][0]["trajectory_id"]
    raw = [json.loads(line) for line in (
        tmp_path / "raw" / "trajectories" /
        f"{trajectory_id}.jsonl").read_text().splitlines() if line]
    state = json.loads((
        tmp_path / "raw" / "state_bank" /
        "shopping_key_states.jsonl").read_text())
    meta = json.loads((
        tmp_path / "raw" / "trajectories_meta.jsonl").read_text())
    manifest = json.loads((
        tmp_path / "manifests" / "collection_runs" /
        "code-version-run.json").read_text())
    assert {row["code_version"] for row in raw} == {code_version}
    assert state["code_version"] == code_version
    assert meta["code_version"] == code_version
    assert manifest["code_version"] == code_version


def test_point_collection_closes_manifest_when_env_factory_fails(tmp_path):
    def factory(_task_id):
        raise RuntimeError("browser bootstrap unavailable")

    with pytest.raises(RuntimeError, match="bootstrap unavailable"):
        collect_point_states([
            PointReachSpec(
                "state-factory-failure", "add_to_cart", [], ["add to cart"])
        ], out_dir=tmp_path, env_factory=factory,
           run_id="failed-point-factory", code_version="worktree:test")

    manifest = json.loads((
        tmp_path / "manifests" / "collection_runs" /
        "failed-point-factory.json").read_text())
    assert manifest["status"] == "FAILED"
    assert manifest["completed_at"]
    assert manifest["trajectories"] == []
    assert manifest["failure"]["stage"] == "point_reach"
    assert manifest["failure"]["error_type"] == "RuntimeError"


def test_point_recollection_requires_explicit_safe_version_suffix():
    original = [PointReachSpec(
        "state-1", "add_to_cart", ["goto('http://shop/item')"],
        ["add to cart"])]
    revised = version_point_reach_specs(original, "recapture-20260716")
    assert revised[0].state_id == "state-1__recapture-20260716"
    assert original[0].state_id == "state-1"
    with pytest.raises(ValueError, match="state_id_suffix"):
        version_point_reach_specs(original, "bad suffix")


def test_point_batch_state_join_ignores_unrelated_legacy_ambiguity(tmp_path):
    bank = tmp_path / "state_bank"
    bank.mkdir()
    rows = [
        {"state_id": "wanted", "replay_prefix": [], "value": 1},
        {"state_id": "legacy-conflict", "replay_prefix": [], "value": 1},
        {"state_id": "legacy-conflict", "replay_prefix": [], "value": 2},
    ]
    (bank / "states.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    assert load_key_states(bank, wanted={"wanted"}) == {"wanted": rows[0]}
    with pytest.raises(PointExecutionError, match="conflicting key-state"):
        load_key_states(bank, wanted={"legacy-conflict"})


def test_point_batch_state_join_ignores_derived_formal_view(tmp_path):
    bank = tmp_path / "state_bank"
    bank.mkdir()
    raw = {"state_id": "wanted", "replay_prefix": [], "value": "raw"}
    derived = {
        "state_id": "wanted", "replay_prefix": [], "value": "derived",
    }
    (bank / "states.jsonl").write_text(json.dumps(raw) + "\n")
    (bank / "formal_point_reached_states.jsonl").write_text(
        json.dumps(derived) + "\n")

    assert load_key_states(bank, wanted={"wanted"}) == {"wanted": raw}


def test_batch_preparation_closes_state_candidate_spec_without_labels(tmp_path):
    bank = tmp_path / "raw" / "state_bank"
    bank.mkdir(parents=True)
    state = {
        "state_id": "state-1", "afforded_action_types": ["add_to_cart"],
        "axtree_snapshot": PRODUCT, "affordance_bid": "42",
        "site": "shopping", "environment_instance": "http://shop",
        "environment_origin": "webarena", "task_id": "webarena.1",
        "trajectory_id": "trajectory-1", "run_id": "run-1", "seed": 0,
        "url": "http://shop/product.html", "account": "fixture",
        "privilege": "customer",
    }
    (bank / "shopping_key_states.jsonl").write_text(json.dumps(state) + "\n")
    output = tmp_path / "grounded" / "specs" / "batch-1.jsonl"
    report = prepare_point_probe_batch(
        data_root=tmp_path, state_ids=["state-1"], probe_run_id="probe-run-1",
        reviewer="test-reviewer", timestamp="2026-07-13T00:00:00+00:00",
        code_version="deadbeef", execution_path=output,
        environment_instance="http://shop")
    row = json.loads(output.read_text())
    assert report["labels_created"] == 0
    assert row["fixture_status"] == "PASSED"
    assert row["code_review_status"] == "APPROVED"
    assert row["candidate_snapshot_hash"]
    assert row["solver_set"] == ["site_specific_deterministic.live-1"]
    assert not ({"label", "effect_status", "recovery_status"} & set(row))
    assert (tmp_path / "raw" / "candidates" /
            "CANDIDATE_MANIFEST.jsonl").read_text().count("\n") >= 4

    # Identical replay is idempotent; a different immutable batch is rejected.
    prepare_point_probe_batch(
        data_root=tmp_path, state_ids=["state-1"], probe_run_id="probe-run-1",
        reviewer="test-reviewer", timestamp="2026-07-13T00:00:00+00:00",
        code_version="deadbeef", execution_path=output,
        environment_instance="http://shop")
    with pytest.raises(PointBatchPreparationError, match="overwrite immutable"):
        prepare_point_probe_batch(
            data_root=tmp_path, state_ids=["state-1"],
            probe_run_id="different-run", reviewer="test-reviewer",
            timestamp="2026-07-13T00:00:00+00:00",
            code_version="deadbeef", execution_path=output,
            environment_instance="http://shop")


def test_reddit_signal_binds_submission_and_ignores_public_score():
    class StaticRenv:
        _last_obs_view = {"axtree_txt": (
            "RootWebArea 'Submission'\n"
            "[10] button 'Upvote'\n"
            "StaticText 'Score: 101'\n"
            "[11] button 'Retract downvote'\n"
            "StaticText 'comment'\n"
            "[20] button 'Retract upvote'\n"
            "StaticText 'Score: 999'\n"
            "[21] button 'Downvote'\n")}

    renv = StaticRenv()
    raw = signals.vote_score(
        renv, f"{REDDIT_BASE}/f/AskReddit/1", navigate=False)
    assert raw["submission_id"] == "1"
    assert raw["score"] == 101
    assert raw["vote_direction"] == -1
    assert raw["up_bid"] == "10"
    assert raw["down_bid"] == "11"
    canonical = signals.reddit_vote_canonical(
        renv, f"{REDDIT_BASE}/f/AskReddit/1", navigate=False)
    assert canonical == {
        "signal": "reddit_own_vote_state",
        "submission_id": "1",
        "vote_direction": -1,
    }
    assert "score" not in canonical


def test_reddit_subscribe_requires_one_exact_control_and_excludes_rss():
    class StaticRenv:
        _last_obs_view = {"axtree_txt": (
            "RootWebArea 'Forum'\n"
            "[1] button 'Subscribe via RSS'\n"
            "[2] button 'Subscribe No subscribers'\n")}

    signal = signals.subscribe_signal(
        StaticRenv(), f"{REDDIT_BASE}/f/books", navigate=False)
    assert signal["found"] is True
    assert signal["bid"] == "2"
    assert signal["matching_control_count"] == 1


def test_action_site_and_same_origin_gates_fail_before_mutation():
    renv = _CartRenv()
    with pytest.raises(PointExecutionError, match="action/site adapter mismatch"):
        run_point_spec(
            renv, replace(_spec(), site="reddit"), _key_state(), "http://shop")
    assert renv.actions == []

    reddit = _reddit_spec()
    with pytest.raises(PointExecutionError, match="outside reviewed site origin"):
        run_point_spec(
            renv, replace(reddit, submission_url="https://evil.invalid/f/x/1"),
            _key_state(), REDDIT_BASE)
    assert renv.actions == []


def test_point_batch_wal_recovers_after_exception_and_excludes_result(
        tmp_path, monkeypatch):
    monkeypatch.setenv("WA_REDDIT", REDDIT_BASE)
    _write_reddit_states(tmp_path, ["reddit-state-1"])
    created = []

    def factory(_task_id, _headless):
        env = _PersistentRedditEnv(fail_post_read=True)
        created.append(env)
        return env

    outcome = run_point_specs(
        [_reddit_spec()], tmp_path, env_factory=factory)
    assert outcome["results"] == []
    assert outcome["transitions"] == []
    assert len(outcome["failures"]) == 1
    failure = outcome["failures"][0]
    assert failure["cleanup_status"] == "RECOVERED_AFTER_EXCEPTION"
    assert created[0]._backend["submissions"]["1"]["my_vote"] == 0
    wal = json.loads((tmp_path / "grounded" / "recovery_journal" /
                      "reddit-point-1.json").read_text(encoding="utf-8"))
    assert wal["status"] == "RECOVERED_AFTER_EXCEPTION"
    assert wal["cleanup_verified"] is True
    assert wal["mutation_attempted"] is True


def test_point_batch_success_is_persistent_reset_safe_and_terminally_journaled(
        tmp_path, monkeypatch):
    monkeypatch.setenv("WA_REDDIT", REDDIT_BASE)
    _write_reddit_states(tmp_path, ["reddit-state-1"])
    created = []

    def factory(_task_id, _headless):
        env = _PersistentRedditEnv()
        created.append(env)
        return env

    outcome = run_point_specs(
        [_reddit_spec()], tmp_path, env_factory=factory)
    assert outcome["failures"] == []
    assert len(outcome["results"]) == 1
    assert len(outcome["transitions"]) == 1
    result, _ctx = outcome["results"][0]
    assert (result.effect_status, result.recovery_status) == (
        "CHANGED", "RECOVERED")
    assert created[0]._backend["submissions"]["1"]["my_vote"] == 0
    wal = json.loads((tmp_path / "grounded" / "recovery_journal" /
                      "reddit-point-1.json").read_text(encoding="utf-8"))
    assert wal["status"] == "RECOVERED"
    assert wal["cleanup_verified"] is True


def test_cleanup_failure_leaves_wal_and_aborts_later_points(
        tmp_path, monkeypatch):
    monkeypatch.setenv("WA_REDDIT", REDDIT_BASE)
    _write_reddit_states(tmp_path, ["reddit-state-1", "reddit-state-2"])
    created = []

    def factory(_task_id, _headless):
        env = _PersistentRedditEnv(fail_cleanup=True)
        created.append(env)
        return env

    specs = [
        _reddit_spec("reddit-point-1", "reddit-state-1"),
        _reddit_spec("reddit-point-2", "reddit-state-2"),
    ]
    outcome = run_point_specs(specs, tmp_path, env_factory=factory)
    assert outcome["results"] == []
    assert outcome["transitions"] == []
    assert [row["cleanup_status"] for row in outcome["failures"]] == [
        "RECOVERY_FAILED", "NOT_ATTEMPTED_BATCH_ABORTED"]
    assert created[0]._backend["submissions"]["1"]["my_vote"] == 1
    wal = json.loads((tmp_path / "grounded" / "recovery_journal" /
                      "reddit-point-1.json").read_text(encoding="utf-8"))
    assert wal["status"] == "RECOVERY_FAILED"
    assert wal["cleanup_verified"] is False
    assert not (tmp_path / "grounded" / "recovery_journal" /
                "reddit-point-2.json").exists()


def test_target_anchored_collection_retains_tail_bid():
    filler = "\n".join(
        f"StaticText 'filler-{i}-{'x' * 80}'" for i in range(200))
    long_tree = f"RootWebArea 'Long'\n{filler}\n[999] button 'Subscribe'"

    class LongEnv:
        def reset(self, seed=0):
            return ({"goal": "", "url": REDDIT_BASE,
                     "axtree_txt": "RootWebArea 'Home'"}, {})

        def step(self, _action):
            obs = {"goal": "", "url": f"{REDDIT_BASE}/f/books",
                   "axtree_txt": long_tree}
            return obs, 0.0, False, False, {}

        def close(self):
            return None

    renv = RevActEnv(LongEnv(), task_id="webarena.27", site="reddit")
    row, summary = _reach_one(
        renv,
        PointReachSpec(
            "tail-state", "reddit_subscribe",
            [f"goto('{REDDIT_BASE}/f/books')"], ["subscribe"], site="reddit"),
        seed=0, run_id="run-tail", trajectory_id="trajectory-tail",
        code_version=hashlib.sha256(b"fixture").hexdigest()[:8])
    assert summary["success"] is True
    assert row is not None
    assert row["affordance_bid"] == "999"
    assert "[999] button 'Subscribe'" in row["axtree_snapshot"]
    replay_hash, replay_evidence = _replay_to_state(
        renv,
        replace(
            _reddit_spec(), raw_action="click('999')",
            action_type="reddit_subscribe",
            forum_url=f"{REDDIT_BASE}/f/books",
            submission_url=""),
        row)
    assert replay_hash == hashlib.sha256(
        row["axtree_snapshot"].encode("utf-8")).hexdigest()
    assert replay_evidence["replay_snapshot_exact"] is True


def test_dynamic_reddit_replay_accepts_only_identical_target_contract():
    recorded = (
        "RootWebArea 'books'\n"
        "[1] heading 'First post'\n"
        "[2] heading 'Second post'\n"
        "[999] button 'Subscribe'")
    reordered = (
        "RootWebArea 'books'\n"
        "[2] heading 'Second post'\n"
        "[1] heading 'First post'\n"
        "[999] button 'Subscribe'")

    class DynamicEnv:
        def __init__(self, tree):
            self.tree = tree

        def reset(self, seed=0):
            return ({"goal": "", "url": REDDIT_BASE,
                     "axtree_txt": "RootWebArea 'Home'"}, {})

        def step(self, _action):
            obs = {"goal": "", "url": f"{REDDIT_BASE}/f/books",
                   "axtree_txt": self.tree}
            return obs, 0.0, False, False, {}

        def close(self):
            return None

    ks = {
        "axtree_snapshot": prune_axtree_txt(recorded, anchor_bids=["999"]),
        "replay_prefix": [f"goto('{REDDIT_BASE}/f/books')"],
        "url": f"{REDDIT_BASE}/f/books",
    }
    spec = replace(
        _reddit_spec(), action_type="reddit_subscribe",
        raw_action="click('999')", canonical_action="click:button:subscribe",
        url=f"{REDDIT_BASE}/f/books", forum_url=f"{REDDIT_BASE}/f/books",
        submission_url="")
    renv = RevActEnv(DynamicEnv(reordered), task_id="webarena.27", site="reddit")
    _hash, evidence = _replay_to_state(renv, spec, ks)
    assert evidence["replay_snapshot_exact"] is False
    assert evidence["replay_verification"] == "dynamic_page_target_contract"

    renv = RevActEnv(
        DynamicEnv(reordered.replace("'Subscribe'", "'Subscribe via RSS'")),
        task_id="webarena.27", site="reddit")
    with pytest.raises(PointExecutionError, match="target contract"):
        _replay_to_state(renv, spec, ks)
