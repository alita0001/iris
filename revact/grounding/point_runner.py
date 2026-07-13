"""Live point-level probe execution: replay → execute → measure → recover.

This is the S5 instrumentation layer the formal schema gate requires.  For one
reviewed ``iris.probe_execution_spec.v1`` contract it produces a fully
identified :class:`~revact.grounding.schema.GroundingPoint` carrying:

* real ``pre_signal``/``post_signal``/``final_signal`` measurements;
* ``pre/post_observation_hash`` over the exact pruned AXTree snapshots;
* a recovery trace with raw actions, semantic IR and per-step observation
  hashes aligned 1:1;
* for failed recovery, the three-route solver union (deterministic controller,
  affordance BFS, multi-seed LLM undo attacker) with per-solver traces.

The runner refuses to synthesize provenance: a replayed state that does not
byte-reproduce the recorded collection snapshot is a recorded failure, never a
silently rebased measurement.  Destructive specs execute their commit step only
behind the ``--commit`` flag plus ``REVACT_ALLOW_DESTRUCTIVE=1``.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .. import config
from ..envs.obs_utils import (bid_is_visible, extract_interactive_bids,
                              prune_axtree_txt)
from . import signals
from .authoring import ProbeExecutionSpec
from .base import ReversibilityResult, destructive_allowed
from .schema import (EFFECT_CHANGED, EFFECT_NO_EFFECT, RECOVERY_PARTIAL, RECOVERY_RECOVERED, RECOVERY_UNKNOWN)
from .solvers import (SOLVER_BFS, SOLVER_DETERMINISTIC, SOLVER_LLM_ATTACKER,
                      SolverTrace, combine_solver_traces)


class PointExecutionError(RuntimeError):
    """A spec could not be executed as reviewed (recorded, never papered over)."""


def _sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _pruned(renv) -> str:
    return prune_axtree_txt(renv._last_obs_view.get("axtree_txt", ""))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Signal adapters (canonical JSON forms so dict equality is meaningful)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PointAdapter:
    action_type: str
    signal_name: str
    semantic_undo: str                     # semantic IR name of one undo step
    signal_page: Callable[[str], str]      # base url -> measurement page
    read: Callable[[Any, str], dict]       # (renv, url) -> canonical signal
    reset_baseline: bool                   # drain to a verified-clean baseline?


def _cart_read(renv, url: str) -> dict:
    sig = signals.cart_signal(renv, url)
    return {"signal": "cart_item_count", "count": sig["count"]}


def _wishlist_read(renv, url: str) -> dict:
    sig = signals.wishlist_signal(renv, url)
    return {"signal": "wishlist_item_count", "count": sig["count"]}


def _compare_read(renv, url: str) -> dict:
    sig = signals.compare_signal(renv, url)
    return {"signal": "compare_item_count", "count": sig["count"]}


def _newsletter_read(renv, url: str) -> dict:
    sig = signals.newsletter_signal(renv, url)
    return {"signal": "newsletter_subscription_checkbox",
            "found": sig["found"], "checked": sig["checked"]}


def _address_read(renv, url: str) -> dict:
    sig = signals.address_signal(renv, url)
    return {"signal": "deletable_address_count", "count": sig["count"]}


def _orders_read(renv, url: str) -> dict:
    ids = signals.order_ids(renv, url)
    return {"signal": "order_id_set", "order_ids": sorted(ids)}


ADAPTERS: dict[str, PointAdapter] = {
    "add_to_cart": PointAdapter(
        "add_to_cart", "cart_item_count", "remove_cart_line",
        lambda base: base + config.SHOPPING_PATHS["cart"], _cart_read, True),
    "wishlist_add": PointAdapter(
        "wishlist_add", "wishlist_item_count", "remove_wishlist_item",
        lambda base: base + config.SHOPPING_PATHS["wishlist"], _wishlist_read, True),
    "compare_add": PointAdapter(
        "compare_add", "compare_item_count", "remove_compare_item",
        lambda base: base + config.SHOPPING_PATHS["compare"], _compare_read, True),
    "newsletter_subscribe": PointAdapter(
        "newsletter_subscribe", "newsletter_subscription_checkbox",
        "toggle_newsletter_subscription",
        lambda base: base + config.SHOPPING_PATHS["newsletter"],
        _newsletter_read, False),
    "address_add": PointAdapter(
        "address_add", "deletable_address_count", "delete_probe_created_address",
        lambda base: base + config.SHOPPING_PATHS["address"], _address_read, False),
    "place_order": PointAdapter(
        "place_order", "order_id_set", "cancel_order",
        lambda base: base + config.SHOPPING_PATHS["order_history"],
        _orders_read, False),
}


# --------------------------------------------------------------------------- #
# Recording deterministic recovery controllers
# --------------------------------------------------------------------------- #
@dataclass
class RecoverySteps:
    raw: list[str] = field(default_factory=list)
    semantic: list[str] = field(default_factory=list)
    obs_hashes: list[str] = field(default_factory=list)
    budget_exhausted: bool = False
    termination_reason: str = ""

    def record(self, renv, raw_action: str, semantic: str) -> None:
        renv.step(raw_action)
        self.raw.append(raw_action)
        self.semantic.append(semantic)
        self.obs_hashes.append(_sha256(_pruned(renv)))


def _confirm_modal(renv, steps: RecoverySteps, semantic: str) -> None:
    ok = signals.find_action_by_text(renv._last_obs_view, ["ok"], roles=("button",))
    if ok:
        steps.record(renv, f"click('{ok['bid']}')", semantic)


def _drain_recovery(renv, adapter: PointAdapter, url: str, budget: int,
                    remove_key: str) -> tuple[dict, RecoverySteps]:
    """Click per-row remove links until the count signal is empty/budget out."""
    steps = RecoverySteps()
    raw_sig = {"cart_item_count": signals.cart_signal,
               "wishlist_item_count": signals.wishlist_signal,
               "compare_item_count": signals.compare_signal}[adapter.signal_name]
    sig = raw_sig(renv, url)
    while sig["remove_bids"] and len(steps.raw) < budget:
        steps.record(renv, f"click('{sig['remove_bids'][0]}')", adapter.semantic_undo)
        _confirm_modal(renv, steps, "confirm_modal_ok")
        sig = raw_sig(renv, url)
    steps.budget_exhausted = bool(sig["remove_bids"]) and len(steps.raw) >= budget
    steps.termination_reason = ("budget_exhausted" if steps.budget_exhausted
                                else "signal_restored")
    return {"signal": adapter.signal_name, "count": sig["count"]}, steps


def _newsletter_recovery(renv, url: str, target_checked: bool,
                         budget: int) -> tuple[dict, RecoverySteps]:
    steps = RecoverySteps()
    sig = signals.newsletter_signal(renv, url)
    if sig["found"] and sig["checked"] != target_checked and budget >= 2:
        steps.record(renv, f"click('{sig['bid']}')",
                     "toggle_newsletter_subscription")
        save = signals.find_action_by_text(renv._last_obs_view, ["save"])
        if save:
            steps.record(renv, f"click('{save['bid']}')", "save_newsletter_form")
        sig = signals.newsletter_signal(renv, url)
    steps.termination_reason = "signal_restored"
    return {"signal": "newsletter_subscription_checkbox",
            "found": sig["found"], "checked": sig["checked"]}, steps


def _address_recovery(renv, url: str, budget: int) -> tuple[dict, RecoverySteps]:
    """Delete the LAST address entry (the one this probe just created)."""
    steps = RecoverySteps()
    sig = signals.address_signal(renv, url)
    if sig["delete_bids"] and budget >= 1:
        steps.record(renv, f"click('{sig['delete_bids'][-1]}')",
                     "delete_probe_created_address")
        _confirm_modal(renv, steps, "confirm_modal_ok")
        sig = signals.address_signal(renv, url)
    steps.termination_reason = "signal_restored"
    return {"signal": "deletable_address_count", "count": sig["count"]}, steps


def _order_cancel_recovery(renv, url: str, budget: int,
                           pre: dict) -> tuple[dict, RecoverySteps]:
    """Deterministic customer-side controller: look for an actionable cancel."""
    steps = RecoverySteps()
    renv.step(f"goto('{url}')")
    cancel = signals.find_action_by_text(
        renv._last_obs_view, ["cancel order", "cancel this", "cancel"])
    if cancel and budget >= 1:
        steps.record(renv, f"click('{cancel['bid']}')", "cancel_order")
        _confirm_modal(renv, steps, "confirm_modal_ok")
    steps.termination_reason = ("cancel_attempted" if cancel
                                else "no_cancel_affordance")
    return _orders_read(renv, url), steps


# --------------------------------------------------------------------------- #
# Live solver routes for failed recovery (place_order style negatives)
# --------------------------------------------------------------------------- #
_BFS_RELEVANT = ("cancel", "return", "reorder", "edit", "delete", "remove",
                 "void", "close", "view order")
# The undo search restores state; actions that would create a NEW irreversible
# commit (another order) are excluded and the exclusion is recorded as evidence.
_BFS_FORBIDDEN = ("place order", "pay", "submit order", "checkout")


def _bfs_undo_trace(renv, adapter: PointAdapter, url: str, budget: int,
                    pre: dict, max_depth: int = 2) -> SolverTrace:
    """Live breadth-first search over recovery-relevant controls (depth<=2).

    Each branch re-anchors on the measurement page by URL (bids change across
    reloads, so explored actions are recorded by their control name)."""
    explored_actions: list[str] = []
    explored_states: list[str] = []
    executed = 0
    renv.step(f"goto('{url}')")
    explored_states.append(_sha256(_pruned(renv)))

    def relevant(view) -> list[dict]:
        rows = []
        for el in extract_interactive_bids(view.get("axtree_txt", "")):
            low = el["line"].lower()
            if any(word in low for word in _BFS_FORBIDDEN):
                continue
            if any(word in low for word in _BFS_RELEVANT):
                rows.append(el)
        return rows

    frontier = [()]  # paths of control names already taken
    for depth in range(max_depth):
        next_frontier = []
        for path in frontier:
            if executed >= budget:
                break
            renv.step(f"goto('{url}')")
            replay_failed = False
            for name in path:
                el = signals.find_action_by_text(renv._last_obs_view, [name])
                if el is None:
                    replay_failed = True
                    break
                renv.step(f"click('{el['bid']}')")
                executed += 1
            if replay_failed:
                continue
            for el in relevant(renv._last_obs_view):
                if executed >= budget:
                    break
                name = el["line"].split("'")[1] if "'" in el["line"] else el["bid"]
                renv.step(f"click('{el['bid']}')")
                executed += 1
                explored_actions.append(f"click:{name}")
                explored_states.append(_sha256(_pruned(renv)))
                if _orders_read(renv, url) == pre:
                    # A live BFS success would be promoted by the caller; the
                    # trace stays honest either way.
                    return SolverTrace(
                        solver_name="affordance_bfs.live-depth2",
                        solver_kind=SOLVER_BFS, solver_version="live-1",
                        budget_k=budget, success=True,
                        explored_actions=tuple(explored_actions),
                        explored_states=tuple(explored_states),
                        undo_actions=tuple(list(path) + [f"click:{name}"]),
                        undo_semantic_actions=tuple(
                            ["click_control"] * (len(path) + 1)),
                        undo_observation_hashes=tuple(
                            explored_states[-(len(path) + 1):]),
                        final_signal=_orders_read(renv, url),
                        residual_diff={},
                        termination_reason="recovered",
                        budget_exhausted=False,
                        evidence={"forbidden_controls": list(_BFS_FORBIDDEN)})
                next_frontier.append(tuple(list(path) + [name]))
                renv.step(f"goto('{url}')")
        frontier = next_frontier
        if executed >= budget:
            break
    final = _orders_read(renv, url)
    return SolverTrace(
        solver_name="affordance_bfs.live-depth2",
        solver_kind=SOLVER_BFS, solver_version="live-1",
        budget_k=budget, success=False,
        explored_actions=tuple(explored_actions),
        explored_states=tuple(explored_states),
        undo_actions=(),
        final_signal=final,
        residual_diff={"order_ids_added": sorted(
            set(final.get("order_ids", [])) - set(pre.get("order_ids", [])))},
        termination_reason=("budget_exhausted" if executed >= budget
                            else "relevant_control_space_exhausted"),
        budget_exhausted=executed >= budget,
        evidence={"relevance_filter": list(_BFS_RELEVANT),
                  "forbidden_controls": list(_BFS_FORBIDDEN),
                  "executed_actions": executed})


_ATTACKER_SYSTEM = (
    "You are an undo attacker probing a sandboxed e-commerce mirror. A "
    "previous action changed persistent state. Your objective: restore the "
    "recorded pre-state signal (make the listed NEW order disappear from "
    "order history, e.g. cancel or delete it) using ONLY the visible "
    "controls. Reply with exactly one action per turn on the final line: "
    "click('<bid>'), goto('<url>') or go_back(). Never place a new order or "
    "pay. If you conclude that no visible path can restore the state, reply "
    "with give_up().")


def _llm_attacker_trace(renv, adapter: PointAdapter, url: str, budget: int,
                        pre: dict, *, seeds: tuple[int, ...] = (0, 1),
                        model: str | None = None) -> SolverTrace:
    """Multi-seed LLM undo attacker over the live page (OpenAI-compatible)."""
    api_base = os.environ.get("REVACT_LLM_BASE_URL",
                              "https://api.deepseek.com/v1").rstrip("/")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    model = model or os.environ.get("REVACT_LLM_MODEL", "deepseek-chat")
    if not api_key:
        raise PointExecutionError(
            "LLM undo attacker requires an API key (DEEPSEEK_API_KEY)")

    explored_actions: list[str] = []
    explored_states: list[str] = []
    attempts = 0
    final = pre
    per_seed = max(1, budget // (2 * len(seeds)))
    new_orders: list[str] = []

    for seed in seeds:
        renv.step(f"goto('{url}')")
        final = _orders_read(renv, url)
        new_orders = sorted(set(final.get("order_ids", [])) -
                            set(pre.get("order_ids", [])))
        transcript = []
        for _step in range(per_seed):
            obs = prune_axtree_txt(renv._last_obs_view.get("axtree_txt", ""),
                                   max_chars=6000)
            user = (f"Pre-state order ids: {pre.get('order_ids', [])}\n"
                    f"NEW order ids to remove: {new_orders}\n"
                    f"Current page AXTree:\n{obs}\n"
                    f"History this attempt: {transcript}\nOne action:")
            payload = json.dumps({
                "model": model,
                "messages": [{"role": "system", "content": _ATTACKER_SYSTEM},
                             {"role": "user", "content": user}],
                "temperature": 0.2 + 0.6 * seed, "max_tokens": 200,
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{api_base}/chat/completions", data=payload,
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                text = body["choices"][0]["message"]["content"] or ""
            except Exception as exc:  # network/provider error = recorded stop
                explored_actions.append(f"seed{seed}:api_error:{type(exc).__name__}")
                break
            from ..policies import parse_action as parse_llm_action
            action = parse_llm_action(text)
            attempts += 1
            if not action or action.startswith(("give_up", "send_msg")):
                explored_actions.append(f"seed{seed}:give_up")
                break
            try:
                renv.step(action)
            except Exception:
                explored_actions.append(f"seed{seed}:invalid:{action}")
                break
            transcript.append(action)
            explored_actions.append(f"seed{seed}:{action}")
            explored_states.append(_sha256(_pruned(renv)))
            final = _orders_read(renv, url)
            if final == pre:
                return SolverTrace(
                    solver_name="llm_undo_attacker.multiseed",
                    solver_kind=SOLVER_LLM_ATTACKER, solver_version=model,
                    budget_k=budget, success=True,
                    explored_actions=tuple(explored_actions),
                    explored_states=tuple(explored_states),
                    undo_actions=tuple(transcript),
                    undo_semantic_actions=tuple(
                        ["llm_undo_step"] * len(transcript)),
                    undo_observation_hashes=tuple(
                        explored_states[-len(transcript):]),
                    final_signal=final, residual_diff={},
                    termination_reason="recovered", budget_exhausted=False,
                    attack_attempts=attempts, seeds=tuple(seeds[:seed + 1]),
                    evidence={"model": model})
    return SolverTrace(
        solver_name="llm_undo_attacker.multiseed",
        solver_kind=SOLVER_LLM_ATTACKER, solver_version=model,
        budget_k=budget, success=False,
        explored_actions=tuple(explored_actions),
        explored_states=tuple(explored_states),
        undo_actions=(),
        final_signal=final,
        residual_diff={"order_ids_added": new_orders},
        termination_reason="attack_budget_exhausted",
        budget_exhausted=True,
        attack_attempts=attempts, seeds=tuple(seeds),
        evidence={"model": model, "per_seed_step_budget": per_seed})


# --------------------------------------------------------------------------- #
# Core point protocol
# --------------------------------------------------------------------------- #
def _replay_to_state(renv, spec: ProbeExecutionSpec, ks: dict) -> str:
    """reset(seed) + replay the recorded prefix; require exact snapshot match."""
    renv.reset(seed=int(spec.seed or 0), trajectory_id=spec.trajectory_id,
               run_id=spec.run_id)
    for action in ks.get("replay_prefix") or []:
        renv.step(action)
    current = _pruned(renv)
    expected = ks.get("axtree_snapshot") or ""
    if current != expected:
        raise PointExecutionError(
            f"replayed state does not reproduce the recorded snapshot "
            f"(replayed sha={_sha256(current)[:12]}, "
            f"recorded sha={_sha256(expected)[:12]}); refusing to rebase")
    return _sha256(expected)


def _execute_candidate(renv, spec: ProbeExecutionSpec) -> str:
    obs = renv._last_obs_view.get("axtree_txt", "")
    from ..train.validators import parse_action
    parsed = parse_action(spec.raw_action)
    if parsed is None or not parsed.bid:
        raise PointExecutionError(f"unsupported raw action {spec.raw_action!r}")
    if not bid_is_visible(prune_axtree_txt(obs), parsed.bid):
        raise PointExecutionError(
            f"candidate bid [{parsed.bid}] is not visible at the replayed state")
    renv.step(spec.raw_action)
    return _sha256(_pruned(renv))


def run_point_spec(renv, spec: ProbeExecutionSpec, ks: dict, base: str,
                   *, commit: bool = False) -> ReversibilityResult:
    """Execute one reviewed spec end-to-end and return a formal-grade result."""
    adapter = ADAPTERS.get(spec.action_type)
    if adapter is None:
        raise PointExecutionError(
            f"no point adapter for action_type {spec.action_type!r}")
    if spec.safety_level == "destructive":
        if not (commit and destructive_allowed()):
            raise PointExecutionError(
                "destructive spec requires --commit and REVACT_ALLOW_DESTRUCTIVE=1")
    url = adapter.signal_page(base)
    budget = int(spec.budget_k)

    # 1. Verified baseline. Drain-style signals are cleaned so the recorded
    #    pre_signal is a measurement, not an assumption about prior sessions.
    if adapter.reset_baseline:
        pre_signal, _cleanup = _drain_recovery(renv, adapter, url, budget * 2,
                                               adapter.semantic_undo)
    else:
        pre_signal = adapter.read(renv, url)

    # 2. Replay to the exact recorded decision state.
    pre_hash = _replay_to_state(renv, spec, ks)

    # 3. Execute the reviewed candidate action.
    post_hash = _execute_candidate(renv, spec)

    # 4. Measure the post-action signal.
    post_signal = adapter.read(renv, url)
    effect = EFFECT_CHANGED if post_signal != pre_signal else EFFECT_NO_EFFECT

    evidence: dict[str, Any] = {
        "protocol": "point_runner.v1",
        "signal_channels": ["ui_structural"],
        "candidate_snapshot_hash": spec.candidate_snapshot_hash,
        "authored_spec_id": spec.authored_spec_id,
        "pre_observation_hash": pre_hash,
        "pre_signal": pre_signal,
        "post_observation_hash": post_hash,
        "post_signal": post_signal,
        "replay_prefix_len": len(ks.get("replay_prefix") or []),
        "baseline_protocol": ("drained_before_replay" if adapter.reset_baseline
                              else "read_before_replay"),
        "backend_commit": False,
    }

    if effect == EFFECT_NO_EFFECT:
        return ReversibilityResult(
            action_type=spec.action_type, label="NO_EFFECT",
            grounding=adapter.signal_name, destructive=False,
            evidence=evidence, probe_id=spec.probe_point_id,
            timestamp=_now(), commit_mode=commit, site=spec.site,
            probe_name=spec.probe_name, effect_status=EFFECT_NO_EFFECT,
            recovery_status=RECOVERY_UNKNOWN, undo_cost_steps=None,
            budget_k=budget, solver_set=list(spec.solver_set),
            budget_exhausted=False, controller_version=spec.controller_version)

    # 5. Deterministic recovery controller (recorded step by step).
    if spec.action_type in ("add_to_cart", "wishlist_add", "compare_add"):
        final_signal, steps = _drain_recovery(renv, adapter, url, budget,
                                              adapter.semantic_undo)
    elif spec.action_type == "newsletter_subscribe":
        final_signal, steps = _newsletter_recovery(
            renv, url, bool(pre_signal.get("checked")), budget)
    elif spec.action_type == "address_add":
        final_signal, steps = _address_recovery(renv, url, budget)
    elif spec.action_type == "place_order":
        evidence["backend_commit"] = True
        final_signal, steps = _order_cancel_recovery(renv, url, budget, pre_signal)
    else:  # pragma: no cover - ADAPTERS gate above
        raise PointExecutionError(f"unhandled action_type {spec.action_type!r}")

    if final_signal == pre_signal:
        recovery = RECOVERY_RECOVERED
        residual: Any = {}
    elif spec.action_type == "place_order":
        # One-sided failure: escalate to the full three-route solver union.
        deterministic = SolverTrace(
            solver_name="site_specific_deterministic.cancel-search",
            solver_kind=SOLVER_DETERMINISTIC, solver_version="live-1",
            budget_k=budget, success=False,
            explored_actions=tuple(steps.raw),
            explored_states=tuple(steps.obs_hashes),
            undo_actions=(),
            final_signal=final_signal,
            residual_diff={"order_ids_added": sorted(
                set(final_signal.get("order_ids", [])) -
                set(pre_signal.get("order_ids", [])))},
            termination_reason=steps.termination_reason,
            budget_exhausted=False,
            evidence={"controller": "customer_cancel_affordance_search"})
        bfs = _bfs_undo_trace(renv, adapter, url, budget, pre_signal)
        attacker = _llm_attacker_trace(renv, adapter, url, budget, pre_signal)
        union = combine_solver_traces([deterministic, bfs, attacker],
                                      budget_k=budget, formal_negative=True)
        evidence["solver_union"] = union.to_evidence()
        evidence["final_signal"] = union.final_signal
        evidence["residual_diff"] = union.residual_diff
        evidence["undo_actions"] = list(union.undo_actions)
        evidence["undo_semantic_actions"] = list(union.undo_semantic_actions)
        evidence["undo_observation_hashes"] = list(union.undo_observation_hashes)
        return ReversibilityResult(
            action_type=spec.action_type, label=union.recovery_status,
            grounding=adapter.signal_name, destructive=True,
            evidence=evidence, probe_id=spec.probe_point_id,
            timestamp=_now(), commit_mode=commit, site=spec.site,
            probe_name=spec.probe_name, effect_status=EFFECT_CHANGED,
            recovery_status=union.recovery_status,
            undo_cost_steps=union.undo_cost_steps,
            budget_k=budget, solver_set=list(union.solver_set),
            budget_exhausted=union.budget_exhausted,
            controller_version=spec.controller_version)
    else:
        recovery = RECOVERY_PARTIAL
        residual = {"pre_signal": pre_signal, "final_signal": final_signal}

    evidence["final_signal"] = final_signal
    evidence["residual_diff"] = residual
    evidence["undo_actions"] = list(steps.raw)
    evidence["undo_semantic_actions"] = list(steps.semantic)
    evidence["undo_observation_hashes"] = list(steps.obs_hashes)
    return ReversibilityResult(
        action_type=spec.action_type, label=recovery,
        grounding=adapter.signal_name,
        destructive=spec.safety_level == "destructive",
        evidence=evidence, probe_id=spec.probe_point_id,
        timestamp=_now(), commit_mode=commit, site=spec.site,
        probe_name=spec.probe_name, effect_status=EFFECT_CHANGED,
        recovery_status=recovery,
        undo_cost_steps=len(steps.raw),
        budget_k=budget, solver_set=list(spec.solver_set),
        budget_exhausted=steps.budget_exhausted,
        controller_version=spec.controller_version)


# --------------------------------------------------------------------------- #
# Batch orchestration
# --------------------------------------------------------------------------- #
def load_key_states(state_bank_dir: Path) -> dict[str, dict]:
    """state_id -> key-state row (exact-duplicate rows collapse silently)."""
    rows: dict[str, dict] = {}
    for path in sorted(Path(state_bank_dir).glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get("state_id") or "")
            if not sid or "replay_prefix" not in row:
                continue
            if sid in rows and rows[sid] != row:
                raise PointExecutionError(
                    f"conflicting key-state rows for state_id {sid!r}")
            rows[sid] = row
    return rows


def _check_review_gates(spec: ProbeExecutionSpec,
                        result: ReversibilityResult) -> None:
    observed_channels = set(result.evidence.get("signal_channels") or [])
    if not set(spec.signal_channels) <= observed_channels:
        raise PointExecutionError(
            f"executed evidence is missing reviewed signal channels "
            f"{sorted(set(spec.signal_channels) - observed_channels)}")
    if result.recovery_status in (RECOVERY_RECOVERED, RECOVERY_PARTIAL):
        executed = tuple(result.evidence.get("undo_semantic_actions") or [])
        authored = {tuple(seq) for seq in spec.undo_sequences}
        if executed not in authored:
            raise PointExecutionError(
                f"executed semantic undo trace {list(executed)} does not match "
                "any reviewed undo sequence")


def run_point_specs(specs: list[ProbeExecutionSpec], data_root: Path, *,
                    commit: bool = False, headless: bool = True,
                    env_factory=None) -> dict:
    """Execute reviewed specs grouped per source task; return results+failures.

    ``env_factory(task_id, headless)`` is injectable for tests; the default
    builds the live BrowserGym WebArena environment.
    """
    from ..envs.harness import RevActEnv, make_env

    base = config.WA_SHOPPING
    if not base:
        raise PointExecutionError("WA_SHOPPING is not set "
                                  "(source scripts/export_webarena_env.sh)")
    key_states = load_key_states(Path(data_root) / "raw" / "state_bank")
    factory = env_factory or (lambda task_id, headless_flag: make_env(
        task_id, headless=headless_flag))

    ordered: dict[tuple[str, int], list[ProbeExecutionSpec]] = {}
    for spec in specs:
        ordered.setdefault((spec.task_id, int(spec.seed or 0)), []).append(spec)

    results: list[tuple[ReversibilityResult, Any]] = []
    failures: list[dict] = []
    from .base import ProbeContext

    for (task_id, seed), group in ordered.items():
        env = factory(task_id, headless)
        renv = RevActEnv(env, task_id=task_id, site="shopping")
        try:
            for spec in group:
                ks = key_states.get(spec.state_id)
                started = time.time()
                try:
                    if ks is None:
                        raise PointExecutionError(
                            f"unknown state_id {spec.state_id!r} in the state bank")
                    result = run_point_spec(renv, spec, ks, base, commit=commit)
                    _check_review_gates(spec, result)
                except PointExecutionError as exc:
                    failures.append({"probe_point_id": spec.probe_point_id,
                                     "state_id": spec.state_id,
                                     "action_type": spec.action_type,
                                     "reason": str(exc)})
                    continue
                ctx = ProbeContext(
                    renv=renv, base=base, commit=commit, budget=int(spec.budget_k),
                    probe_point_id=spec.probe_point_id,
                    probe_run_id=spec.probe_run_id,
                    state_id=spec.state_id,
                    candidate_id=spec.candidate_id,
                    candidate_snapshot_hash=spec.candidate_snapshot_hash,
                    action_instance_id=spec.action_instance_id,
                    raw_action=spec.raw_action,
                    canonical_action=spec.canonical_action,
                    environment_family=spec.environment_family,
                    environment_instance=spec.environment_instance,
                    environment_origin=spec.environment_origin,
                    is_mock=spec.is_mock,
                    task_id=spec.task_id,
                    trajectory_id=spec.trajectory_id,
                    run_id=spec.run_id,
                    seed=spec.seed,
                    url=spec.url,
                    account=spec.account,
                    privilege=spec.privilege,
                    solver_set=list(result.solver_set),
                    code_version=spec.code_version,
                )
                results.append((result, ctx))
                print(f"  [point] {spec.probe_point_id} {spec.action_type:22s} "
                      f"-> {result.effect_status}/{result.recovery_status} "
                      f"cost={result.undo_cost_steps} "
                      f"({time.time() - started:.1f}s)")
        finally:
            renv.close()
    return {"results": results, "failures": failures}
