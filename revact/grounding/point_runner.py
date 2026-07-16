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
import tempfile
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import fcntl

from .. import config, prompts
from ..envs.obs_utils import (bid_is_visible, extract_interactive_bids,
                              prune_axtree_txt)
from . import signals
from .authoring import ProbeExecutionSpec
from .backend_observers import (OBSERVER_VERSION, BackendSignalObserver,
                                build_live_backend_observer)
from .base import ReversibilityResult, destructive_allowed
from .schema import (EFFECT_CHANGED, EFFECT_NO_EFFECT, RECOVERY_PARTIAL, RECOVERY_RECOVERED, RECOVERY_UNKNOWN)
from .solvers import (SOLVER_BFS, SOLVER_DETERMINISTIC, SOLVER_LLM_ATTACKER,
                      SolverTrace, combine_solver_traces)
from .signal_evidence import materialize_signal_evidence
from .transitions import (TRANSITION_SCHEMA_VERSION, ExecutedTransitionStep,
                          ObservationBody, ProbeTransition,
                          transition_manifest_row)


class PointExecutionError(RuntimeError):
    """A spec could not be executed as reviewed (recorded, never papered over)."""


def _sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _pruned(renv, *, anchor_bids: list[str] | None = None) -> str:
    return prune_axtree_txt(
        renv._last_obs_view.get("axtree_txt", ""),
        anchor_bids=anchor_bids,
    )


def _capture_observation(renv, stage: str, *,
                         anchor_bids: list[str] | None = None
                         ) -> ObservationBody:
    """Capture both the exact AXTree and the target-aware policy projection."""
    return ObservationBody.capture(
        dict(renv._last_obs_view or {}), stage=stage,
        anchor_bids=anchor_bids or [])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_journal(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        # Persist the directory entry as well as the file contents.  A live
        # mutation must never outrun the RECOVERY_PENDING marker on a host
        # crash between rename and journal discovery.
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _update_journal(path: Path, **updates: Any) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload.update(updates)
    payload["updated_at"] = _now()
    _atomic_journal(path, payload)
    return payload


@contextmanager
def _account_lock(data_root: Path, site: str, account: str):
    """Serialize live mutations sharing one site/account."""
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "-"
                   for ch in f"{site}__{account or 'default'}")
    path = Path(data_root) / "grounded" / "recovery_journal" / f"{safe}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PointExecutionError(
                f"another live batch holds account lock {path.name}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


# --------------------------------------------------------------------------- #
# Signal adapters (canonical JSON forms so dict equality is meaningful)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PointAdapter:
    action_type: str
    signal_name: str
    semantic_undo: str                     # semantic IR name of one undo step
    signal_page: Callable[[str, ProbeExecutionSpec], str]
    read: Callable[[Any, str], dict]       # (renv, url) -> canonical signal
    reset_baseline: bool                   # require an isolated clean baseline?
    site: str = "shopping"


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


def _orders_read(renv, url: str, *, navigate: bool = True) -> dict:
    ids = signals.order_ids(renv, url, navigate=navigate)
    return {"signal": "order_id_set", "order_ids": sorted(ids)}


def _reddit_vote_read(renv, url: str, *, navigate: bool = True) -> dict:
    return signals.reddit_vote_canonical(renv, url, navigate=navigate)


def _reddit_subscribe_read(renv, url: str, *, navigate: bool = True) -> dict:
    return signals.reddit_subscribe_canonical(renv, url, navigate=navigate)


ADAPTERS: dict[str, PointAdapter] = {
    "add_to_cart": PointAdapter(
        "add_to_cart", "cart_item_count", "remove_cart_line",
        lambda base, _spec: base + config.SHOPPING_PATHS["cart"], _cart_read, True),
    "wishlist_add": PointAdapter(
        "wishlist_add", "wishlist_item_count", "remove_wishlist_item",
        lambda base, _spec: base + config.SHOPPING_PATHS["wishlist"], _wishlist_read, True),
    "compare_add": PointAdapter(
        "compare_add", "compare_item_count", "remove_compare_item",
        lambda base, _spec: base + config.SHOPPING_PATHS["compare"], _compare_read, True),
    "newsletter_subscribe": PointAdapter(
        "newsletter_subscribe", "newsletter_subscription_checkbox",
        "toggle_newsletter_subscription",
        lambda base, _spec: base + config.SHOPPING_PATHS["newsletter"],
        _newsletter_read, False),
    "address_add": PointAdapter(
        "address_add", "deletable_address_count", "delete_probe_created_address",
        lambda base, _spec: base + config.SHOPPING_PATHS["address"], _address_read, False),
    "place_order": PointAdapter(
        "place_order", "order_id_set", "cancel_order",
        lambda base, _spec: base + config.SHOPPING_PATHS["order_history"],
        _orders_read, False),
    "reddit_vote": PointAdapter(
        "reddit_vote", "reddit_own_vote_state", "toggle_reddit_upvote",
        lambda _base, spec: spec.submission_url or spec.url,
        _reddit_vote_read, False, site="reddit"),
    "reddit_subscribe": PointAdapter(
        "reddit_subscribe", "reddit_forum_subscription",
        "toggle_forum_subscription",
        lambda _base, spec: spec.forum_url or spec.url,
        _reddit_subscribe_read, False, site="reddit"),
}


# --------------------------------------------------------------------------- #
# Recording deterministic recovery controllers
# --------------------------------------------------------------------------- #
@dataclass
class RecoverySteps:
    raw: list[str] = field(default_factory=list)
    semantic: list[str] = field(default_factory=list)
    obs_hashes: list[str] = field(default_factory=list)
    observations: list[ObservationBody] = field(default_factory=list)
    step_kinds: list[str] = field(default_factory=list)
    budget_exhausted: bool = False
    termination_reason: str = ""

    def record(self, renv, raw_action: str, semantic: str) -> None:
        renv.step(raw_action)
        from ..train.validators import parse_action
        parsed = parse_action(raw_action)
        anchors = [parsed.bid] if parsed is not None and parsed.bid else []
        observation = _capture_observation(
            renv, f"recovery_after_{len(self.raw)}", anchor_bids=anchors)
        self.raw.append(raw_action)
        self.semantic.append(semantic)
        self.obs_hashes.append(observation.policy_axtree_sha256)
        self.observations.append(observation)
        self.step_kinds.append("recovery")


def _confirm_modal(renv, steps: RecoverySteps, semantic: str,
                   budget: int) -> None:
    if len(steps.raw) >= budget:
        return
    ok = signals.find_action_by_text(renv._last_obs_view, ["ok"], roles=("button",))
    if ok:
        steps.record(renv, f"click('{ok['bid']}')", semantic)


def _drain_recovery(renv, adapter: PointAdapter, url: str, budget: int,
                    remove_key: str, *, initial: RecoverySteps | None = None
                    ) -> tuple[dict, RecoverySteps]:
    """Click per-row remove links until the count signal is empty/budget out."""
    steps = initial or RecoverySteps()
    raw_sig = {"cart_item_count": signals.cart_signal,
               "wishlist_item_count": signals.wishlist_signal,
               "compare_item_count": signals.compare_signal}[adapter.signal_name]
    sig = raw_sig(renv, url, navigate=initial is None)
    while sig["remove_bids"] and len(steps.raw) < budget:
        steps.record(renv, f"click('{sig['remove_bids'][0]}')", adapter.semantic_undo)
        _confirm_modal(renv, steps, "confirm_modal_ok", budget)
        sig = raw_sig(renv, url, navigate=False)
    steps.budget_exhausted = bool(sig["remove_bids"]) and len(steps.raw) >= budget
    steps.termination_reason = ("budget_exhausted" if steps.budget_exhausted
                                else "signal_restored")
    return {"signal": adapter.signal_name, "count": sig["count"]}, steps


def _newsletter_recovery(renv, url: str, target_checked: bool,
                         budget: int, *, initial: RecoverySteps | None = None
                         ) -> tuple[dict, RecoverySteps]:
    steps = initial or RecoverySteps()
    sig = signals.newsletter_signal(renv, url, navigate=initial is None)
    if (sig["found"] and sig["checked"] != target_checked and
            len(steps.raw) < budget):
        steps.record(renv, f"click('{sig['bid']}')",
                     "toggle_newsletter_subscription")
        save = signals.find_action_by_text(renv._last_obs_view, ["save"])
        if save and len(steps.raw) < budget:
            steps.record(renv, f"click('{save['bid']}')", "save_newsletter_form")
        sig = signals.newsletter_signal(renv, url, navigate=False)
    restored = sig["found"] and sig["checked"] == target_checked
    steps.budget_exhausted = not restored and len(steps.raw) >= budget
    steps.termination_reason = ("signal_restored" if restored else
                                "budget_exhausted" if steps.budget_exhausted
                                else "recovery_control_missing")
    return {"signal": "newsletter_subscription_checkbox",
            "found": sig["found"], "checked": sig["checked"]}, steps


def _reddit_vote_recovery(renv, url: str, pre: dict, budget: int, *,
                          initial: RecoverySteps | None = None
                          ) -> tuple[dict, RecoverySteps]:
    """Restore a neutral own-vote state; public score drift is telemetry only."""
    steps = initial or RecoverySteps()
    raw = signals.vote_score(renv, url, navigate=initial is None)
    if (raw.get("vote_direction") != pre.get("vote_direction") and
            len(steps.raw) < budget):
        bid = raw.get("up_bid") if raw.get("vote_direction") == 1 \
            else raw.get("down_bid")
        if bid:
            steps.record(renv, f"click('{bid}')", "toggle_reddit_upvote")
        raw = signals.vote_score(renv, url, navigate=False)
    final = {
        "signal": "reddit_own_vote_state",
        "submission_id": raw.get("submission_id"),
        "vote_direction": raw.get("vote_direction"),
    }
    restored = final == pre
    steps.budget_exhausted = not restored and len(steps.raw) >= budget
    steps.termination_reason = (
        "signal_restored" if restored else
        "budget_exhausted" if steps.budget_exhausted else
        "recovery_control_missing")
    return final, steps


def _reddit_subscription_recovery(renv, url: str, pre: dict, budget: int, *,
                                  initial: RecoverySteps | None = None
                                  ) -> tuple[dict, RecoverySteps]:
    steps = initial or RecoverySteps()
    raw = signals.subscribe_signal(renv, url, navigate=initial is None)
    if (raw.get("found") and raw.get("subscribed") != pre.get("subscribed") and
            len(steps.raw) < budget):
        steps.record(renv, f"click('{raw['bid']}')",
                     "toggle_forum_subscription")
        raw = signals.subscribe_signal(renv, url, navigate=False)
    final = {
        "signal": "reddit_forum_subscription",
        "forum": raw.get("forum"),
        "found": raw.get("found"),
        "subscribed": raw.get("subscribed"),
    }
    restored = final == pre
    steps.budget_exhausted = not restored and len(steps.raw) >= budget
    steps.termination_reason = (
        "signal_restored" if restored else
        "budget_exhausted" if steps.budget_exhausted else
        "recovery_control_missing")
    return final, steps


def _address_recovery(renv, url: str, budget: int, *,
                      initial: RecoverySteps | None = None
                      ) -> tuple[dict, RecoverySteps]:
    """Delete the LAST address entry (the one this probe just created)."""
    steps = initial or RecoverySteps()
    sig = signals.address_signal(renv, url, navigate=initial is None)
    if sig["delete_bids"] and len(steps.raw) < budget:
        steps.record(renv, f"click('{sig['delete_bids'][-1]}')",
                     "delete_probe_created_address")
        _confirm_modal(renv, steps, "confirm_modal_ok", budget)
        sig = signals.address_signal(renv, url, navigate=False)
    steps.budget_exhausted = bool(sig["delete_bids"]) and len(steps.raw) >= budget
    steps.termination_reason = ("budget_exhausted" if steps.budget_exhausted
                                else "recovery_attempt_complete")
    return {"signal": "deletable_address_count", "count": sig["count"]}, steps


def _order_cancel_recovery(renv, url: str, budget: int,
                           pre: dict, *, initial: RecoverySteps | None = None
                           ) -> tuple[dict, RecoverySteps]:
    """Deterministic customer-side controller: look for an actionable cancel."""
    steps = initial or RecoverySteps()
    if initial is None:
        steps.record(renv, f"goto('{url}')", "navigate_to_signal_page")
    cancel = signals.find_action_by_text(
        renv._last_obs_view, ["cancel order", "cancel this", "cancel"])
    if cancel and len(steps.raw) < budget:
        steps.record(renv, f"click('{cancel['bid']}')", "cancel_order")
        _confirm_modal(renv, steps, "confirm_modal_ok", budget)
    steps.termination_reason = ("cancel_attempted" if cancel
                                else "no_cancel_affordance")
    final = _orders_read(renv, url, navigate=False)
    steps.budget_exhausted = final != pre and len(steps.raw) >= budget
    return final, steps


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
            path_raw: list[str] = []
            path_hashes: list[str] = []
            for name in path:
                el = signals.find_action_by_text(renv._last_obs_view, [name])
                if el is None:
                    replay_failed = True
                    break
                raw = f"click('{el['bid']}')"
                renv.step(raw)
                executed += 1
                path_raw.append(raw)
                path_hashes.append(_sha256(_pruned(renv)))
            if replay_failed:
                continue
            for el in relevant(renv._last_obs_view):
                if executed >= budget:
                    break
                name = el["line"].split("'")[1] if "'" in el["line"] else el["bid"]
                raw = f"click('{el['bid']}')"
                renv.step(raw)
                executed += 1
                explored_actions.append(f"click:{name}")
                explored_states.append(_sha256(_pruned(renv)))
                current_hash = _sha256(_pruned(renv))
                if _orders_read(renv, url) == pre:
                    # A live BFS success would be promoted by the caller; the
                    # trace stays honest either way.
                    return SolverTrace(
                        solver_name="affordance_bfs.live-depth2",
                        solver_kind=SOLVER_BFS, solver_version="live-1",
                        budget_k=budget, success=True,
                        explored_actions=tuple(explored_actions),
                        explored_states=tuple(explored_states),
                        undo_actions=tuple(path_raw + [raw]),
                        undo_semantic_actions=tuple(
                            ["click_control"] * (len(path) + 1)),
                        undo_observation_hashes=tuple(
                            path_hashes + [current_hash]),
                        final_signal=_orders_read(renv, url, navigate=False),
                        residual_diff={},
                        termination_reason="recovered",
                        budget_exhausted=False,
                        evidence={
                            "forbidden_controls": list(_BFS_FORBIDDEN),
                            "branch_reset": "url_reanchor_only",
                        })
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
                  "executed_actions": executed,
                  "branch_reset": "url_reanchor_only"})


def _attacker_action_error(action: str, view: dict, measurement_url: str
                           ) -> str | None:
    """Hard safety/legality gate for model-proposed undo actions.

    Prompt text is not a security boundary.  The attacker may only click a
    currently visible non-commit control, navigate within the same origin, or
    go back.  All other BrowserGym primitives are rejected before ``step``.
    """
    from ..train.validators import parse_action

    parsed = parse_action(action)
    if parsed is None:
        return "unparseable_action"
    if parsed.name == "click":
        if not parsed.bid:
            return "click_missing_bid"
        matches = [el for el in extract_interactive_bids(
            view.get("axtree_txt", "")) if el["bid"] == parsed.bid]
        if len(matches) != 1:
            return "click_bid_not_currently_legal"
        line = matches[0]["line"].lower()
        if any(term in line for term in _BFS_FORBIDDEN):
            return "commit_like_control_forbidden"
        return None
    if parsed.name == "goto":
        if len(parsed.args) != 1 or not isinstance(parsed.args[0], str):
            return "goto_requires_one_literal_url"
        wanted = urlparse(parsed.args[0])
        allowed = urlparse(measurement_url)
        if (wanted.scheme, wanted.netloc) != (allowed.scheme, allowed.netloc):
            return "cross_origin_goto_forbidden"
        return None
    if parsed.name == "go_back" and not parsed.args:
        return None
    return f"primitive_forbidden:{parsed.name}"


def _llm_attacker_trace(renv, adapter: PointAdapter, url: str, budget: int,
                        pre: dict, *, seeds: tuple[int, ...] = (0, 1),
                        model: str | None = None) -> SolverTrace:
    """Multi-seed LLM undo attacker over the live page (OpenAI-compatible)."""
    api_base = os.environ.get("REVACT_LLM_BASE_URL",
                              "https://api.deepseek.com/v1").rstrip("/")
    api_key_env = os.environ.get("REVACT_LLM_API_KEY_ENV",
                                 "DEEPSEEK_API_KEY")
    api_key = os.environ.get(api_key_env, "").strip()
    model = model or os.environ.get("REVACT_LLM_MODEL", "deepseek-chat")
    if not api_key:
        raise PointExecutionError(
            f"LLM undo attacker requires an API key ({api_key_env})")

    explored_actions: list[str] = []
    explored_states: list[str] = []
    attempts = 0
    final = pre
    per_seed = max(1, budget // (2 * len(seeds)))
    new_orders: list[str] = []
    prompt_provenance = prompts.snapshot_generation(
        root=config.DATA_ROOT, author="point-undo-attacker",
        producer="revact.grounding.point_runner._llm_attacker_trace",
        model={"provider": "openai-compatible", "name": model},
        decode_config={
            "seeds": list(seeds),
            "temperatures": [0.2 + 0.6 * seed for seed in seeds],
            "max_tokens": 200,
            "budget_k": budget,
        })

    for seed in seeds:
        renv.step(f"goto('{url}')")
        final = _orders_read(renv, url)
        new_orders = sorted(set(final.get("order_ids", [])) -
                            set(pre.get("order_ids", [])))
        transcript = []
        for _step in range(per_seed):
            obs = prune_axtree_txt(renv._last_obs_view.get("axtree_txt", ""),
                                   max_chars=6000)
            user = prompts.get("undo_attacker_user").format(
                pre_order_ids=pre.get("order_ids", []),
                new_order_ids=new_orders,
                observation=obs,
                history=transcript,
            )
            payload = json.dumps({
                "model": model,
                "messages": [{"role": "system", "content": prompts.get(
                    "undo_attacker_system")},
                             {"role": "user", "content": user}],
                "temperature": 0.2 + 0.6 * seed, "max_tokens": 200,
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{api_base}/chat/completions", data=payload,
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"})
            try:
                attempts += 1
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                text = body["choices"][0]["message"]["content"] or ""
            except Exception as exc:  # network/provider error = recorded stop
                raise PointExecutionError(
                    "LLM undo attacker provider request failed; refusing to "
                    f"treat an incomplete route as negative evidence "
                    f"({type(exc).__name__})") from exc
            from ..policies import parse_action as parse_llm_action
            action = parse_llm_action(text)
            if not action or action.startswith(("give_up", "send_msg")):
                explored_actions.append(f"seed{seed}:give_up")
                break
            rejection = _attacker_action_error(
                action, renv._last_obs_view, url)
            if rejection:
                explored_actions.append(
                    f"seed{seed}:rejected:{rejection}:{action}")
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
                    evidence={"model": model, **prompt_provenance})
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
        evidence={"model": model, "per_seed_step_budget": per_seed,
                  **prompt_provenance})


# --------------------------------------------------------------------------- #
# Core point protocol
# --------------------------------------------------------------------------- #
def _target_contract(axtree: str, bid: str) -> dict[str, str] | None:
    """Stable identity of one reviewed control inside a dynamic page."""
    from ..data.candidates import (canonical_click_for_element,
                                   interactive_elements)

    matches = [row for row in interactive_elements(axtree)
               if row["bid"] == str(bid)]
    if len(matches) != 1:
        return None
    row = matches[0]
    return {
        "bid": row["bid"], "role": row["role"], "name": row["name"],
        "canonical_action": canonical_click_for_element(row),
    }


def _replay_to_state(renv, spec: ProbeExecutionSpec, ks: dict
                     ) -> tuple[str, dict[str, Any]]:
    """Reset+replay and verify either exact or explicit target-scoped identity.

    Reddit listing/submission bodies can reorder because other users vote or
    because the site changes ranking.  Full-page byte equality is therefore
    retained as the preferred path, while a narrowly enumerated Reddit action
    may use a recorded target contract: exact URL, exact bid, role/name and
    canonical click must all remain identical.  The drift is written into the
    point evidence; it is never a silent snapshot rebase.
    """
    renv.reset(seed=int(spec.seed or 0), trajectory_id=spec.trajectory_id,
               run_id=spec.run_id)
    for action in ks.get("replay_prefix") or []:
        renv.step(action)
    from ..train.validators import parse_action
    parsed = parse_action(spec.raw_action)
    anchors = [parsed.bid] if parsed is not None and parsed.bid else []
    full_current = renv._last_obs_view.get("axtree_txt", "") or ""
    current = prune_axtree_txt(full_current, anchor_bids=anchors)
    expected = ks.get("axtree_snapshot") or ""
    current_hash = _sha256(current)
    expected_hash = _sha256(expected)
    exact = current == expected
    replay_evidence: dict[str, Any] = {
        "replay_snapshot_exact": exact,
        "replay_recorded_observation_hash": expected_hash,
        "replay_current_observation_hash": current_hash,
        "replay_verification": "exact_snapshot",
    }
    if not exact:
        dynamic_action = spec.action_type in {
            "reddit_vote", "reddit_subscribe",
        }
        expected_url = str(ks.get("url") or spec.url).rstrip("/")
        current_url = str(renv._last_obs_view.get("url") or "").rstrip("/")
        bid = anchors[0] if anchors else ""
        recorded_target = _target_contract(expected, bid)
        current_target = _target_contract(full_current, bid)
        target_matches = (
            recorded_target is not None and
            current_target == recorded_target and
            current_target.get("canonical_action") == spec.canonical_action
        )
        if not (dynamic_action and current_url == expected_url and target_matches):
            raise PointExecutionError(
                "replayed state does not reproduce the recorded snapshot or "
                "its reviewed dynamic-page target contract "
                f"(replayed sha={current_hash[:12]}, "
                f"recorded sha={expected_hash[:12]}, "
                f"url_match={current_url == expected_url}, "
                f"target_match={target_matches}); refusing to rebase")
        replay_evidence.update({
            "replay_verification": "dynamic_page_target_contract",
            "replay_target_contract": current_target,
            "replay_url": current_url,
            "replay_drift_recorded_chars": len(expected),
            "replay_drift_current_chars": len(current),
        })
    return current_hash, replay_evidence


def _execute_candidate(renv, spec: ProbeExecutionSpec) -> str:
    obs = renv._last_obs_view.get("axtree_txt", "")
    from ..train.validators import parse_action
    parsed = parse_action(spec.raw_action)
    if parsed is None or not parsed.bid:
        raise PointExecutionError(f"unsupported raw action {spec.raw_action!r}")
    if not bid_is_visible(
            prune_axtree_txt(obs, anchor_bids=[parsed.bid]), parsed.bid):
        raise PointExecutionError(
            f"candidate bid [{parsed.bid}] is not visible at the replayed state")
    renv.step(spec.raw_action)
    return _sha256(_pruned(renv, anchor_bids=[parsed.bid]))


def _attach_transition(result: ReversibilityResult, spec: ProbeExecutionSpec,
                       pre_observation: ObservationBody,
                       post_observation: ObservationBody,
                       post_signal_observation: ObservationBody,
                       recovery_steps: RecoverySteps | None,
                       replay_evidence: dict[str, Any]) -> ReversibilityResult:
    """Validate and attach an out-of-band observation-body record.

    ``ReversibilityResult`` remains the public return type for backwards
    compatibility.  The batch orchestrator reads the private attachment and
    persists it in the separate immutable transition artifact; dataclass
    serialization of smoke results cannot accidentally inline AXTree bodies.
    """
    steps = recovery_steps or RecoverySteps()
    if not (len(steps.raw) == len(steps.semantic) == len(steps.obs_hashes) ==
            len(steps.observations) == len(steps.step_kinds)):
        raise PointExecutionError(
            "recovery observation bodies do not align 1:1 with executed actions")
    executed = [
        ExecutedTransitionStep(
            step_index=index,
            raw_action=raw,
            semantic_action=semantic,
            observation_after=observation,
            step_kind=kind,
        )
        for index, (raw, semantic, observation, kind) in enumerate(zip(
            steps.raw, steps.semantic, steps.observations, steps.step_kinds))
    ]
    final_signal = (result.evidence.get("final_signal")
                    if result.effect_status == EFFECT_CHANGED
                    else result.evidence.get("post_signal"))
    transition = ProbeTransition(
        schema_version=TRANSITION_SCHEMA_VERSION,
        transition_id=f"transition::{spec.probe_point_id}",
        probe_point_id=spec.probe_point_id,
        probe_run_id=spec.probe_run_id,
        state_id=spec.state_id,
        candidate_id=spec.candidate_id,
        action_instance_id=spec.action_instance_id,
        action_type=spec.action_type,
        raw_action=spec.raw_action,
        canonical_action=spec.canonical_action,
        candidate_snapshot_hash=spec.candidate_snapshot_hash,
        pre_observation=pre_observation,
        post_observation=post_observation,
        post_signal_observation=post_signal_observation,
        recovery_steps=executed,
        pre_signal=result.evidence.get("pre_signal"),
        post_signal=result.evidence.get("post_signal"),
        final_signal=final_signal,
        effect_status=result.effect_status,
        recovery_status=result.recovery_status,
        undo_cost_steps=result.undo_cost_steps,
        budget_k=int(result.budget_k or spec.budget_k),
        replay_verification=str(
            replay_evidence.get("replay_verification") or ""),
        replay_target_contract=dict(
            replay_evidence.get("replay_target_contract") or {}),
        timestamp=result.timestamp,
        code_version=spec.code_version,
    )
    transition.validate()
    if result.evidence.get("pre_observation_hash") != \
            pre_observation.policy_axtree_sha256:
        raise PointExecutionError("pre observation body contradicts point hash")
    if result.evidence.get("post_observation_hash") != \
            post_observation.policy_axtree_sha256:
        raise PointExecutionError("post observation body contradicts point hash")
    result.evidence["transition_ref"] = {
        "schema_version": TRANSITION_SCHEMA_VERSION,
        "transition_id": transition.transition_id,
        "probe_point_id": transition.probe_point_id,
        "record_sha256": transition_manifest_row(transition)["record_sha256"],
    }
    # Dataclasses without slots permit this non-wire attribute.  ``asdict`` on
    # ReversibilityResult ignores it, keeping bodies in their dedicated asset.
    setattr(result, "_probe_transition", transition)
    return result


def _attach_backend_observations(
        result: ReversibilityResult,
        observer: BackendSignalObserver | None,
        observations: dict[str, dict[str, Any]],
        ) -> ReversibilityResult:
    """Attach an in-memory three-phase trace for batch-level materialization.

    Raw observer responses are deliberately not inserted into the point row.
    The batch orchestrator freezes them as content-addressed sidecars and only
    then adds their hashes and normalized summaries to ``result.evidence``.
    """
    if observer is None:
        if observations:
            raise PointExecutionError(
                "backend observations exist without a reviewed observer")
        return result
    if set(observations) != {"pre", "post", "final"}:
        raise PointExecutionError(
            "backend observer must capture exactly pre/post/final")
    setattr(result, "_backend_signal_observer", observer)
    setattr(result, "_backend_signal_observations", dict(observations))
    return result


def _emergency_restore(renv, spec: ProbeExecutionSpec, base: str, pre: dict
                       ) -> tuple[dict, list[str]]:
    """Best-effort deterministic cleanup after an exception post-mutation."""
    adapter = ADAPTERS[spec.action_type]
    url = adapter.signal_page(base, spec)
    budget = int(spec.budget_k)
    if spec.action_type in ("add_to_cart", "wishlist_add", "compare_add"):
        final, steps = _drain_recovery(
            renv, adapter, url, budget, adapter.semantic_undo)
    elif spec.action_type == "newsletter_subscribe":
        final, steps = _newsletter_recovery(
            renv, url, bool(pre.get("checked")), budget)
    elif spec.action_type == "address_add":
        final, steps = _address_recovery(renv, url, budget)
    elif spec.action_type == "reddit_vote":
        final, steps = _reddit_vote_recovery(renv, url, pre, budget)
    elif spec.action_type == "reddit_subscribe":
        final, steps = _reddit_subscription_recovery(renv, url, pre, budget)
    elif spec.action_type == "place_order":
        final, steps = _order_cancel_recovery(renv, url, budget, pre)
    else:  # pragma: no cover - adapter gate
        raise PointExecutionError(
            f"no emergency recovery controller for {spec.action_type!r}")
    return final, list(steps.raw)


def run_point_spec(renv, spec: ProbeExecutionSpec, ks: dict, base: str,
                   *, commit: bool = False,
                   recovery_journal: Path | None = None,
                   backend_observer: BackendSignalObserver | None = None
                   ) -> ReversibilityResult:
    """Execute one reviewed spec end-to-end and return a formal-grade result."""
    adapter = ADAPTERS.get(spec.action_type)
    if adapter is None:
        raise PointExecutionError(
            f"no point adapter for action_type {spec.action_type!r}")
    if adapter.site != spec.site:
        raise PointExecutionError(
            f"action/site adapter mismatch: {spec.action_type!r} belongs to "
            f"{adapter.site!r}, not {spec.site!r}")
    if spec.safety_level == "destructive":
        if not (commit and destructive_allowed()):
            raise PointExecutionError(
                "destructive spec requires --commit and REVACT_ALLOW_DESTRUCTIVE=1")
    url = adapter.signal_page(base, spec)
    allowed_origin = urlparse(base)
    measured_origin = urlparse(url)
    if ((measured_origin.scheme, measured_origin.netloc) !=
            (allowed_origin.scheme, allowed_origin.netloc)):
        raise PointExecutionError(
            f"signal URL {url!r} is outside reviewed site origin {base!r}")
    budget = int(spec.budget_k)
    backend_observations: dict[str, dict[str, Any]] = {}
    if backend_observer is not None:
        if backend_observer.provider.channel not in spec.signal_channels:
            raise PointExecutionError(
                "backend observer channel was not present in the reviewed spec")
        if backend_observer.action_type != spec.action_type:
            raise PointExecutionError(
                "backend observer action_type contradicts the reviewed spec")
        if backend_observer.environment_instance.rstrip("/") != \
                spec.environment_instance.rstrip("/"):
            raise PointExecutionError(
                "backend observer environment contradicts the reviewed spec")

    # 0. BrowserGym forbids navigation before the first reset.  Bootstrap the
    #    authenticated session solely for baseline measurement; exact point
    #    replay below performs its own reset and clears this diagnostic log.
    renv.reset(seed=int(spec.seed or 0),
               trajectory_id=f"{spec.probe_point_id}__baseline",
               run_id=spec.probe_run_id)

    # 1. Verified baseline.  Never "clean" a shared account by deleting all
    #    existing cart/wishlist/compare entities: that would mutate unrelated
    #    user state before the probe and a count-only signal could then hide an
    #    entity swap.  Short-term live probes therefore require an isolated,
    #    already-clean account and fail closed without clicking anything.
    pre_signal = adapter.read(renv, url)
    if adapter.reset_baseline and int(pre_signal.get("count", -1)) != 0:
        raise PointExecutionError(
            f"{spec.action_type} requires an isolated clean baseline; "
            f"observed {pre_signal!r}. Refusing to drain existing account state")
    if spec.action_type == "reddit_vote" and (
            pre_signal.get("vote_direction") != 0 or
            not pre_signal.get("submission_id")):
        raise PointExecutionError(
            f"reddit_vote requires a neutral entity-bound baseline, got {pre_signal!r}")
    if spec.action_type == "reddit_subscribe" and (
            pre_signal.get("found") is not True or
            pre_signal.get("subscribed") is not False or
            not pre_signal.get("forum")):
        raise PointExecutionError(
            "reddit_subscribe requires one exact unsubscribed forum control; "
            f"got {pre_signal!r}")
    if backend_observer is not None:
        # This is the last independent backend read before exact state replay
        # and the candidate action.  Any observer failure is therefore safely
        # pre-mutation.
        backend_observations["pre"] = backend_observer.capture("pre")

    # 2. Replay to the exact recorded decision state.
    pre_hash, replay_evidence = _replay_to_state(renv, spec, ks)
    from ..train.validators import parse_action
    parsed_candidate = parse_action(spec.raw_action)
    candidate_anchors = ([parsed_candidate.bid]
                         if parsed_candidate is not None and parsed_candidate.bid
                         else [])
    pre_observation = _capture_observation(
        renv, "pre_action", anchor_bids=candidate_anchors)
    if pre_observation.policy_axtree_sha256 != pre_hash:
        raise PointExecutionError(
            "captured pre-action observation contradicts replay verification")

    # 3. Execute the reviewed candidate action.
    if recovery_journal is not None:
        _atomic_journal(recovery_journal, {
            "schema_version": "iris.recovery_wal.v1",
            "status": "RECOVERY_PENDING",
            "mutation_attempted": True,
            "probe_point_id": spec.probe_point_id,
            "site": spec.site, "account": spec.account,
            "action_type": spec.action_type, "raw_action": spec.raw_action,
            "signal_url": url, "pre_signal": pre_signal,
            "budget_k": budget, "updated_at": _now(),
        })
    post_hash = _execute_candidate(renv, spec)
    post_observation = _capture_observation(
        renv, "post_action_immediate", anchor_bids=candidate_anchors)
    if post_observation.policy_axtree_sha256 != post_hash:
        raise PointExecutionError(
            "captured immediate post-action observation contradicts execution hash")

    # 4. Measure the post-action signal.
    post_signal = adapter.read(renv, url)
    post_signal_observation = _capture_observation(
        renv, "post_signal_measurement")
    effect = EFFECT_CHANGED if post_signal != pre_signal else EFFECT_NO_EFFECT
    if backend_observer is not None:
        backend_observations["post"] = backend_observer.capture("post")

    evidence: dict[str, Any] = {
        "protocol": "point_runner.v2",
        "signal_channels": ["ui_structural"],
        "candidate_snapshot_hash": spec.candidate_snapshot_hash,
        "authored_spec_id": spec.authored_spec_id,
        "pre_observation_hash": pre_hash,
        "pre_signal": pre_signal,
        "post_observation_hash": post_hash,
        "post_signal": post_signal,
        "replay_prefix_len": len(ks.get("replay_prefix") or []),
        "baseline_protocol": (
            "verified_clean_refuse_nonempty" if adapter.reset_baseline
            else "measured_without_cleanup"),
        "backend_commit": False,
        **replay_evidence,
    }

    if effect == EFFECT_NO_EFFECT:
        evidence["solver_traces"] = [{
            "solver_name": "effect_measurement.no_recovery_applicable",
            "executed": True,
            "reason": "measured action produced NO_EFFECT",
        }]
        result = ReversibilityResult(
            action_type=spec.action_type, label="NO_EFFECT",
            grounding=adapter.signal_name, destructive=False,
            evidence=evidence, probe_id=spec.probe_point_id,
            timestamp=_now(), commit_mode=commit, site=spec.site,
            probe_name=spec.probe_name, effect_status=EFFECT_NO_EFFECT,
            recovery_status=RECOVERY_UNKNOWN, undo_cost_steps=None,
            budget_k=budget,
            solver_set=["effect_measurement.no_recovery_applicable"],
            budget_exhausted=False, controller_version=spec.controller_version)
        if backend_observer is not None:
            backend_observations["final"] = backend_observer.capture("final")
        _attach_backend_observations(
            result, backend_observer, backend_observations)
        return _attach_transition(
            result, spec, pre_observation, post_observation,
            post_signal_observation, None,
            replay_evidence)

    # 5. Deterministic recovery controller (recorded step by step).
    # ``adapter.read`` navigated from the post-action page to the structural
    # signal page.  That browser action is part of the operational undo path,
    # not free instrumentation, so it is recorded and charged to budget_k.
    signal_navigation = f"goto('{url}')"
    signal_observation = post_signal_observation
    initial = RecoverySteps(
        raw=[signal_navigation], semantic=["navigate_to_signal_page"],
        obs_hashes=[signal_observation.policy_axtree_sha256],
        observations=[signal_observation],
        step_kinds=["measurement_navigation"])
    if spec.action_type in ("add_to_cart", "wishlist_add", "compare_add"):
        final_signal, steps = _drain_recovery(renv, adapter, url, budget,
                                              adapter.semantic_undo,
                                              initial=initial)
    elif spec.action_type == "newsletter_subscribe":
        final_signal, steps = _newsletter_recovery(
            renv, url, bool(pre_signal.get("checked")), budget,
            initial=initial)
    elif spec.action_type == "address_add":
        final_signal, steps = _address_recovery(
            renv, url, budget, initial=initial)
    elif spec.action_type == "reddit_vote":
        final_signal, steps = _reddit_vote_recovery(
            renv, url, pre_signal, budget, initial=initial)
    elif spec.action_type == "reddit_subscribe":
        final_signal, steps = _reddit_subscription_recovery(
            renv, url, pre_signal, budget, initial=initial)
    elif spec.action_type == "place_order":
        evidence["backend_commit"] = True
        final_signal, steps = _order_cancel_recovery(
            renv, url, budget, pre_signal, initial=initial)
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
        result = ReversibilityResult(
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
        if backend_observer is not None:
            backend_observations["final"] = backend_observer.capture("final")
        _attach_backend_observations(
            result, backend_observer, backend_observations)
        return _attach_transition(
            result, spec, pre_observation, post_observation,
            post_signal_observation, steps,
            replay_evidence)
    else:
        recovery = RECOVERY_PARTIAL
        residual = {"pre_signal": pre_signal, "final_signal": final_signal}

    evidence["final_signal"] = final_signal
    evidence["residual_diff"] = residual
    evidence["undo_actions"] = list(steps.raw)
    evidence["undo_semantic_actions"] = list(steps.semantic)
    evidence["undo_observation_hashes"] = list(steps.obs_hashes)
    executed_solver = "site_specific_deterministic.live-1"
    evidence["solver_traces"] = [{
        "solver_name": executed_solver,
        "executed": True,
        "success": recovery == RECOVERY_RECOVERED,
        "budget_k": budget,
        "termination_reason": steps.termination_reason,
        "undo_actions": list(steps.raw),
    }]
    result = ReversibilityResult(
        action_type=spec.action_type, label=recovery,
        grounding=adapter.signal_name,
        destructive=spec.safety_level == "destructive",
        evidence=evidence, probe_id=spec.probe_point_id,
        timestamp=_now(), commit_mode=commit, site=spec.site,
        probe_name=spec.probe_name, effect_status=EFFECT_CHANGED,
        recovery_status=recovery,
        undo_cost_steps=len(steps.raw),
        budget_k=budget, solver_set=[executed_solver],
        budget_exhausted=steps.budget_exhausted,
        controller_version=spec.controller_version)
    if backend_observer is not None:
        backend_observations["final"] = backend_observer.capture("final")
    _attach_backend_observations(result, backend_observer, backend_observations)
    return _attach_transition(
        result, spec, pre_observation, post_observation,
        post_signal_observation, steps,
        replay_evidence)


# --------------------------------------------------------------------------- #
# Batch orchestration
# --------------------------------------------------------------------------- #
def load_key_states(state_bank_dir: Path, *, wanted: set[str] | None = None
                    ) -> dict[str, dict]:
    """Load uniquely identified key states, optionally scoped to one batch.

    Legacy state banks contain unrelated ambiguous ids that remain quarantined
    diagnostics.  A point batch must fail on ambiguity in any state it cites,
    but should not be coupled to every unrelated legacy row in the repository.
    """
    rows: dict[str, dict] = {}
    for path in sorted(Path(state_bank_dir).glob("*.jsonl")):
        # This file is a derived assembler view joined from canonical points,
        # transitions and raw collection rows.  Treating it as a second raw
        # key-state source makes every already-materialized state ambiguous
        # and prevents a fresh, immutable transition recapture from starting.
        # The same exclusion is enforced by batch_prepare and governance.
        if path.name == "formal_point_reached_states.jsonl":
            continue
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get("state_id") or "")
            if not sid or "replay_prefix" not in row:
                continue
            if wanted is not None and sid not in wanted:
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
    if result.effect_status == EFFECT_CHANGED:
        undeclared = sorted(set(result.solver_set) - set(spec.solver_set))
        if undeclared:
            raise PointExecutionError(
                f"executed solver routes were not reviewed: {undeclared}")


def run_point_specs(specs: list[ProbeExecutionSpec], data_root: Path, *,
                    commit: bool = False, headless: bool = True,
                    env_factory=None, backend_observer_factory=None) -> dict:
    """Execute reviewed specs grouped per source task; return results+failures.

    ``env_factory(task_id, headless)`` is injectable for tests; the default
    builds the live BrowserGym WebArena environment.
    """
    from ..envs.harness import RevActEnv, make_env

    wanted = {spec.state_id for spec in specs}
    key_states = load_key_states(
        Path(data_root) / "raw" / "state_bank", wanted=wanted)
    factory = env_factory or (lambda task_id, headless_flag: make_env(
        task_id, headless=headless_flag))

    ordered: dict[tuple[str, str, int], list[ProbeExecutionSpec]] = {}
    for spec in specs:
        ordered.setdefault(
            (spec.site, spec.task_id, int(spec.seed or 0)), []).append(spec)

    results: list[tuple[ReversibilityResult, Any]] = []
    transitions: list[ProbeTransition] = []
    failures: list[dict] = []
    from .base import ProbeContext

    for (site, task_id, seed), group in ordered.items():
        base = config.site_base(site)
        if not base:
            registered = config.SITES.get(site)
            env_name = registered.base_env if registered else "WA_*"
            raise PointExecutionError(
                f"{env_name} is not set for point site {site!r}")
        reviewed_instances = {spec.environment_instance.rstrip("/") for spec in group}
        if reviewed_instances != {base.rstrip("/")}:
            raise PointExecutionError(
                f"reviewed environment_instance does not match live {site} base: "
                f"spec={sorted(reviewed_instances)} live={base!r}")
        env = factory(task_id, headless)
        renv = RevActEnv(env, task_id=task_id, site=site)
        abort_group = False
        try:
            for spec in group:
                if abort_group:
                    failures.append({
                        "probe_point_id": spec.probe_point_id,
                        "state_id": spec.state_id,
                        "action_type": spec.action_type,
                        "reason": (
                            "batch aborted after an earlier recovery failure; "
                            "no action attempted"),
                        "cleanup_status": "NOT_ATTEMPTED_BATCH_ABORTED",
                    })
                    continue
                ks = key_states.get(spec.state_id)
                started = time.time()
                journal_path = (Path(data_root) / "grounded" /
                                "recovery_journal" /
                                f"{spec.probe_point_id}.json")
                try:
                    with _account_lock(Path(data_root), site, spec.account):
                        if ks is None:
                            raise PointExecutionError(
                                f"unknown state_id {spec.state_id!r} in the state bank")
                        if journal_path.exists():
                            prior = json.loads(journal_path.read_text(
                                encoding="utf-8"))
                            raise PointExecutionError(
                                "recovery journal already exists with status "
                                f"{prior.get('status')!r}; use a new immutable "
                                "probe_point_id after resolving/auditing it")

                        try:
                            backend_channels = (
                                set(spec.signal_channels) & {"api", "db"})
                            if len(backend_channels) > 1:
                                raise PointExecutionError(
                                    "one point cannot bind multiple backend "
                                    "observer channels")
                            observer = None
                            if backend_channels:
                                channel = next(iter(backend_channels))
                                if channel != "db":
                                    raise PointExecutionError(
                                        "no reviewed live API provider is "
                                        f"registered for {spec.site}/{spec.action_type}; "
                                        "use signal channel db")
                                observer = (
                                    backend_observer_factory(spec)
                                    if backend_observer_factory is not None else
                                    build_live_backend_observer(
                                        spec.site, spec.action_type,
                                        spec.environment_instance))
                            result = run_point_spec(
                                renv, spec, ks, base, commit=commit,
                                recovery_journal=journal_path,
                                backend_observer=observer)

                            # A self-recovering live probe is not allowed to
                            # leave the account changed merely because PARTIAL
                            # is a representable research label.  Attempt an
                            # independent cleanup and exclude the run instead.
                            if (result.effect_status == EFFECT_CHANGED and
                                    spec.safety_level == "self_recovering" and
                                    result.recovery_status != RECOVERY_RECOVERED):
                                raise PointExecutionError(
                                    "self-recovering point did not restore its "
                                    "baseline; excluding result and entering "
                                    "emergency cleanup")

                            pre = json.loads(journal_path.read_text(
                                encoding="utf-8"))["pre_signal"]
                            final = (result.evidence.get("final_signal")
                                     if result.effect_status == EFFECT_CHANGED
                                     else result.evidence.get("post_signal"))
                            if final != pre:
                                raise PointExecutionError(
                                    "terminal result does not match the WAL "
                                    "baseline; entering emergency cleanup")

                            _update_journal(
                                journal_path,
                                status=("NO_EFFECT" if
                                        result.effect_status == EFFECT_NO_EFFECT
                                        else "RECOVERED"),
                                final_signal=final,
                                recovery_status=result.recovery_status,
                                undo_actions=list(
                                    result.evidence.get("undo_actions") or []),
                                cleanup_verified=True,
                            )
                            transition = getattr(result, "_probe_transition", None)
                            if not isinstance(transition, ProbeTransition):
                                raise PointExecutionError(
                                    "point_runner.v2 completed without one captured "
                                    "transition body")
                            transition.validate()
                            if observer is not None:
                                captured_observer = getattr(
                                    result, "_backend_signal_observer", None)
                                observations = getattr(
                                    result, "_backend_signal_observations", None)
                                if captured_observer is not observer or not \
                                        isinstance(observations, dict):
                                    raise PointExecutionError(
                                        "point completed without its reviewed "
                                        "backend observer trace")
                                signal_bundle = materialize_signal_evidence(
                                    Path(data_root),
                                    probe_point_id=spec.probe_point_id,
                                    channel=observer.provider.channel,
                                    environment_instance=spec.environment_instance,
                                    collection_run_id=spec.probe_run_id,
                                    observer_version=OBSERVER_VERSION,
                                    endpoint_or_query_descriptor=(
                                        observer.endpoint_or_query_descriptor),
                                    provider_metadata=observer.provider_metadata,
                                    code_version=spec.code_version,
                                    collection_timestamp=result.timestamp,
                                    observations=observations,
                                    collected_live=observer.collected_live,
                                    is_fixture=observer.is_fixture,
                                    pii_review_status="REDACTED_AND_REVIEWED",
                                    redaction_applied=True,
                                )
                                patch = signal_bundle["point_evidence_patch"]
                                result.evidence["signal_channels"] = sorted(
                                    set(result.evidence.get("signal_channels") or []) |
                                    set(patch["signal_channels"]))
                                result.evidence.setdefault(
                                    "api_db_signal_summaries", {}).update(
                                        patch["api_db_signal_summaries"])
                                result.evidence.setdefault(
                                    "signal_evidence_assets", []).extend(
                                        patch["signal_evidence_assets"])
                            _check_review_gates(spec, result)
                        except Exception as exc:
                            cleanup_status = "NOT_REQUIRED_PRE_MUTATION"
                            cleanup_actions: list[str] = []
                            cleanup_error: str | None = None
                            if journal_path.exists():
                                wal = json.loads(journal_path.read_text(
                                    encoding="utf-8"))
                                if wal.get("status") == "RECOVERY_PENDING":
                                    try:
                                        final, cleanup_actions = _emergency_restore(
                                            renv, spec, base,
                                            dict(wal.get("pre_signal") or {}))
                                        restored = final == wal.get("pre_signal")
                                        cleanup_status = (
                                            "RECOVERED_AFTER_EXCEPTION" if restored
                                            else "RECOVERY_FAILED")
                                        _update_journal(
                                            journal_path, status=cleanup_status,
                                            final_signal=final,
                                            cleanup_actions=cleanup_actions,
                                            cleanup_verified=restored,
                                            original_error=(
                                                f"{type(exc).__name__}: {exc}"),
                                        )
                                        if not restored:
                                            abort_group = True
                                    except Exception as cleanup_exc:
                                        cleanup_status = "RECOVERY_FAILED"
                                        cleanup_error = (
                                            f"{type(cleanup_exc).__name__}: "
                                            f"{cleanup_exc}")
                                        abort_group = True
                                        _update_journal(
                                            journal_path, status=cleanup_status,
                                            cleanup_actions=cleanup_actions,
                                            cleanup_verified=False,
                                            cleanup_error=cleanup_error,
                                            original_error=(
                                                f"{type(exc).__name__}: {exc}"),
                                        )
                            failures.append({
                                "probe_point_id": spec.probe_point_id,
                                "state_id": spec.state_id,
                                "action_type": spec.action_type,
                                "reason": str(exc),
                                "exception_type": type(exc).__name__,
                                "cleanup_status": cleanup_status,
                                "cleanup_actions": cleanup_actions,
                                "cleanup_error": cleanup_error,
                                "recovery_journal": str(journal_path),
                            })
                            continue
                except Exception as exc:
                    # Includes lock acquisition and malformed pre-existing WAL
                    # failures, both of which occur before this spec mutates.
                    failures.append({
                        "probe_point_id": spec.probe_point_id,
                        "state_id": spec.state_id,
                        "action_type": spec.action_type,
                        "reason": str(exc),
                        "exception_type": type(exc).__name__,
                        "cleanup_status": "NOT_REQUIRED_PRE_MUTATION",
                        "recovery_journal": str(journal_path),
                    })
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
                transitions.append(transition)
                print(f"  [point] {spec.probe_point_id} {spec.action_type:22s} "
                      f"-> {result.effect_status}/{result.recovery_status} "
                      f"cost={result.undo_cost_steps} "
                      f"({time.time() - started:.1f}s)")
        finally:
            renv.close()
    return {"results": results, "transitions": transitions,
            "failures": failures}
