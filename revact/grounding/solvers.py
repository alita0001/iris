"""Offline-safe solver-union evidence for operational recovery probes.

This module makes the asymmetry explicit: one successful solver constructively
proves recovery for the recorded signals, while unanimous failure only yields
``NOT_RECOVERED_WITHIN_BUDGET``.  It never emits mathematical
``IRREVERSIBLE``.  Solver implementations return full traces so fixed
controller false negatives, BFS coverage, and multi-seed attacker attempts can
be audited independently.
"""
from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Mapping, Protocol, Sequence

from .schema import RECOVERY_NOT_WITHIN_BUDGET, RECOVERY_RECOVERED


SOLVER_DETERMINISTIC = "site_specific_deterministic"
SOLVER_BFS = "affordance_bfs"
SOLVER_LLM_ATTACKER = "llm_undo_attacker"
FORMAL_NEGATIVE_SOLVER_KINDS = frozenset({
    SOLVER_DETERMINISTIC, SOLVER_BFS, SOLVER_LLM_ATTACKER,
})


class SolverProtocolError(ValueError):
    """Raised when solver evidence is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class SolverContext:
    probe_point_id: str
    initial_signal: Any
    post_action_signal: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Exact checkpoint every independent solver must restore before searching.
    # This is deliberately distinct from a URL: navigation can reproduce a page
    # while leaving the mutated backend state from a previous solver branch.
    post_action_state_hash: str = ""


@dataclass(frozen=True)
class SolverTrace:
    solver_name: str
    solver_kind: str
    solver_version: str
    budget_k: int
    success: bool
    explored_actions: tuple[str, ...]
    explored_states: tuple[str, ...]
    undo_actions: tuple[str, ...]
    final_signal: Any
    residual_diff: Any
    termination_reason: str
    budget_exhausted: bool
    attack_attempts: int = 0
    seeds: tuple[int, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)
    undo_semantic_actions: tuple[str, ...] = ()
    undo_observation_hashes: tuple[str, ...] = ()
    # Formal-negative start-point provenance.  These remain optional for a
    # constructive positive / legacy fixture, but every failed trace in a formal
    # negative union must populate and agree on all four fields.
    start_state_hash: str = ""
    start_signal: Any = None
    reset_method: str = ""
    reset_verified: bool = False

    def validate(self) -> None:
        errors: list[str] = []
        if not self.solver_name:
            errors.append("missing solver_name")
        if self.solver_kind not in FORMAL_NEGATIVE_SOLVER_KINDS:
            errors.append(f"unsupported solver_kind {self.solver_kind!r}")
        if not self.solver_version:
            errors.append("missing solver_version")
        if not isinstance(self.budget_k, int) or self.budget_k <= 0:
            errors.append("budget_k must be a positive integer")
        if not self.termination_reason:
            errors.append("missing termination_reason")
        if self.success:
            if not self.undo_actions:
                errors.append("successful recovery requires a constructive undo path")
            if len(self.undo_actions) > self.budget_k:
                errors.append("successful undo path exceeds budget_k")
            if len(self.undo_semantic_actions) != len(self.undo_actions):
                errors.append("successful trace requires semantic undo actions 1:1")
            if len(self.undo_observation_hashes) != len(self.undo_actions):
                errors.append("successful trace requires undo observation hashes 1:1")
            if self.budget_exhausted:
                errors.append("successful trace cannot be budget_exhausted")
        if self.attack_attempts < 0:
            errors.append("attack_attempts cannot be negative")
        if not isinstance(self.reset_verified, bool):
            errors.append("reset_verified must be boolean")
        if self.solver_kind == SOLVER_LLM_ATTACKER:
            if self.attack_attempts <= 0:
                errors.append("LLM attacker must record attack_attempts")
            if not self.seeds:
                errors.append("LLM attacker must record seeds")
        if errors:
            raise SolverProtocolError(
                f"{self.solver_name or '<unnamed solver>'}: " + "; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UndoSolver(Protocol):
    solver_name: str
    solver_kind: str
    solver_version: str

    def solve(self, context: SolverContext, budget_k: int) -> SolverTrace: ...


@dataclass(frozen=True)
class SolverUnionResult:
    recovery_status: str
    budget_k: int
    solver_set: tuple[str, ...]
    successful_solver: str | None
    undo_actions: tuple[str, ...]
    undo_semantic_actions: tuple[str, ...]
    undo_observation_hashes: tuple[str, ...]
    undo_cost_steps: int | None
    final_signal: Any
    residual_diff: Any
    explored_actions: Mapping[str, tuple[str, ...]]
    explored_states: Mapping[str, tuple[str, ...]]
    termination_reasons: Mapping[str, str]
    attack_attempts: int
    budget_exhausted: bool
    traces: tuple[SolverTrace, ...]
    # Populated for formal negatives after all traces prove an independently
    # reset, common post-action starting point.  Kept optional so constructive
    # positive evidence produced before this protocol remains compatible.
    post_action_state_hash: str = ""
    post_signal: Any = None

    @property
    def display_label(self) -> str:
        # Deliberately identical to canonical recovery_status.  In particular,
        # a negative result is never rendered as IRREVERSIBLE.
        return self.recovery_status

    def to_evidence(self) -> dict[str, Any]:
        evidence = {
            # Requiring independent, verified solver starts is a breaking change
            # only for negative evidence.  Positives retain the v1 wire protocol.
            "protocol": (
                "solver_union.v2"
                if self.recovery_status == RECOVERY_NOT_WITHIN_BUDGET
                else "solver_union.v1"),
            "recovery_status": self.recovery_status,
            "budget_k": self.budget_k,
            "solver_set": list(self.solver_set),
            "successful_solver": self.successful_solver,
            "undo_actions": list(self.undo_actions),
            "undo_semantic_actions": list(self.undo_semantic_actions),
            "undo_observation_hashes": list(self.undo_observation_hashes),
            "undo_cost_steps": self.undo_cost_steps,
            "explored_actions": {
                name: list(actions) for name, actions in self.explored_actions.items()},
            "explored_states": {
                name: list(states) for name, states in self.explored_states.items()},
            "termination_reasons": dict(self.termination_reasons),
            "attack_attempts": self.attack_attempts,
            "budget_exhausted": self.budget_exhausted,
            "traces": [trace.to_dict() for trace in self.traces],
        }
        if self.post_action_state_hash:
            evidence["post_action_state_hash"] = self.post_action_state_hash
        if self.post_signal is not None:
            evidence["post_signal"] = self.post_signal
        return evidence


def _canonical_signal(value: Any, *, where: str) -> str:
    """Stable equality form for serialized solver signals.

    Formal evidence is JSON, so accepting a Python-only/non-serializable signal
    here would make the in-memory gate stronger or weaker than import-time
    validation.  Fail before emitting such evidence.
    """
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise SolverProtocolError(
            f"{where}: signal must be JSON-serializable") from exc


def _normalized_reset_method(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _formal_negative_start(
        traces: Sequence[SolverTrace]) -> tuple[str, Any]:
    """Validate independent resets to one exact post-action checkpoint.

    A URL reload is not a reset: it can leave backend mutations made by an
    earlier solver attempt in place.  Each route must therefore record a
    verified reset and independently observe the same state hash and signal.
    """
    hashes: list[str] = []
    signals: list[str] = []
    for trace in traces:
        name = trace.solver_name or "<unnamed solver>"
        if not str(trace.start_state_hash or "").strip():
            raise SolverProtocolError(
                f"{name}: formal negative requires start_state_hash")
        if trace.start_signal is None:
            raise SolverProtocolError(
                f"{name}: formal negative requires start_signal")
        method = _normalized_reset_method(trace.reset_method)
        if not method:
            raise SolverProtocolError(
                f"{name}: formal negative requires reset_method")
        branch_reset = _normalized_reset_method(
            (trace.evidence or {}).get("branch_reset")
            if isinstance(trace.evidence, Mapping) else "")
        if method == "url_reanchor_only" or branch_reset == "url_reanchor_only":
            raise SolverProtocolError(
                f"{name}: url_reanchor_only is not an independent state reset")
        if trace.reset_verified is not True:
            raise SolverProtocolError(
                f"{name}: formal negative requires reset_verified=true")
        hashes.append(str(trace.start_state_hash))
        signals.append(_canonical_signal(
            trace.start_signal, where=f"{name}.start_signal"))

    if len(set(hashes)) != 1:
        raise SolverProtocolError(
            "formal negative traces do not share one post_action_state_hash")
    if len(set(signals)) != 1:
        raise SolverProtocolError(
            "formal negative traces do not share one post_signal")
    return hashes[0], traces[0].start_signal


def combine_solver_traces(
    traces: Sequence[SolverTrace],
    *,
    budget_k: int,
    formal_negative: bool = True,
) -> SolverUnionResult:
    """Combine independent traces under the constructive-positive rule."""
    if not traces:
        raise SolverProtocolError("solver union requires at least one trace")
    if not isinstance(budget_k, int) or budget_k <= 0:
        raise SolverProtocolError("union budget_k must be a positive integer")
    materialized = tuple(traces)
    for trace in materialized:
        trace.validate()
        if trace.budget_k != budget_k:
            raise SolverProtocolError(
                f"{trace.solver_name}: trace budget {trace.budget_k} != union budget {budget_k}")
    names = [trace.solver_name for trace in materialized]
    if len(names) != len(set(names)):
        raise SolverProtocolError("solver names must be unique within a point")

    successes = [trace for trace in materialized if trace.success]
    if successes:
        # The shortest recorded construction defines point-level undo cost.
        winner = min(successes, key=lambda trace: (len(trace.undo_actions), trace.solver_name))
        recovery_status = RECOVERY_RECOVERED
        successful_solver = winner.solver_name
        undo_actions = winner.undo_actions
        undo_semantic_actions = winner.undo_semantic_actions
        undo_observation_hashes = winner.undo_observation_hashes
        undo_cost_steps: int | None = len(winner.undo_actions)
        final_signal = winner.final_signal
        residual_diff = winner.residual_diff
        budget_exhausted = False
        post_action_state_hash = winner.start_state_hash
        post_signal = winner.start_signal
    else:
        kinds = {trace.solver_kind for trace in materialized}
        if formal_negative:
            missing = sorted(FORMAL_NEGATIVE_SOLVER_KINDS - kinds)
            if missing:
                raise SolverProtocolError(
                    f"formal negative is missing solver routes: {missing}")
            llm_traces = [
                trace for trace in materialized
                if trace.solver_kind == SOLVER_LLM_ATTACKER]
            llm_seeds = {seed for trace in llm_traces for seed in trace.seeds}
            if len(llm_seeds) < 2:
                raise SolverProtocolError(
                    "formal negative requires LLM undo attacker evidence from >=2 seeds")
            post_action_state_hash, post_signal = _formal_negative_start(materialized)
        else:
            post_action_state_hash, post_signal = "", None
        recovery_status = RECOVERY_NOT_WITHIN_BUDGET
        successful_solver = None
        undo_actions = ()
        undo_semantic_actions = ()
        undo_observation_hashes = ()
        undo_cost_steps = None
        # Keep the last measured state only as evidence; it is not a proof that
        # no alternative path exists.
        final_signal = materialized[-1].final_signal
        residual_diff = materialized[-1].residual_diff
        budget_exhausted = any(trace.budget_exhausted for trace in materialized)

    return SolverUnionResult(
        recovery_status=recovery_status,
        budget_k=budget_k,
        solver_set=tuple(trace.solver_name for trace in materialized),
        successful_solver=successful_solver,
        undo_actions=undo_actions,
        undo_semantic_actions=undo_semantic_actions,
        undo_observation_hashes=undo_observation_hashes,
        undo_cost_steps=undo_cost_steps,
        final_signal=final_signal,
        residual_diff=residual_diff,
        explored_actions={
            trace.solver_name: trace.explored_actions for trace in materialized},
        explored_states={
            trace.solver_name: trace.explored_states for trace in materialized},
        termination_reasons={
            trace.solver_name: trace.termination_reason for trace in materialized},
        attack_attempts=sum(trace.attack_attempts for trace in materialized),
        budget_exhausted=budget_exhausted,
        traces=materialized,
        post_action_state_hash=post_action_state_hash,
        post_signal=post_signal,
    )


def run_solver_union(
    solvers: Sequence[UndoSolver],
    context: SolverContext,
    *,
    budget_k: int,
    formal_negative: bool = True,
) -> SolverUnionResult:
    """Run injected solvers and validate that trace provenance matches them."""
    traces: list[SolverTrace] = []
    for solver in solvers:
        trace = solver.solve(context, budget_k)
        declared = (solver.solver_name, solver.solver_kind, solver.solver_version)
        recorded = (trace.solver_name, trace.solver_kind, trace.solver_version)
        if declared != recorded:
            raise SolverProtocolError(
                f"solver declaration {declared!r} != returned trace {recorded!r}")
        traces.append(trace)
    result = combine_solver_traces(
        traces, budget_k=budget_k, formal_negative=formal_negative)
    if result.recovery_status == RECOVERY_NOT_WITHIN_BUDGET and formal_negative:
        if not str(context.post_action_state_hash or "").strip():
            raise SolverProtocolError(
                "formal negative SolverContext requires post_action_state_hash")
        if result.post_action_state_hash != context.post_action_state_hash:
            raise SolverProtocolError(
                "solver-union post_action_state_hash contradicts SolverContext")
        if _canonical_signal(result.post_signal, where="solver_union.post_signal") != \
                _canonical_signal(context.post_action_signal,
                                  where="SolverContext.post_action_signal"):
            raise SolverProtocolError(
                "solver-union post_signal contradicts SolverContext")
    return result


def attach_solver_union(point, result: SolverUnionResult):
    """Return a new GroundingPoint carrying the complete solver-union evidence.

    Kept generic at the signature to avoid a schema↔solver import cycle; the
    returned dataclass is validated by the caller/persistence gate.
    """
    evidence = dict(point.evidence or {})
    evidence["solver_union"] = result.to_evidence()
    return replace(
        point,
        recovery_status=result.recovery_status,
        budget_k=result.budget_k,
        solver_set=list(result.solver_set),
        undo_actions=list(result.undo_actions),
        undo_semantic_actions=list(result.undo_semantic_actions),
        undo_observation_hashes=list(result.undo_observation_hashes),
        undo_cost_steps=result.undo_cost_steps,
        final_signal=result.final_signal,
        residual_diff=result.residual_diff,
        budget_exhausted=result.budget_exhausted,
        evidence=evidence,
    )


class AffordanceBFSSolver:
    """Generic depth-limited BFS over an injected fixture/dry-run state graph.

    The adapter callbacks may represent a browser fixture, an API fixture, or a
    pure state machine.  This implementation refuses ``safety_mode='live'``;
    approved live orchestration must wrap it behind the repository's external
    destructive gates.
    """

    solver_kind = SOLVER_BFS

    def __init__(
        self,
        *,
        initial_state: Any,
        enumerate_actions: Callable[[Any], Sequence[Any]],
        transition: Callable[[Any, Any], Any],
        state_key: Callable[[Any], str],
        is_recovered: Callable[[Any], bool],
        render_action: Callable[[Any], str] = str,
        state_signal: Callable[[Any], Any] = lambda state: state,
        residual_diff: Callable[[Any], Any] = lambda state: {},
        max_depth: int = 3,
        safety_mode: str = "fixture",
        solver_name: str = "affordance_bfs.depth3",
        solver_version: str = "1",
    ) -> None:
        if safety_mode not in {"fixture", "dry_run"}:
            raise SolverProtocolError(
                "AffordanceBFSSolver refuses live adapters without an external approval gate")
        if max_depth not in {2, 3}:
            raise SolverProtocolError("formal affordance BFS max_depth must be 2 or 3")
        self.initial_state = initial_state
        self.enumerate_actions = enumerate_actions
        self.transition = transition
        self.state_key = state_key
        self.is_recovered = is_recovered
        self.render_action = render_action
        self.state_signal = state_signal
        self.residual_diff_fn = residual_diff
        self.max_depth = max_depth
        self.safety_mode = safety_mode
        self.solver_name = solver_name
        self.solver_version = solver_version

    def solve(self, context: SolverContext, budget_k: int) -> SolverTrace:
        if budget_k < self.max_depth:
            raise SolverProtocolError(
                f"budget_k={budget_k} is smaller than BFS max_depth={self.max_depth}")
        initial_key = self.state_key(self.initial_state)
        start_signal = self.state_signal(self.initial_state)
        start_hash = str(context.post_action_state_hash or "")
        # Fixture/dry-run BFS owns an independent in-memory initial state.  It is
        # verified only when both its state key and signal match the checkpoint
        # declared by the orchestration context.
        reset_method = ("fixture_state_clone" if self.safety_mode == "fixture"
                        else "dry_run_state_clone")
        reset_verified = bool(
            start_hash and str(initial_key) == start_hash
            and _canonical_signal(start_signal, where="bfs.start_signal") ==
            _canonical_signal(context.post_action_signal,
                              where="SolverContext.post_action_signal"))
        queue: deque[tuple[Any, tuple[str, ...], tuple[str, ...]]] = deque(
            [(self.initial_state, (), ())])
        visited = {initial_key}
        explored_states = [initial_key]
        explored_actions: list[str] = []
        last_state = self.initial_state
        reached_depth_limit = False

        while queue:
            state, path, path_hashes = queue.popleft()
            last_state = state
            if path and self.is_recovered(state):
                return SolverTrace(
                    solver_name=self.solver_name,
                    solver_kind=self.solver_kind,
                    solver_version=self.solver_version,
                    budget_k=budget_k,
                    success=True,
                    explored_actions=tuple(explored_actions),
                    explored_states=tuple(explored_states),
                    undo_actions=path,
                    undo_semantic_actions=path,
                    undo_observation_hashes=path_hashes,
                    final_signal=self.state_signal(state),
                    residual_diff=self.residual_diff_fn(state),
                    termination_reason="recovered",
                    budget_exhausted=False,
                    evidence={"probe_point_id": context.probe_point_id,
                              "max_depth": self.max_depth},
                    start_state_hash=start_hash,
                    start_signal=start_signal,
                    reset_method=reset_method,
                    reset_verified=reset_verified,
                )
            if len(path) >= self.max_depth:
                reached_depth_limit = True
                continue
            for action in self.enumerate_actions(state):
                rendered = self.render_action(action)
                explored_actions.append(rendered)
                next_state = self.transition(state, action)
                key = self.state_key(next_state)
                if key in visited:
                    continue
                visited.add(key)
                explored_states.append(key)
                observation_hash = hashlib.sha256(
                    str(key).encode("utf-8")).hexdigest()
                queue.append((next_state, path + (rendered,),
                              path_hashes + (observation_hash,)))

        return SolverTrace(
            solver_name=self.solver_name,
            solver_kind=self.solver_kind,
            solver_version=self.solver_version,
            budget_k=budget_k,
            success=False,
            explored_actions=tuple(explored_actions),
            explored_states=tuple(explored_states),
            undo_actions=(),
            final_signal=self.state_signal(last_state),
            residual_diff=self.residual_diff_fn(last_state),
            termination_reason=(
                "depth_limit_exhausted" if reached_depth_limit else "state_space_exhausted"),
            budget_exhausted=reached_depth_limit,
            evidence={"probe_point_id": context.probe_point_id,
                      "max_depth": self.max_depth},
            start_state_hash=start_hash,
            start_signal=start_signal,
            reset_method=reset_method,
            reset_verified=reset_verified,
        )
