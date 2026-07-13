"""Constructive-positive and budget-relative-negative solver semantics."""
from __future__ import annotations

import pytest

from revact.grounding.schema import (
    RECOVERY_NOT_WITHIN_BUDGET,
    RECOVERY_RECOVERED,
)
from revact.grounding.solvers import (
    SOLVER_BFS,
    SOLVER_DETERMINISTIC,
    SOLVER_LLM_ATTACKER,
    AffordanceBFSSolver,
    SolverContext,
    SolverProtocolError,
    SolverTrace,
    attach_solver_union,
    combine_solver_traces,
)


CONTEXT = SolverContext(
    probe_point_id="point-1", initial_signal={"item": 0},
    post_action_signal={"item": 1})


def _failure(kind, name, *, seeds=(), attack_attempts=0, exhausted=False):
    return SolverTrace(
        solver_name=name,
        solver_kind=kind,
        solver_version="fixture-1",
        budget_k=3,
        success=False,
        explored_actions=(f"{name}:attempt",),
        explored_states=("mutated",),
        undo_actions=(),
        final_signal={"item": 1},
        residual_diff={"item": [0, 1]},
        termination_reason="budget_exhausted" if exhausted else "no_channel_found",
        budget_exhausted=exhausted,
        attack_attempts=attack_attempts,
        seeds=tuple(seeds),
        evidence={"fixture": True},
    )


def test_second_undo_channel_flips_fixed_controller_false_negative():
    # The fixed controller only tried a stale shortcut and failed.  BFS opens a
    # menu and discovers the actual Remove action at depth two.
    fixed_failure = _failure(
        SOLVER_DETERMINISTIC, "shopping.fixed_controller")
    transitions = {
        ("mutated", "stale_shortcut"): "mutated",
        ("mutated", "open_menu"): "menu_open",
        ("menu_open", "remove_item"): "initial",
        ("menu_open", "close_menu"): "mutated",
    }
    actions = {
        "mutated": ("stale_shortcut", "open_menu"),
        "menu_open": ("close_menu", "remove_item"),
        "initial": (),
    }
    bfs = AffordanceBFSSolver(
        initial_state="mutated",
        enumerate_actions=lambda state: actions[state],
        transition=lambda state, action: transitions[(state, action)],
        state_key=str,
        is_recovered=lambda state: state == "initial",
        state_signal=lambda state: {"item": 0 if state == "initial" else 1},
        residual_diff=lambda state: {} if state == "initial" else {"item": [0, 1]},
        max_depth=2,
        solver_name="shopping.affordance_bfs",
        solver_version="fixture-1",
    )
    bfs_trace = bfs.solve(CONTEXT, budget_k=3)
    result = combine_solver_traces(
        [fixed_failure, bfs_trace], budget_k=3)
    assert fixed_failure.success is False
    assert bfs_trace.success is True
    assert result.recovery_status == RECOVERY_RECOVERED
    assert result.successful_solver == "shopping.affordance_bfs"
    assert result.undo_actions == ("open_menu", "remove_item")
    assert result.undo_cost_steps == 2


def test_all_solver_failures_only_mean_not_recovered_within_budget():
    traces = [
        _failure(SOLVER_DETERMINISTIC, "deterministic"),
        _failure(SOLVER_BFS, "bfs", exhausted=True),
        _failure(
            SOLVER_LLM_ATTACKER, "llm-attacker", seeds=(7, 11),
            attack_attempts=2, exhausted=True),
    ]
    result = combine_solver_traces(traces, budget_k=3)
    assert result.recovery_status == RECOVERY_NOT_WITHIN_BUDGET
    assert result.display_label == "NOT_RECOVERED_WITHIN_BUDGET"
    assert "IRREVERSIBLE" not in str(result.to_evidence())
    assert result.solver_set == ("deterministic", "bfs", "llm-attacker")
    assert result.attack_attempts == 2
    assert result.budget_exhausted is True
    assert set(result.explored_actions) == set(result.solver_set)


def test_solver_union_evidence_attaches_to_formal_point():
    from revact.grounding.schema import GroundingPoint
    base = GroundingPoint(
        probe_point_id="point-1", probe_run_id="run-1", probe_name="fixture",
        state_id="state-1", candidate_id="candidate-1",
        action_instance_id="action-1", action_type="delete",
        raw_action="click('1')", canonical_action="click:button:delete",
        site="fixture", environment_family="fixture",
        environment_instance="fixture-1", environment_origin="fixture",
        task_id="task-1",
        trajectory_id="trajectory-1", run_id="collection-1",
        url="https://fixture.invalid/item/1", account="fixture-user",
        privilege="user", budget_k=3,
        solver_set=["initial"], controller_version="test",
        pre_observation_hash="pre", pre_signal={"item": 0},
        post_observation_hash="post", post_signal={"item": 1},
        effect_status="CHANGED", recovery_status="RECOVERED",
        undo_cost_steps=1, undo_actions=["initial"],
        timestamp="2026-07-13T00:00:00+00:00", code_version="deadbeef",
        evidence={"fixture": True,
                  "candidate_snapshot_hash": "candidate-snapshot-hash"})
    result = combine_solver_traces([
        _failure(SOLVER_DETERMINISTIC, "deterministic"),
        _failure(SOLVER_BFS, "bfs", exhausted=True),
        _failure(SOLVER_LLM_ATTACKER, "llm", seeds=(7, 11),
                 attack_attempts=2, exhausted=True),
    ], budget_k=3)
    point = attach_solver_union(base, result)
    point.validate(formal=True)
    assert point.recovery_status == RECOVERY_NOT_WITHIN_BUDGET
    assert point.evidence["solver_union"]["attack_attempts"] == 2


def test_formal_negative_requires_three_routes_and_multi_seed_attacker():
    with pytest.raises(SolverProtocolError, match="missing solver routes"):
        combine_solver_traces([
            _failure(SOLVER_DETERMINISTIC, "deterministic"),
            _failure(SOLVER_BFS, "bfs"),
        ], budget_k=3)

    with pytest.raises(SolverProtocolError, match=">=2 seeds"):
        combine_solver_traces([
            _failure(SOLVER_DETERMINISTIC, "deterministic"),
            _failure(SOLVER_BFS, "bfs"),
            _failure(
                SOLVER_LLM_ATTACKER, "llm", seeds=(7,), attack_attempts=1),
        ], budget_k=3)


def test_bfs_refuses_live_adapter_and_path_longer_than_budget():
    with pytest.raises(SolverProtocolError, match="refuses live"):
        AffordanceBFSSolver(
            initial_state=0,
            enumerate_actions=lambda state: (),
            transition=lambda state, action: state,
            state_key=str,
            is_recovered=lambda state: False,
            safety_mode="live",
        )

    trace = SolverTrace(
        solver_name="bad", solver_kind=SOLVER_BFS, solver_version="1",
        budget_k=2, success=True, explored_actions=("a", "b", "c"),
        explored_states=("0", "1", "2"), undo_actions=("a", "b", "c"),
        final_signal={}, residual_diff={}, termination_reason="recovered",
        budget_exhausted=False)
    with pytest.raises(SolverProtocolError, match="exceeds budget"):
        trace.validate()
