"""Task-independent mutation discovery and signal sufficiency fixtures."""
from __future__ import annotations

from dataclasses import replace

import pytest

from revact.data.mutation_miner import (
    InteractiveControl,
    MutationMiner,
    MutationMiningError,
    SignalSnapshot,
    mutation_detection_report,
    signal_sufficiency_audit,
)


class FixtureExecutor:
    safety_mode = "fixture"

    def __init__(self):
        self.state = SignalSnapshot(ui=("home",), api={"items": 0}, db={"items": 0})
        self.controls = (
            InteractiveControl("1", "add_item", "button", "Add"),
            InteractiveControl("2", "open_help", "button", "Help"),
            InteractiveControl("3", "api_only_mutation", "button", "Refresh"),
        )

    def reset_to_anchor(self):
        self.state = SignalSnapshot(ui=("home",), api={"items": 0}, db={"items": 0})

    def enumerate_interactive_controls(self):
        return self.controls

    def capture_signals(self):
        return self.state

    def execute_control(self, control):
        if control.bid == "1":
            self.state = SignalSnapshot(
                ui=("home", "item"), api={"items": 1}, db={"items": 1})
        elif control.bid == "3":
            self.state = replace(self.state, api={"items": 1}, db={"items": 1})
        return {"fixture": True, "bid": control.bid}


def test_mutation_miner_enumerates_all_controls_and_reports_wilson_stats():
    executor = FixtureExecutor()
    miner = MutationMiner(executor, ranker=lambda controls: tuple(reversed(controls)))
    trials = miner.mine()
    assert [trial.bid for trial in trials] == ["3", "2", "1"]
    assert {trial.bid for trial in trials if trial.mutation_candidate} == {"1", "3"}
    assert all(trial.reset_restored for trial in trials)
    report = mutation_detection_report(trials, {"1", "3"})
    assert report["enumerated_n"] == 3
    assert report["precision"] == report["recall"] == 1.0
    assert report["precision_wilson_95"][0] < 1.0


def test_ranker_may_reorder_but_cannot_filter_controls():
    with pytest.raises(MutationMiningError, match="may rank but may not filter"):
        MutationMiner(
            FixtureExecutor(), ranker=lambda controls: controls[:1]).mine()


def test_mutation_miner_refuses_live_execution_mode():
    executor = FixtureExecutor()
    executor.safety_mode = "live"
    with pytest.raises(MutationMiningError, match="refuses"):
        MutationMiner(executor)


def test_signal_sufficiency_exposes_ui_backend_agreement_and_blind_spot():
    before = SignalSnapshot(ui="same", api={"x": 0}, db={"x": 0})
    hidden = SignalSnapshot(ui="same", api={"x": 1}, db={"x": 1})
    audit = signal_sufficiency_audit(before, hidden)
    assert audit["ui_backend_agree"] is False
    assert audit["ui_vs_api_agree"] is False
    assert audit["ui_vs_db_agree"] is False
    assert "ui_missed_backend_change" in audit["blind_spots"]

    visible = SignalSnapshot(ui="changed", api={"x": 1}, db={"x": 1})
    audit = signal_sufficiency_audit(before, visible)
    assert audit["ui_backend_agree"] is True
    assert audit["blind_spots"] == []


def test_signal_sufficiency_marks_async_and_external_blind_spots():
    before = SignalSnapshot(ui="same", api=0, db=0, external=0)
    after = SignalSnapshot(
        ui="same", api=0, db=0, external=1, async_pending=True)
    audit = signal_sufficiency_audit(before, after)
    assert audit["conclusive"] is False
    assert "external_side_effect_not_reflected_in_api_or_db" in audit["blind_spots"]
    assert "async_state_unsettled" in audit["blind_spots"]
