#!/usr/bin/env python3
"""Run the task-independent mutation miner on a 240-control offline fixture."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from revact.data.mutation_miner import (  # noqa: E402
    InteractiveControl,
    MutationMiner,
    SignalSnapshot,
    mutation_detection_report,
)


class LargeFixtureExecutor:
    safety_mode = "fixture"

    def __init__(self, n: int = 240):
        verbs = ("create_issue", "edit_page", "delete_branch", "open_help",
                 "navigate_map", "read_wikipedia")
        self.controls = tuple(
            InteractiveControl(str(i), f"{verbs[i % len(verbs)]}:{i}")
            for i in range(n))
        self.mutated_bids = {str(i) for i in range(n) if i % 5 == 0}
        self.state = SignalSnapshot()
        self.reset_to_anchor()

    def reset_to_anchor(self):
        self.state = SignalSnapshot(
            ui={"revision": 0}, api={"revision": 0}, db={"revision": 0},
            external={"events": 0})

    def enumerate_interactive_controls(self):
        return self.controls

    def capture_signals(self):
        return self.state

    def execute_control(self, control):
        if control.bid in self.mutated_bids:
            self.state = replace(
                self.state,
                ui={"revision": int(control.bid) + 1},
                api={"revision": int(control.bid) + 1},
                db={"revision": int(control.bid) + 1})
        return {"fixture": True, "canonical_action": control.canonical_action}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "docs" / "audit" /
        "mutation-miner-fixture-report.json")
    args = parser.parse_args()
    executor = LargeFixtureExecutor()
    trials = MutationMiner(executor).mine()
    report = mutation_detection_report(trials, executor.mutated_bids)
    report.update({
        "protocol": "mutation-miner.fixture.v1",
        "safety_mode": executor.safety_mode,
        "environment_families_simulated": [
            "gitlab(create/edit/delete)", "map", "wikipedia"],
        "detector_action_class_keywords": [],
        "reset_restored_n": sum(trial.reset_restored for trial in trials),
        "claim_scope": (
            "Offline fixture validation only; this is not a live WebArena/GitLab "
            "precision or recall result."),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
