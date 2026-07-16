"""Task-independent mutation discovery and signal sufficiency fixtures."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from revact import cli
from revact.data.candidates import snapshot_sha256
from revact.data.mutation_miner import (
    MUTATION_REPORT_BODY_NAME,
    MUTATION_REPORT_MANIFEST_NAME,
    MUTATION_SAMPLE_SCHEMA_VERSION,
    InteractiveControl,
    MutationCensusSample,
    MutationMiner,
    MutationMiningError,
    SignalSnapshot,
    assert_mutation_census_integrity,
    audit_live_mutation_census,
    build_mutation_census_preflight,
    build_live_mutation_census,
    make_mutation_sample_id,
    mutation_control_set_sha256,
    mutation_detection_report,
    save_live_mutation_census,
    save_mutation_census_preflight,
    signal_sufficiency_audit,
    validate_live_mutation_census,
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


def test_preflight_deduplicates_physical_aliases_and_never_claims_evidence(
        tmp_path):
    state_bank = tmp_path / "raw" / "state_bank"
    state_bank.mkdir(parents=True)
    snapshot_a = "RootWebArea 'A'\n" + "".join(
        f"  [{index}] link 'Read item {index}'\n" for index in range(120))
    snapshot_b = "RootWebArea 'B'\n" + "".join(
        f"  [{index}] button '{'Place Order' if index == 0 else f'Control {index}'}'\n"
        for index in range(100))
    snapshot_b += "  [999] button 'Disabled', disabled=True\n"
    rows = [
        {
            "state_id": "state-a", "task_id": "webarena.1", "seed": 0,
            "trajectory_id": "trajectory-a", "run_id": "run-a",
            "url": "http://shopping:7770/a", "replay_prefix": [],
            "axtree_snapshot": snapshot_a, "axtree_complete": True,
            "capture_protocol": "fixture-full-axtree.v1",
        },
        # A logical alias of the exact same physical snapshot cannot pad 120
        # controls into 240.
        {
            "state_id": "state-a-alias", "task_id": "webarena.1", "seed": 0,
            "trajectory_id": "trajectory-a", "run_id": "run-a",
            "url": "http://shopping:7770/a", "replay_prefix": [],
            "axtree_snapshot": snapshot_a, "axtree_complete": True,
            "capture_protocol": "fixture-full-axtree.v1",
        },
        {
            "state_id": "state-b", "task_id": "webarena.2", "seed": 0,
            "trajectory_id": "trajectory-b", "run_id": "run-b",
            "url": "http://shopping:7770/b", "replay_prefix": [],
            "axtree_snapshot": snapshot_b, "axtree_complete": True,
            "capture_protocol": "fixture-full-axtree.v1",
        },
    ]
    (state_bank / "states.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))

    report = build_mutation_census_preflight(
        tmp_path, minimum_controls=200, code_version="worktree:test")
    assert report["unique_physical_snapshot_n"] == 2
    assert report["enumerable_control_n"] == 220
    assert report["selected_snapshot_n"] == 2
    assert report["selected_unique_page_n"] == 2
    assert report["selected_duplicate_canonical_page_n"] == 0
    assert report["selected_enumerated_control_n"] == 220
    assert report["selection_reaches_minimum"] is True
    assert report["executed_live_n"] == 0
    assert report["independent_reference_n"] == 0
    assert report["counts_as_live_census"] is False
    assert report["gate_satisfied"] is False
    aliases = next(item for item in report["selected_snapshots"]
                   if item["n_controls"] == 120)
    assert aliases["state_aliases"] == ["state-a", "state-a-alias"]
    assert report["selected_review_priority_counts"] == {
        "elevated_surface_review": 1,
        "unclassified_review": 219,
    }

    output = tmp_path / "audit" / "preflight.json"
    assert save_mutation_census_preflight(report, output) == "created"
    assert save_mutation_census_preflight(report, output) == "already_identical"
    assert cli.main([
        "plan-mutation-census", "--data-root", str(tmp_path),
        "--minimum-controls", "200", "--code-version", "worktree:test",
    ]) == 0


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _live_census_fixture(n: int = 4, *, include_disabled: bool = False):
    snapshot = "RootWebArea 'Census'\n" + "".join(
        f"  [{index}] button 'Control {index}'\n" for index in range(n))
    if include_disabled:
        snapshot += "  [999] button 'Disabled mutation', disabled=True\n"
    snapshot_hash = snapshot_sha256(snapshot)
    controls = [(str(index), f"click('{index}')") for index in range(n)]
    control_set_hash = mutation_control_set_sha256(controls)
    rows = []
    # Repeating TP, FN, FP, TN proves both rates are non-vacuous.
    outcomes = ((True, True), (False, True), (True, False), (False, False))
    for index, (detected, reference) in enumerate(
            outcomes[index % len(outcomes)] for index in range(n)):
        control_id = f"control-{index}"
        bid, action = controls[index]
        sample_id = make_mutation_sample_id(
            collection_run_id="live-run-1",
            environment_instance="fixture-instance-for-schema-test",
            state_id="state-1", snapshot_hash=snapshot_hash,
            control_id=control_id, bid=bid, canonical_action=action)
        rows.append(MutationCensusSample(
            schema_version=MUTATION_SAMPLE_SCHEMA_VERSION,
            mutation_sample_id=sample_id,
            collection_run_id="live-run-1",
            state_id="state-1", control_id=control_id, bid=bid,
            canonical_action=action, role="button", name=f"Control {index}",
            environment_family="schema-test",
            environment_instance="fixture-instance-for-schema-test",
            snapshot_hash=snapshot_hash, snapshot_control_count=n,
            control_set_sha256=control_set_hash, legal_at_snapshot=True,
            executed_live=True, is_fixture=False,
            safety_class="non_destructive", review_status="approved",
            destructive_commit_authorized=False, destructive_env_gate=False,
            detector_channels=("ui",),
            changed_channels=("ui",) if detected else (),
            mutation_candidate=detected, reference_mutated=reference,
            reference_source="independent_db_diff",
            reference_evidence_sha256=_hash(f"reference-{index}"),
            execution_evidence_sha256=_hash(f"execution-{index}"),
            pre_signal_sha256=_hash("anchor"),
            post_signal_sha256=_hash(
                f"post-{index}" if detected else "anchor"),
            reset_signal_sha256=_hash("anchor"), reset_restored=True,
            async_pending=False))
    return snapshot, rows


def _save_state_snapshot(root, snapshot: str) -> None:
    path = root / "raw" / "state_bank" / "census.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "state_id": "state-1", "axtree_snapshot": snapshot,
        "axtree_complete": True,
        "capture_protocol": "fixture-full-axtree.v1",
    }) + "\n")


def test_live_census_recomputes_metrics_and_rejects_hand_filled_drift():
    _snapshot, samples = _live_census_fixture()
    report = build_live_mutation_census(
        samples, release_id="release-1",
        collection_timestamp="2026-07-16T00:00:00Z",
        code_version="worktree:test", protocol_id="protocol-1")
    assert report["true_positive_n"] == 1
    assert report["false_positive_n"] == 1
    assert report["false_negative_n"] == 1
    assert report["true_negative_n"] == 1
    assert report["precision"] == report["recall"] == .5
    assert report["snapshot_state_n"] == 1
    assert report["unique_physical_snapshot_n"] == 1
    assert report["duplicate_alias_audit"] == {
        "performed": True,
        "passed": True,
        "physical_snapshot_identity": [
            "environment_instance", "snapshot_hash"],
        "duplicate_physical_snapshot_n": 0,
        "duplicate_state_alias_n": 0,
        "duplicate_state_aliases": [],
    }
    validate_live_mutation_census(report)

    forged = {**report, "precision": .99}
    with pytest.raises(MutationMiningError, match="aggregates/order"):
        validate_live_mutation_census(forged)


def test_live_census_rejects_duplicate_control_padding_and_fixture_rows():
    _snapshot, samples = _live_census_fixture()
    duplicated = replace(
        samples[1], control_id=samples[0].control_id,
        mutation_sample_id=make_mutation_sample_id(
            collection_run_id=samples[1].collection_run_id,
            environment_instance=samples[1].environment_instance,
            state_id=samples[1].state_id,
            snapshot_hash=samples[1].snapshot_hash,
            control_id=samples[0].control_id,
            bid=samples[1].bid,
            canonical_action=samples[1].canonical_action))
    with pytest.raises(MutationMiningError, match="cannot pad census"):
        build_live_mutation_census(
            [samples[0], duplicated, *samples[2:]], release_id="release-1",
            collection_timestamp="2026-07-16T00:00:00Z",
            code_version="worktree:test", protocol_id="protocol-1")

    for field_name, value, match in (
        ("is_fixture", True, "fixture/unspecified"),
        ("executed_live", False, "lacks live execution"),
        ("legal_at_snapshot", False, "not legal"),
        ("reset_restored", False, "not restored"),
    ):
        broken = replace(samples[0], **{field_name: value})
        with pytest.raises(MutationMiningError, match=match):
            broken.validate()


def test_live_census_rejects_200_rows_padded_by_state_aliases():
    """Fifty logical aliases cannot turn four physical controls into 200."""
    _snapshot, samples = _live_census_fixture()
    padded = []
    for alias_index in range(50):
        state_id = f"state-alias-{alias_index:02d}"
        for sample in samples:
            sample_id = make_mutation_sample_id(
                collection_run_id=sample.collection_run_id,
                environment_instance=sample.environment_instance,
                state_id=state_id,
                snapshot_hash=sample.snapshot_hash,
                control_id=sample.control_id,
                bid=sample.bid,
                canonical_action=sample.canonical_action,
            )
            padded.append(replace(
                sample, state_id=state_id, mutation_sample_id=sample_id))
    assert len(padded) == 200
    with pytest.raises(
            MutationMiningError,
            match="physical snapshot.*multiple state_id aliases"):
        build_live_mutation_census(
            padded, release_id="forged-release-200",
            collection_timestamp="2026-07-16T00:00:00Z",
            code_version="worktree:test", protocol_id="protocol-1")


def test_destructive_census_sample_requires_both_external_authorization_gates():
    _snapshot, samples = _live_census_fixture()
    destructive = replace(samples[0], safety_class="destructive")
    with pytest.raises(MutationMiningError, match="both authorization gates"):
        destructive.validate()
    replace(
        destructive, destructive_commit_authorized=True,
        destructive_env_gate=True).validate()


def test_immutable_census_manifest_snapshot_coverage_and_tamper(tmp_path):
    snapshot, samples = _live_census_fixture()
    _save_state_snapshot(tmp_path, snapshot)
    body = tmp_path / "audit" / MUTATION_REPORT_BODY_NAME
    manifest = tmp_path / "audit" / MUTATION_REPORT_MANIFEST_NAME
    status = save_live_mutation_census(
        samples, body=body, manifest=manifest, release_id="release-1",
        collection_timestamp="2026-07-16T00:00:00Z",
        code_version="worktree:test", protocol_id="protocol-1")
    assert status == "created"
    assert save_live_mutation_census(
        samples, body=body, manifest=manifest, release_id="release-1",
        collection_timestamp="2026-07-16T00:00:00Z",
        code_version="worktree:test", protocol_id="protocol-1",
    ) == "already_identical"
    assert_mutation_census_integrity(body, manifest)
    audit = audit_live_mutation_census(tmp_path)
    assert audit["integrity"] is True
    assert audit["snapshot_coverage"] is True
    assert audit["unique_physical_snapshot_n"] == 1
    assert audit["duplicate_alias_audit"]["performed"] is True
    assert audit["duplicate_alias_audit"]["passed"] is True
    assert audit["duplicate_alias_audit"]["duplicate_state_alias_n"] == 0
    assert audit["passed"] is False  # four real rows cannot satisfy >=200

    original_manifest = manifest.read_text()
    forged_manifest = json.loads(original_manifest)
    forged_manifest["n_samples"] = 400
    manifest.write_text(json.dumps(forged_manifest))
    manifest_tamper = audit_live_mutation_census(tmp_path)
    assert manifest_tamper["passed"] is False
    assert "manifest/hash pin mismatch" in manifest_tamper["error"]
    manifest.write_text(original_manifest)

    payload = json.loads(body.read_text())
    payload["samples"][0]["reference_mutated"] = False
    body.write_text(json.dumps(payload))
    tampered = audit_live_mutation_census(tmp_path)
    assert tampered["passed"] is False
    assert "aggregates/order" in tampered["error"]


def test_snapshot_control_omission_and_cli_fail_closed(tmp_path, capsys):
    snapshot, samples = _live_census_fixture()
    _save_state_snapshot(tmp_path, snapshot)
    # Internally self-consistent subset, but the state-bank replay exposes the
    # omitted fourth legal control.
    subset = samples[:3]
    subset_hash = mutation_control_set_sha256(
        (sample.bid, sample.canonical_action) for sample in subset)
    subset = [replace(
        sample, snapshot_control_count=3, control_set_sha256=subset_hash)
        for sample in subset]
    body = tmp_path / "audit" / MUTATION_REPORT_BODY_NAME
    manifest = tmp_path / "audit" / MUTATION_REPORT_MANIFEST_NAME
    save_live_mutation_census(
        subset, body=body, manifest=manifest, release_id="release-1",
        collection_timestamp="2026-07-16T00:00:00Z",
        code_version="worktree:test", protocol_id="protocol-1")
    audit = audit_live_mutation_census(tmp_path)
    assert audit["passed"] is False
    assert "control census mismatch" in audit["error"]
    assert cli.main([
        "audit-mutation-census", "--data-root", str(tmp_path)]) == 1
    assert cli.main([
        "audit-mutation-census", "--data-root", str(tmp_path),
        "--allow-blocked"]) == 0
    output = capsys.readouterr().out
    assert '"passed": false' in output


def test_snapshot_coverage_rejects_pruned_or_unattested_axtree(tmp_path):
    snapshot, samples = _live_census_fixture()
    state_path = tmp_path / "raw" / "state_bank" / "census.jsonl"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps({
        "state_id": "state-1",
        "axtree_snapshot": snapshot,
        # Deliberately no axtree_complete attestation.
    }) + "\n")
    body = tmp_path / "audit" / MUTATION_REPORT_BODY_NAME
    manifest = tmp_path / "audit" / MUTATION_REPORT_MANIFEST_NAME
    save_live_mutation_census(
        samples, body=body, manifest=manifest, release_id="release-1",
        collection_timestamp="2026-07-16T00:00:00Z",
        code_version="worktree:test", protocol_id="protocol-1")
    audit = audit_live_mutation_census(tmp_path)
    assert audit["passed"] is False
    assert "snapshot is absent from state bank" in audit["error"]


def test_snapshot_coverage_excludes_explicitly_disabled_controls(tmp_path):
    snapshot, samples = _live_census_fixture(include_disabled=True)
    _save_state_snapshot(tmp_path, snapshot)
    body = tmp_path / "audit" / MUTATION_REPORT_BODY_NAME
    manifest = tmp_path / "audit" / MUTATION_REPORT_MANIFEST_NAME
    save_live_mutation_census(
        samples, body=body, manifest=manifest, release_id="release-1",
        collection_timestamp="2026-07-16T00:00:00Z",
        code_version="worktree:test", protocol_id="protocol-1")
    audit = audit_live_mutation_census(tmp_path)
    assert audit["integrity"] is True
    assert audit["snapshot_coverage"] is True
