"""Canonical grounding ontology, persistence and legacy quarantine gates."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from revact.grounding import base as grounding_base
from revact.grounding.base import mk_result, save_results
from revact.grounding.migration import migrate_legacy_grounding
from revact.grounding.schema import (
    EFFECT_CHANGED,
    EFFECT_NO_EFFECT,
    EFFECT_UNKNOWN,
    RECOVERY_NOT_WITHIN_BUDGET,
    RECOVERY_PARTIAL,
    RECOVERY_RECOVERED,
    RECOVERY_UNKNOWN,
    GroundingPoint,
    GroundingValidationError,
    apply_solver_union,
    assert_manifest_integrity,
    legacy_label_to_statuses,
    load_probe_points,
    save_probe_points,
    statuses_to_display_label,
)


def _point(**updates) -> GroundingPoint:
    point = GroundingPoint(
        probe_point_id="point-1",
        probe_run_id="probe-run-1",
        probe_name="shopping.add_to_cart",
        state_id="state-1",
        candidate_id="candidate-1",
        action_instance_id="action-1",
        action_type="add_to_cart",
        raw_action="click('42')",
        canonical_action="click:add_to_cart:sku-1",
        site="shopping",
        environment_family="webarena",
        environment_instance="shopping:7770",
        environment_origin="webarena",
        task_id="webarena.1",
        trajectory_id="traj-1",
        run_id="run-1",
        seed=0,
        url="http://shopping/product/1",
        account="user-1",
        privilege="customer",
        budget_k=12,
        solver_set=["site_specific_deterministic"],
        controller_version="test-controller",
        pre_observation_hash="pre-hash",
        pre_signal={"cart_count": 0},
        post_observation_hash="post-hash",
        post_signal={"cart_count": 1},
        undo_actions=["click('remove-1')"],
        undo_semantic_actions=["remove_cart_item(sku-1)"],
        undo_observation_hashes=["undo-hash"],
        final_signal={"cart_count": 0},
        effect_status=EFFECT_CHANGED,
        recovery_status=RECOVERY_RECOVERED,
        undo_cost_steps=1,
        residual_diff={"cart_count": 0},
        timestamp="2026-07-13T00:00:00+00:00",
        code_version="deadbeef",
        evidence={"measurement": "fixture",
                  "candidate_snapshot_hash": "candidate-snapshot-hash"},
    )
    return replace(point, **updates)


def test_legacy_label_mapping_never_emits_new_irreversible():
    assert legacy_label_to_statuses("NO_EFFECT") == (
        EFFECT_NO_EFFECT, RECOVERY_UNKNOWN)
    assert legacy_label_to_statuses("IRREVERSIBLE") == (
        EFFECT_CHANGED, RECOVERY_NOT_WITHIN_BUDGET)
    assert statuses_to_display_label(
        EFFECT_CHANGED, RECOVERY_NOT_WITHIN_BUDGET) == \
        "NOT_RECOVERED_WITHIN_BUDGET"


def test_grounding_schema_round_trip_and_manifest_1_to_1(tmp_path):
    body = tmp_path / "probe_points.jsonl"
    manifest = tmp_path / "POINT_MANIFEST.jsonl"
    save_probe_points([_point()], body, manifest)
    assert load_probe_points(body)["point-1"].undo_cost_steps == 1
    assert_manifest_integrity(body, manifest)
    assert body.read_text().count("\n") == manifest.read_text().count("\n") == 1


def test_every_legal_effect_recovery_pair_round_trips(tmp_path):
    """Exercise every ontology cell that the formal validator can admit.

    The two axes are orthogonal, but NO_EFFECT/UNKNOWN effects intentionally
    make recovery inapplicable. A single RECOVERED happy path would not catch
    serialization drift in the partial, failed-search, or unknown branches.
    """
    solver_set = ["deterministic", "bfs", "llm-attacker"]
    traces = [
        {"solver_name": "deterministic",
         "solver_kind": "site_specific_deterministic", "budget_k": 12,
         "success": False, "undo_actions": [],
         "termination_reason": "no_channel_found", "seeds": [],
         "start_state_hash": "post-hash", "start_signal": {"cart_count": 1},
         "reset_method": "fixture_state_clone", "reset_verified": True},
        {"solver_name": "bfs", "solver_kind": "affordance_bfs",
         "budget_k": 12, "success": False, "undo_actions": [],
         "termination_reason": "budget_exhausted", "seeds": [],
         "start_state_hash": "post-hash", "start_signal": {"cart_count": 1},
         "reset_method": "fixture_state_clone", "reset_verified": True},
        {"solver_name": "llm-attacker", "solver_kind": "llm_undo_attacker",
         "budget_k": 12, "success": False, "undo_actions": [],
         "termination_reason": "budget_exhausted", "seeds": [7, 11],
         "start_state_hash": "post-hash", "start_signal": {"cart_count": 1},
         "reset_method": "fixture_state_clone", "reset_verified": True},
    ]
    negative_evidence = {
        "measurement": "fixture",
        "candidate_snapshot_hash": "candidate-snapshot-hash",
        "solver_union": {
            "protocol": "solver_union.v2",
            "recovery_status": RECOVERY_NOT_WITHIN_BUDGET,
            "budget_k": 12,
            "solver_set": solver_set,
            "successful_solver": None,
            "post_action_state_hash": "post-hash",
            "post_signal": {"cart_count": 1},
            "traces": traces,
        },
    }
    points = [
        _point(probe_point_id="point-recovered"),
        _point(
            probe_point_id="point-partial", recovery_status=RECOVERY_PARTIAL,
            final_signal={"cart_count": 0.5},
            residual_diff={"cart_count": [0, 0.5]}),
        _point(
            probe_point_id="point-not-recovered",
            recovery_status=RECOVERY_NOT_WITHIN_BUDGET,
            solver_set=solver_set, undo_actions=[], undo_semantic_actions=[],
            undo_observation_hashes=[], undo_cost_steps=None,
            final_signal={"cart_count": 1},
            residual_diff={"cart_count": [0, 1]}, budget_exhausted=True,
            evidence=negative_evidence),
        _point(
            probe_point_id="point-recovery-unknown",
            recovery_status=RECOVERY_UNKNOWN, undo_actions=[],
            undo_semantic_actions=[], undo_observation_hashes=[],
            undo_cost_steps=None, final_signal={}, residual_diff={}),
        _point(
            probe_point_id="point-no-effect", effect_status=EFFECT_NO_EFFECT,
            recovery_status=RECOVERY_UNKNOWN, post_signal={"cart_count": 0},
            undo_actions=[], undo_semantic_actions=[],
            undo_observation_hashes=[], undo_cost_steps=None,
            final_signal={}, residual_diff={}),
        _point(
            probe_point_id="point-effect-unknown", effect_status=EFFECT_UNKNOWN,
            recovery_status=RECOVERY_UNKNOWN, pre_signal={}, post_signal={},
            undo_actions=[], undo_semantic_actions=[],
            undo_observation_hashes=[], undo_cost_steps=None,
            final_signal={}, residual_diff={}),
    ]
    body = tmp_path / "probe_points.jsonl"
    manifest = tmp_path / "POINT_MANIFEST.jsonl"
    save_probe_points(points, body, manifest, append=False)

    loaded = load_probe_points(body)
    assert {point_id: point.to_dict() for point_id, point in loaded.items()} == {
        point.probe_point_id: point.to_dict() for point in points}
    assert {
        (point.effect_status, point.recovery_status) for point in loaded.values()
    } == {
        (EFFECT_CHANGED, RECOVERY_RECOVERED),
        (EFFECT_CHANGED, RECOVERY_PARTIAL),
        (EFFECT_CHANGED, RECOVERY_NOT_WITHIN_BUDGET),
        (EFFECT_CHANGED, RECOVERY_UNKNOWN),
        (EFFECT_NO_EFFECT, RECOVERY_UNKNOWN),
        (EFFECT_UNKNOWN, RECOVERY_UNKNOWN),
    }
    assert_manifest_integrity(body, manifest)
    assert body.read_text().count("\n") == manifest.read_text().count("\n") == 6


@pytest.mark.parametrize("field", [
    "schema_version", "is_mock", "effect_status", "recovery_status",
    "budget_exhausted",
])
def test_formal_wire_schema_never_synthesizes_missing_fields(field):
    row = _point().to_dict()
    row.pop(field)
    with pytest.raises(GroundingValidationError, match="missing serialized fields"):
        GroundingPoint.from_dict(row, validate=True)


def test_formal_effect_and_recovery_require_measured_signal_evidence():
    empty = _point(pre_signal={}, post_signal={}, final_signal={})
    errors = empty.validation_errors(formal=True)
    assert "measured effect requires non-empty pre_signal" in errors
    assert "measured effect requires non-empty post_signal" in errors
    assert "measured recovery requires non-empty final_signal" in errors

    unchanged = _point(post_signal={"cart_count": 0})
    assert "CHANGED requires pre_signal != post_signal" in \
        unchanged.validation_errors(formal=True)

    unrestored = _point(final_signal={"cart_count": 1})
    assert "RECOVERED requires final_signal == pre_signal" in \
        unrestored.validation_errors(formal=True)


def test_formal_point_requires_unique_point_provenance(tmp_path):
    with pytest.raises(GroundingValidationError, match="state_id"):
        _point(state_id="").validate(formal=True)
    with pytest.raises(GroundingValidationError, match="overwrite immutable"):
        body = tmp_path / "points.jsonl"
        manifest = tmp_path / "manifest.jsonl"
        save_probe_points([_point()], body, manifest)
        save_probe_points([_point()], body, manifest)


def test_no_effect_is_not_a_recovery_class():
    bad = _point(effect_status=EFFECT_NO_EFFECT,
                 recovery_status=RECOVERY_RECOVERED,
                 undo_cost_steps=0, undo_actions=[])
    with pytest.raises(GroundingValidationError, match="NO_EFFECT"):
        bad.validate(formal=True)


def test_budget_exhaustion_does_not_equal_irreversible():
    traces = [
        {"solver_name": "deterministic", "solver_kind": "site_specific_deterministic",
         "budget_k": 12, "success": False, "undo_actions": [],
         "termination_reason": "no_channel_found", "seeds": [],
         "start_state_hash": "post-hash", "start_signal": {"cart_count": 1},
         "reset_method": "fixture_state_clone", "reset_verified": True},
        {"solver_name": "bfs", "solver_kind": "affordance_bfs",
         "budget_k": 12, "success": False, "undo_actions": [],
         "termination_reason": "budget_exhausted", "seeds": [],
         "start_state_hash": "post-hash", "start_signal": {"cart_count": 1},
         "reset_method": "fixture_state_clone", "reset_verified": True},
        {"solver_name": "llm-attacker", "solver_kind": "llm_undo_attacker",
         "budget_k": 12, "success": False, "undo_actions": [],
         "termination_reason": "budget_exhausted", "seeds": [7, 11],
         "start_state_hash": "post-hash", "start_signal": {"cart_count": 1},
         "reset_method": "fixture_state_clone", "reset_verified": True},
    ]
    point = _point(
        recovery_status=RECOVERY_NOT_WITHIN_BUDGET,
        undo_cost_steps=None,
        undo_actions=[],
        undo_semantic_actions=[],
        undo_observation_hashes=[],
        final_signal={"cart_count": 1},
        budget_exhausted=True,
        solver_set=["deterministic", "bfs", "llm-attacker"],
        evidence={"candidate_snapshot_hash": "candidate-snapshot-hash",
                  "solver_union": {
            "protocol": "solver_union.v2",
            "recovery_status": RECOVERY_NOT_WITHIN_BUDGET,
            "budget_k": 12,
            "solver_set": ["deterministic", "bfs", "llm-attacker"],
            "successful_solver": None,
            "post_action_state_hash": "post-hash",
            "post_signal": {"cart_count": 1},
            "traces": traces}},
    )
    point.validate(formal=True)
    assert point.display_label == "NOT_RECOVERED_WITHIN_BUDGET"
    assert "IRREVERSIBLE" not in json.dumps(point.to_dict())


def test_negative_rejects_successful_solver_and_partial_rejects_full_restore():
    negative = _point(
        recovery_status=RECOVERY_NOT_WITHIN_BUDGET,
        undo_cost_steps=None, undo_actions=[], final_signal={"cart_count": 1},
        solver_set=["deterministic", "bfs", "llm-attacker"],
        evidence={"candidate_snapshot_hash": "candidate-snapshot-hash",
                  "solver_union": {
            "protocol": "solver_union.v2",
            "recovery_status": RECOVERY_NOT_WITHIN_BUDGET,
            "budget_k": 12,
            "solver_set": ["deterministic", "bfs", "llm-attacker"],
            "successful_solver": "bfs",
            "post_action_state_hash": "post-hash",
            "post_signal": {"cart_count": 1},
            "traces": [
                {"solver_name": "deterministic",
                 "solver_kind": "site_specific_deterministic", "budget_k": 12,
                 "success": False, "undo_actions": [],
                 "termination_reason": "no_channel_found", "seeds": [],
                 "start_state_hash": "post-hash",
                 "start_signal": {"cart_count": 1},
                 "reset_method": "fixture_state_clone", "reset_verified": True},
                {"solver_name": "bfs", "solver_kind": "affordance_bfs",
                 "budget_k": 12, "success": True, "undo_actions": ["remove"],
                 "termination_reason": "recovered", "seeds": [],
                 "start_state_hash": "post-hash",
                 "start_signal": {"cart_count": 1},
                 "reset_method": "fixture_state_clone", "reset_verified": True},
                {"solver_name": "llm-attacker", "solver_kind": "llm_undo_attacker",
                 "budget_k": 12, "success": False, "undo_actions": [],
                 "termination_reason": "budget_exhausted", "seeds": [7, 11],
                 "start_state_hash": "post-hash",
                 "start_signal": {"cart_count": 1},
                 "reset_method": "fixture_state_clone", "reset_verified": True},
            ]}})
    errors = negative.validation_errors(formal=True)
    assert any("successful_solver" in error for error in errors)
    assert any("not an explicit failure" in error for error in errors)

    partial = _point(
        recovery_status="PARTIALLY_RECOVERED", final_signal={"cart_count": 0},
        residual_diff={}, undo_actions=["click('remove-1')"], undo_cost_steps=1)
    errors = partial.validation_errors(formal=True)
    assert any("fully restored" in error for error in errors)
    assert any("non-empty residual_diff" in error for error in errors)


def test_legacy_migration_is_non_destructive_and_creates_zero_formal_points(tmp_path):
    grounded = tmp_path / "grounded"
    grounded.mkdir()
    body = grounded / "reversibility.jsonl"
    manifest = grounded / "MANIFEST.jsonl"
    body.write_text(
        json.dumps({"action_type": "place_order", "label": "IRREVERSIBLE"}) + "\n" +
        json.dumps({"probe_id": "p-1", "probe_name": "shopping.add_to_cart",
                    "action_type": "add_to_cart", "label": "REVERSIBLE"}) + "\n")
    manifest.write_text(json.dumps({"probe_id": "p-1"}) + "\n")
    before = hashlib.sha256(body.read_bytes()).hexdigest()
    report = migrate_legacy_grounding(tmp_path)
    assert hashlib.sha256(body.read_bytes()).hexdigest() == before
    assert report["legacy_quarantine_rows"] == 1
    assert report["class_probe_smoke_rows"] == 1
    assert report["formal_points_created"] == 0
    assert str(grounded / "probe_points.jsonl") not in report["rollback"]
    assert str(grounded / "POINT_MANIFEST.jsonl") not in report["rollback"]
    assert report["canonical_point_artifacts"]["rollback_policy"].startswith(
        "never delete or rewrite")
    assert (grounded / "probe_points.jsonl").read_text() == ""
    assert (grounded / "POINT_MANIFEST.jsonl").read_text() == ""
    quarantined = json.loads(
        (grounded / "quarantine" / "legacy_rows.jsonl").read_text())
    assert "missing_probe_id" in quarantined["reason_codes"]


def test_new_class_smoke_run_cannot_append_to_frozen_legacy_body(tmp_path):
    legacy = tmp_path / "grounded" / "reversibility.jsonl"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"frozen": true}\n')
    before = hashlib.sha256(legacy.read_bytes()).hexdigest()

    path = save_results([mk_result(
        "add_to_cart", "UNKNOWN", "fixture", False,
        {"reason": "smoke-only"})], out_dir=tmp_path)

    assert path == tmp_path / "grounded" / "smoke" / "reversibility.jsonl"
    assert hashlib.sha256(legacy.read_bytes()).hexdigest() == before
    assert path.read_text().count("\n") == 1
    assert (path.parent / "MANIFEST.jsonl").read_text().count("\n") == 1


def test_formal_probe_persistence_rejects_missing_candidate_artifact(
        tmp_path, monkeypatch):
    """The point writer must translate a missing/corrupt candidate index into
    the canonical grounding exception instead of failing with a NameError.
    """
    monkeypatch.setattr(
        grounding_base, "grounding_point_from_result",
        lambda _result, _ctx: _point())
    with pytest.raises(GroundingValidationError,
                       match="formal candidate artifact invalid"):
        grounding_base.save_formal_probe_results(
            [(object(), object())], out_dir=tmp_path)


def test_manifest_hash_tampering_is_detected(tmp_path):
    body = tmp_path / "points.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    save_probe_points([_point()], body, manifest)
    row = json.loads(manifest.read_text())
    row["record_sha256"] = "bad"
    manifest.write_text(json.dumps(row) + "\n")
    with pytest.raises(GroundingValidationError, match="manifest"):
        assert_manifest_integrity(body, manifest)


def test_solver_union_evidence_round_trips_into_point():
    union = SimpleNamespace(
        recovery_status=RECOVERY_RECOVERED,
        budget_k=3,
        solver_set=("deterministic", "affordance_bfs"),
        undo_actions=("click('menu')", "click('remove')"),
        undo_semantic_actions=("open menu", "remove item"),
        undo_observation_hashes=("undo-1", "undo-2"),
        undo_cost_steps=2,
        final_signal={"cart_count": 0},
        residual_diff={"cart_count": 0},
        budget_exhausted=False,
        to_evidence=lambda: {"protocol": "solver_union.v1", "traces": [1, 2]},
    )
    point = apply_solver_union(_point(), union)
    assert point.undo_cost_steps == 2
    assert point.solver_set == ["deterministic", "affordance_bfs"]
    assert point.evidence["solver_union"]["protocol"] == "solver_union.v1"
