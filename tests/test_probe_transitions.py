"""Observation-body provenance and immutable transition sidecars."""
from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from revact.grounding.transitions import (
    TRANSITION_SCHEMA_VERSION, ExecutedTransitionStep, ObservationBody,
    ProbeTransition, TransitionValidationError,
    assert_point_transition_integrity, assert_transition_manifest_integrity,
    load_probe_transitions, save_probe_transitions)


PRE = "RootWebArea 'Product'\n[42] button 'Add to Cart'\n"
POST = "RootWebArea 'Product'\n[42] button 'Add to Cart'\nStaticText 'Added'\n"
CART = "RootWebArea 'Cart'\n[55] link 'Remove item'\n"
EMPTY = "RootWebArea 'Cart'\nStaticText 'empty'\n"


def _observation(tree: str, stage: str, url: str, anchors=()):
    return ObservationBody.capture(
        {"url": url, "axtree_txt": tree}, stage=stage,
        anchor_bids=anchors)


def _transition() -> ProbeTransition:
    signal_page = _observation(
        CART, "post_signal_measurement", "http://shop/cart")
    recovered = _observation(
        EMPTY, "recovery_after_1", "http://shop/cart", ("55",))
    return ProbeTransition(
        schema_version=TRANSITION_SCHEMA_VERSION,
        transition_id="transition::point-1",
        probe_point_id="point-1",
        probe_run_id="run-1",
        state_id="state-1",
        candidate_id="candidate-1",
        action_instance_id="action-1",
        action_type="add_to_cart",
        raw_action="click('42')",
        canonical_action="click:button:add-to-cart",
        candidate_snapshot_hash="candidate-snapshot-hash",
        pre_observation=_observation(
            PRE, "pre_action", "http://shop/product", ("42",)),
        post_observation=_observation(
            POST, "post_action_immediate", "http://shop/product", ("42",)),
        post_signal_observation=signal_page,
        recovery_steps=[
            ExecutedTransitionStep(
                0, "goto('http://shop/cart')", "navigate_to_signal_page",
                signal_page, "measurement_navigation"),
            ExecutedTransitionStep(
                1, "click('55')", "remove_cart_line", recovered,
                "recovery"),
        ],
        pre_signal={"signal": "cart_item_count", "count": 0},
        post_signal={"signal": "cart_item_count", "count": 1},
        final_signal={"signal": "cart_item_count", "count": 0},
        effect_status="CHANGED",
        recovery_status="RECOVERED",
        undo_cost_steps=2,
        budget_k=12,
        replay_verification="exact_snapshot",
        replay_target_contract={},
        timestamp="2026-07-15T00:00:00+00:00",
        code_version="deadbeef",
    )


def test_transition_round_trip_and_exact_body_hashes(tmp_path):
    transition = _transition()
    transition.validate()
    body = tmp_path / "probe_transitions.v1.jsonl"
    manifest = tmp_path / "TRANSITION_MANIFEST.v1.jsonl"
    save_probe_transitions([transition], body, manifest)
    loaded = load_probe_transitions(body)
    assert loaded[transition.transition_id] == transition
    assert_transition_manifest_integrity(body, manifest)
    assert json.loads(body.read_text())["pre_observation"]["raw_axtree"] == PRE


def test_observation_and_manifest_hash_tampering_fail_closed(tmp_path):
    transition = _transition()
    row = transition.to_dict()
    row["post_observation"]["raw_axtree"] += "tampered"
    with pytest.raises(TransitionValidationError, match="raw_axtree_sha256"):
        ProbeTransition.from_dict(row)

    body = tmp_path / "body.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    save_probe_transitions([transition], body, manifest)
    manifest_row = json.loads(manifest.read_text())
    manifest_row["record_sha256"] = "forged"
    manifest.write_text(json.dumps(manifest_row) + "\n")
    with pytest.raises(TransitionValidationError, match="manifest"):
        assert_transition_manifest_integrity(body, manifest)


def test_transition_wire_schema_rejects_missing_and_unknown_fields():
    row = _transition().to_dict()
    row["invented"] = True
    with pytest.raises(TransitionValidationError, match="unknown"):
        ProbeTransition.from_dict(row)
    row = _transition().to_dict()
    del row["post_signal_observation"]
    with pytest.raises(TransitionValidationError, match="missing"):
        ProbeTransition.from_dict(row)


def test_transition_ids_are_immutable(tmp_path):
    body = tmp_path / "body.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    save_probe_transitions([_transition()], body, manifest)
    with pytest.raises(TransitionValidationError, match="immutable"):
        save_probe_transitions([_transition()], body, manifest)


def test_point_transition_cross_asset_gate_reports_unrecoverable_legacy_gap():
    transition = _transition()
    point = SimpleNamespace(
        state_id=transition.state_id,
        candidate_id=transition.candidate_id,
        action_instance_id=transition.action_instance_id,
        pre_observation_hash=transition.pre_observation.policy_axtree_sha256,
        post_observation_hash=transition.post_observation.policy_axtree_sha256,
        effect_status=transition.effect_status,
        recovery_status=transition.recovery_status,
        undo_cost_steps=transition.undo_cost_steps,
    )
    legacy = SimpleNamespace(**vars(point))
    report = assert_point_transition_integrity(
        {"point-1": point, "legacy-hash-only": legacy},
        {transition.transition_id: transition}, require_all=False)
    assert report["covered_point_count"] == 1
    assert report["missing_transition_point_ids"] == ["legacy-hash-only"]
    with pytest.raises(TransitionValidationError, match="legacy-hash-only"):
        assert_point_transition_integrity(
            {"point-1": point, "legacy-hash-only": legacy},
            {transition.transition_id: transition}, require_all=True)


def test_supervised_bid_and_recovery_alignment_are_hard_gates():
    transition = _transition()
    invisible = replace(
        transition,
        pre_observation=_observation(
            "RootWebArea 'Other'\n[7] button 'Continue'\n",
            "pre_action", "http://shop/product", ("7",)))
    with pytest.raises(TransitionValidationError, match=r"bid \[42\]"):
        invisible.validate()
    mismatched = replace(transition, undo_cost_steps=1)
    with pytest.raises(TransitionValidationError, match="recovery_steps"):
        mismatched.validate()
