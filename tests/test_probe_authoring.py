import json
from dataclasses import replace

import pytest

from revact.cli import _load_formal_probe_specs
from revact.grounding.authoring import (
    ProbeAuthoringError,
    load_authored_specs,
    load_probe_execution_specs,
    promote_authored_spec,
    save_authored_spec,
    spec_from_workbench,
)


def proposal(**updates):
    row = {
        "name": "shopping.remove-cart-item",
        "site": "shopping",
        "action_type": "remove_cart_item",
        "candidate_id": "candidate-1",
        "state_id": "state-1",
        "action_instance_id": "action-1",
        "raw_action": "click('17')",
        "canonical_action": "click:remove_cart_item:item-1",
        "signal_channels": ["ui_structural", "api"],
        "undo_sequences": [["click('19')"]],
        "solver_set": ["site_specific_deterministic", "affordance_bfs",
                       "llm_undo_attacker"],
        "budget_k": 12,
        "safety_level": "self_recovering",
        "author": "test",
    }
    row.update(updates)
    return row


def make(row=None, timestamp="2026-07-13T00:00:00+00:00"):
    return spec_from_workbench(
        row or proposal(), timestamp=timestamp, controller_version="test-v1")


@pytest.mark.parametrize("field", [
    "label", "reversibility", "effect_status", "recovery_status",
    "normative_risk", "expected_decision",
])
def test_authoring_rejects_final_label_channels(field):
    with pytest.raises(ProbeAuthoringError, match="cannot accept final labels"):
        make(proposal(**{field: "RECOVERED"}))


def test_destructive_spec_retains_double_gate():
    spec = make(proposal(safety_level="destructive"))
    assert spec.requires_commit_flag is True
    assert spec.requires_destructive_env is True
    assert spec.fixture_status == spec.code_review_status == "PENDING"


def test_immutable_idempotent_store(tmp_path):
    path = tmp_path / "authored_specs.jsonl"
    first = make()
    replay = make(timestamp="2026-07-13T01:00:00+00:00")
    save_authored_spec(first, path)
    save_authored_spec(replay, path)
    assert len(load_authored_specs(path)) == 1
    assert len([line for line in path.read_text().splitlines() if line]) == 1
    persisted = json.loads(path.read_text())
    assert not any(key in persisted for key in ("label", "recovery_status"))


def test_cli_formal_probe_spec_is_point_keyed_and_label_free(tmp_path):
    draft = make(proposal(
        name="shopping.add_to_cart", action_type="add_to_cart",
        raw_action="click('3')", canonical_action="click:button:add-to-cart"))
    promoted = promote_authored_spec(draft, {
        "probe_point_id": "point-1", "probe_run_id": "probe-run-1",
        "candidate_snapshot_hash": "snapshot-hash",
        "environment_family": "webarena",
        "environment_instance": "shopping:7770",
        "environment_origin": "webarena", "is_mock": False,
        "task_id": "webarena.1", "trajectory_id": "trajectory-1",
        "run_id": "run-1", "seed": 0, "url": "http://shop/item/1",
        "account": "user", "privilege": "customer",
        "code_version": "deadbeef", "product_url": "http://shop/item/1",
    }, {"fixture_status": "PASSED", "code_review_status": "APPROVED",
        "reviewer": "reviewer", "review_timestamp": "2026-07-13T02:00:00+00:00"})
    row = promoted.to_dict()
    path = tmp_path / "point.jsonl"
    path.write_text(json.dumps(row) + "\n")
    loaded = _load_formal_probe_specs(path)
    assert loaded["shopping.add_to_cart"]["candidate_id"] == "candidate-1"
    assert loaded["shopping.add_to_cart"]["budget"] == 12

    row["recovery_status"] = "RECOVERED"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="cannot accept final labels"):
        _load_formal_probe_specs(path)


def test_pending_draft_cannot_be_promoted_to_execution():
    with pytest.raises(ProbeAuthoringError, match="fixture_status"):
        promote_authored_spec(make(), {
            "probe_point_id": "p", "probe_run_id": "r",
            "candidate_snapshot_hash": "h", "environment_family": "webarena",
            "environment_instance": "shop", "environment_origin": "webarena",
            "is_mock": False, "task_id": "t", "trajectory_id": "tr",
            "run_id": "run", "seed": 0, "url": "https://example.invalid",
            "account": "u", "privilege": "customer", "code_version": "v",
        }, {"fixture_status": "PENDING", "code_review_status": "APPROVED",
            "reviewer": "r", "review_timestamp": "now"})


def test_execution_spec_loader_accepts_multiline_jsonl(tmp_path):
    draft = make(proposal(
        name="shopping.add_to_cart", action_type="add_to_cart",
        raw_action="click('3')", canonical_action="click:button:add-to-cart"))
    first = promote_authored_spec(draft, {
        "probe_point_id": "point-1", "probe_run_id": "probe-run-1",
        "candidate_snapshot_hash": "snapshot-hash",
        "environment_family": "webarena", "environment_instance": "shop",
        "environment_origin": "webarena", "is_mock": False,
        "task_id": "webarena.1", "trajectory_id": "trajectory-1",
        "run_id": "run-1", "seed": 0, "url": "http://shop/item/1",
        "account": "user", "privilege": "customer", "code_version": "v",
    }, {"fixture_status": "PASSED", "code_review_status": "APPROVED",
        "reviewer": "r", "review_timestamp": "now"})
    second = replace(
        first, probe_name="shopping.add_to_cart.second",
        probe_point_id="point-2")
    path = tmp_path / "batch.jsonl"
    path.write_text("".join(json.dumps(spec.to_dict()) + "\n"
                            for spec in (first, second)))
    assert [spec.probe_point_id for spec in load_probe_execution_specs(path)] == [
        "point-1", "point-2"]

    path.write_text("".join(json.dumps(spec.to_dict()) + "\n" for spec in (
        first, replace(second, probe_run_id="different-run"))))
    with pytest.raises(ProbeAuthoringError, match="share exactly one"):
        load_probe_execution_specs(path)
