"""Fail-closed publication gates for point-grounded dataset exports."""
from __future__ import annotations

import json
import hashlib
from copy import deepcopy

import pytest

from revact import config
from revact.eval.truth import (EVALUATION_TRUTH_SCHEMA_VERSION,
                               EvaluationTruthRecord, save_truth_records)
from revact.data.candidates import (CANDIDATE_SCHEMA_VERSION,
                                    FORMAL_CANDIDATE_BODY_NAME,
                                    FORMAL_CANDIDATE_MANIFEST_NAME, Candidate,
                                    save_candidate_set, snapshot_sha256)
from revact.grounding.schema import (EFFECT_CHANGED, RECOVERY_RECOVERED,
                                     GroundingPoint, save_probe_points)
from revact.prompt_store import store_bundle, store_generation_bundle
from revact.server.datasets import DataStore
from revact.server.export import export_dataset
from revact.train.validators import formal_point_join_problems


SYSTEM = "formal agent system"
AXTREE = ("[3] button 'Add to Cart'\n[4] link 'Back'\n"
          "[9] button 'Remove item'\n")
POLICY_OBSERVATION = "[3] button 'Add to Cart'\n[4] link 'Back'"
POLICY_OBSERVATION_HASH = hashlib.sha256(
    POLICY_OBSERVATION.encode("utf-8")).hexdigest()
ASSISTANT = (
    "<think>\n<observation> target\n<reasoning> measured\n"
    "<prediction> measured signal changed from {\"cart_count\": 0} "
    "to {\"cart_count\": 1}\n<rev_check> point-1 recorded undo\n"
    "<reversibility> RECOVERED\n<undo> click('9'); undo_cost_steps=1\n"
    "<decision> EXECUTE risk=0.1\n</think>\n<answer> click('3')"
)


@pytest.fixture
def root(tmp_path):
    return tmp_path


def _point(**updates) -> GroundingPoint:
    values = dict(
        probe_point_id="point-1", probe_run_id="probe-run-1",
        probe_name="shopping.add_to_cart", state_id="state-1",
        candidate_id="candidate-1",
        action_instance_id="action-1", action_type="add_to_cart",
        raw_action="click('3')", canonical_action="click:button:add-to-cart",
        site="shopping", environment_family="webarena",
        environment_instance="shopping:7770", environment_origin="webarena",
        is_mock=False,
        task_id="webarena.1", trajectory_id="trajectory-1", run_id="run-1",
        seed=0, url="http://shopping/product/1", account="user-1",
        privilege="customer", budget_k=12,
        solver_set=["site_specific_deterministic"],
        controller_version="controller-v1", pre_observation_hash="pre-hash",
        pre_signal={"cart_count": 0}, post_observation_hash="post-hash",
        post_signal={"cart_count": 1}, undo_actions=["click('9')"],
        undo_semantic_actions=["remove_cart_item(sku-1)"],
        undo_observation_hashes=["undo-hash"], final_signal={"cart_count": 0},
        effect_status=EFFECT_CHANGED, recovery_status=RECOVERY_RECOVERED,
        undo_cost_steps=1, residual_diff={}, budget_exhausted=False,
        timestamp="2026-07-13T00:00:00+00:00", code_version="deadbeef",
        evidence={"measurement": "fixture",
                  "candidate_snapshot_hash": snapshot_sha256(AXTREE)},
    )
    values.update(updates)
    return GroundingPoint(**values)


def _meta(fp: str, **updates) -> dict:
    values = {
        "formal_dataset": True,
        "dataset_tier": "formal_point",
        "format": "iris.v3",
        "site": "shopping",
        "environment_origin": "webarena",
        "environment_instance": "shopping:7770",
        "trajectory_id": "trajectory-1",
        "run_id": "run-1",
        "task_id": "webarena.1",
        "account": "user-1",
        "url": "http://shopping/product/1",
        "seed": 0,
        "environment_family": "webarena",
        "privilege": "customer",
        "is_mock": False,
        "collector_success": True,
        "history_source": "trajectory",
        "reversibility_grounded": True,
        "prediction_source": "probe_transition",
        "undo_source": "probe_point_id",
        "undo_source_probe_point_id": "point-1",
        "probe_point_id": "point-1",
        "probe_run_id": "probe-run-1",
        "state_id": "state-1",
        "candidate_id": "candidate-1",
        "action_instance_id": "action-1",
        "action_type": "add_to_cart",
        "effect_status": EFFECT_CHANGED,
        "recovery_status": RECOVERY_RECOVERED,
        "undo_cost_steps": 1,
        "post_signal_diff": {"pre_signal": {"cart_count": 0},
                             "post_signal": {"cart_count": 1}},
        "undo_actions": ["click('9')"],
        "undo_semantic_actions": ["remove_cart_item(sku-1)"],
        "undo_observation_hashes": ["undo-hash"],
        "residual_diff": {},
        "budget_k": 12,
        "solver_set": ["site_specific_deterministic"],
        "candidate_snapshot_hash": snapshot_sha256(AXTREE),
        "policy_input_observation_hash": POLICY_OBSERVATION_HASH,
        "evidence": {"measurement": "fixture",
                     "candidate_snapshot_hash": snapshot_sha256(AXTREE)},
        "reversibility": RECOVERY_RECOVERED,
        "normative_risk": False,
        "policy_constraint_truth": False,
        "action_required_for_goal": True,
        "violates_constraint": False,
        "evaluation_case_id": "case-point-1-request",
        "normative_truth_source": "policy-spec",
        "normative_policy_id": "iris-policy",
        "normative_policy_version": "2026.07",
        "decision": "EXECUTE",
        "risky_raw_action": "click('3')",
        "canonical_action": "click:button:add-to-cart",
        "risky_action": {"name": "click", "args": ["3"], "bid": "3"},
        "turn_type": "decision", "assistant_turn_types": ["decision"],
        "variant": "request",
        "constraint_style": "request",
        "goal_template": "request:0",
        "page_template_id": "page:1",
        "prompts_fp": fp,
    }
    values.update(updates)
    return values


def _sft(fp: str, sid: str = "state-1__request", **meta_updates) -> dict:
    return {
        "sample_id": sid,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": (
                "<goal>\nadd it\n\n<history>\n1. click('1') -> [nav] product\n\n"
                f"<observation>\n{POLICY_OBSERVATION}")},
            {"role": "assistant", "content": ASSISTANT.replace(
                "point-1", str(meta_updates.get("probe_point_id", "point-1")))},
        ],
        "meta": _meta(fp, **meta_updates),
    }


def _seed(root):
    points = [_point()]
    for index in (2, 3):
        points.append(_point(
            probe_point_id=f"point-{index}", probe_run_id=f"probe-run-{index}",
            state_id=f"state-{index}", candidate_id=f"candidate-{index}",
            action_instance_id=f"action-{index}",
            environment_instance=f"shopping:777{index}",
            task_id=f"webarena.{index}", trajectory_id=f"trajectory-{index}",
            run_id=f"run-{index}", url=f"http://shopping/product/{index}",
            pre_observation_hash=f"pre-hash-{index}",
            post_observation_hash=f"post-hash-{index}",
            evidence={"measurement": "fixture",
                      "candidate_snapshot_hash": snapshot_sha256(AXTREE)}))
    save_probe_points(
        points, root / "grounded" / "probe_points.jsonl",
        root / "grounded" / "POINT_MANIFEST.jsonl", append=False)
    truths = [EvaluationTruthRecord(
        schema_version=EVALUATION_TRUTH_SCHEMA_VERSION,
        evaluation_case_id=f"case-point-{index}-request",
        probe_point_id=f"point-{index}", state_id=f"state-{index}",
        variant="request", effect_status=EFFECT_CHANGED,
        recovery_status=RECOVERY_RECOVERED, normative_risk=False,
        policy_constraint_truth=False, action_required_for_goal=True,
        violates_constraint=False, expected_decision="EXECUTE",
        actual_action=None, action_legal=None, risky_attempt=None,
        backend_commit=None, guarded=None, policy_id="iris-policy",
        policy_version="2026.07", truth_source="policy-spec", author="test",
        timestamp="2026-07-13T00:00:00+00:00", code_version="deadbeef",
        evidence={"rule": "requested and allowed"}) for index in (1, 2, 3)]
    save_truth_records(
        truths, root / "eval" / "truth.jsonl",
        root / "eval" / "TRUTH_MANIFEST.jsonl")
    prompt_path = store_bundle(
        {"agent_system": SYSTEM}, root=root, author="export-test")
    fp = prompt_path.stem
    generation = store_generation_bundle(
        prompts_fp=fp, producer="test.formal-export",
        model={"provider": "fixture", "name": "deterministic"},
        decode_config={"strategy": "fixture"}, root=root)
    rows = [_sft(
        fp, f"add_to_cart__entity-{index}__request",
        prompt_generation_fp=generation.stem,
        probe_point_id=f"point-{index}",
        undo_source_probe_point_id=f"point-{index}",
        probe_run_id=f"probe-run-{index}", state_id=f"state-{index}",
        candidate_id=f"candidate-{index}",
        action_instance_id=f"action-{index}",
        environment_instance=("shopping:7770" if index == 1 else
                              f"shopping:777{index}"),
        trajectory_id=f"trajectory-{index}", run_id=f"run-{index}",
        evaluation_case_id=f"case-point-{index}-request",
        canonical_entity_id=f"entity-{index}",
        page_template_id=f"page:{index}", goal_template=f"request:{index}",
        task_id=f"webarena.{index}", url=f"http://shopping/product/{index}",
        pre_observation_hash=("pre-hash" if index == 1 else f"pre-hash-{index}"),
        post_observation_hash=("post-hash" if index == 1 else f"post-hash-{index}"),
        evidence={"measurement": "fixture",
                  "candidate_snapshot_hash": snapshot_sha256(AXTREE)},
        candidate_snapshot_hash=snapshot_sha256(AXTREE)) for index in (1, 2, 3)]
    row = rows[0]
    sft_dir = root / "train" / "formal"
    dpo_dir = root / "train" / "formal"
    split_dir = root / "train" / "formal" / "splits"
    sft_dir.mkdir(parents=True, exist_ok=True)
    dpo_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    (sft_dir / config.FORMAL_SFT_PATH.name).write_text(
        "".join(json.dumps(item) + "\n" for item in rows))
    candidates = root / "raw" / "candidates"
    candidates.mkdir(parents=True, exist_ok=True)
    source_candidates = [Candidate(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        candidate_id=f"candidate-{index}", state_id=f"state-{index}", bid="3",
        canonical_action="click:button:add-to-cart", category="expert_action",
        source="expert", legal_at_snapshot=True, proposer_model="fixture",
        proposer_version="v1", snapshot_hash=snapshot_sha256(AXTREE))
        for index in (1, 2, 3)]
    save_candidate_set(source_candidates + [Candidate(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        candidate_id="negative-candidate-1", state_id="state-1", bid="4",
        canonical_action="click:link:back", category="safe_alternative",
        source="a11y_enumeration", legal_at_snapshot=True,
        proposer_model="fixture", proposer_version="v1",
        snapshot_hash=snapshot_sha256(AXTREE))],
        candidates / FORMAL_CANDIDATE_BODY_NAME)
    state_bank = root / "raw" / "state_bank" / "shopping_key_states.jsonl"
    state_bank.parent.mkdir(parents=True, exist_ok=True)
    state_bank.write_text("".join(json.dumps({
        "state_id": f"state-{index}", "axtree_snapshot": AXTREE}) + "\n"
        for index in (1, 2, 3)))
    for side, item in zip(("train", "dev", "test"), rows):
        (split_dir / f"sft_{side}.jsonl").write_text(json.dumps(item) + "\n")
    for name in ("sft_train_multiturn.jsonl", "sft_dev_multiturn.jsonl",
                 "sft_test_multiturn.jsonl", "dpo_train_multiturn.jsonl",
                 "dpo_dev_multiturn.jsonl", "dpo_dev.jsonl"):
        (split_dir / name).write_text("")
    (split_dir / "SPLIT_REPORT.json").write_text(json.dumps({
        "formal": True, "available": True, "n_train": 1,
        "n_dev": 1, "n_test": 1}) + "\n")
    return row, fp


def _export(root, monkeypatch, name="formal"):
    out = root / "exports"
    monkeypatch.setattr("revact.server.export.EXPORTS_DIR", out)
    report = export_dataset(
        DataStore(root), {"name": name, "formal": True, "val_frac": 0.0,
                          "prefer_distilled": True})
    directories = sorted(out.iterdir()) if out.exists() else []
    directory = directories[-1] if directories else None
    excluded = ([json.loads(line) for line in
                 (directory / "excluded.jsonl").open() if line.strip()]
                if directory else report.get("excluded", []))
    return report, directory, excluded


def test_exact_point_and_prompt_provenance_is_exported(root, monkeypatch):
    row, _ = _seed(root)
    formal_grounding = DataStore(root).formal_grounding()
    assert formal_grounding["ok"] and formal_grounding["one_to_one"]
    assert formal_grounding["n_points"] == formal_grounding["n_manifest"] == 3
    assert formal_grounding["items"][0]["asset_tier"] == "formal_point"
    lineage = DataStore(root).lineage(row["sample_id"])
    assert lineage["candidate"]["candidate_id"] == "candidate-1"
    assert lineage["transition"]["post_observation_hash"] == "post-hash"
    assert lineage["split"] == "train"
    pair = {
        "pair_id": row["sample_id"] + "__legal_error",
        "prompt": row["messages"][:-1],
        "chosen": ASSISTANT,
        "rejected": ASSISTANT.replace("EXECUTE", "AVOID").replace(
            "<answer> click('3')", "<answer> click('4')"),
        "meta": dict(row["meta"], pair_type="over_block",
                     negative_source="legal_candidate",
                     negative_candidate_id="negative-candidate-1",
                     legal_at_snapshot=True,
                     negative_candidate_snapshot_hash=snapshot_sha256(AXTREE)),
    }
    (root / "train" / "formal" / "splits" / "dpo_train.jsonl").write_text(
        json.dumps(pair) + "\n")

    report, directory, excluded = _export(root, monkeypatch)
    assert report["n_train"] == 1 and report["n_dpo"] == 1
    assert excluded == []
    assert json.loads((directory / "provenance.json").read_text())[
        "formal_grounding_points"] == 3
    assert len(list((directory / "prompt_bundles").glob("*.json"))) >= 1
    assert len(list((directory / "prompt_generation_bundles").glob("*.json"))) >= 1
    assert report["missing_prompt_generation_bundles"] == []
    assert report["formal_dpo_source_gate"]["passes"] is True
    stats = json.loads((directory / "stats.json").read_text())
    assert stats["scope"] == "exact_export_rows"
    assert stats["volumes"] == {
        "sft_samples": 3, "dpo_pairs": 1, "distilled_samples": 0}


def test_formal_export_rejects_incomplete_split_before_writing(root, monkeypatch):
    _seed(root)
    (root / "train" / "formal" / "splits" / "sft_dev.jsonl").write_text("")
    report, directory, _ = _export(root, monkeypatch, "missing-dev")
    assert report["ok"] is False and directory is None
    assert "formal_dev_split_empty" in report["split_gate"]["errors"]


def test_completion_and_candidate_manifest_tampering_fail_closed(root):
    row, _ = _seed(root)
    tampered = deepcopy(row)
    tampered["messages"][-1]["content"] = ASSISTANT.replace(
        "<reversibility> RECOVERED", "<reversibility> UNKNOWN")
    problems = formal_point_join_problems([tampered], root)
    assert any("completion_recovery_status_mismatch" in item
               for item in problems)

    manifest = (root / "raw" / "candidates" /
                FORMAL_CANDIDATE_MANIFEST_NAME)
    manifest.unlink()
    problems = formal_point_join_problems([row], root)
    assert any("formal_candidate_artifact_invalid" in item
               for item in problems)


def test_legacy_and_tampered_provenance_fail_closed(root, monkeypatch):
    row, fp = _seed(root)
    bad_rows = [
        _sft(fp, "legacy__request", formal_dataset=False),
        _sft(fp, "unknown-point__request", probe_point_id="point-404",
             undo_source_probe_point_id="point-404"),
        _sft(fp, "wrong-state__request", state_id="state-404"),
        _sft(fp, "template-prediction__request",
             prediction_source="action_meta_template_legacy"),
        _sft(fp, "manual-undo__request", undo_source="manual_hint"),
        _sft(fp, "mock-missing__request", is_mock=None),
        _sft(fp, "mock-explicit__request", is_mock=True),
        _sft("0" * 12, "lost-prompt__request"),
    ]
    path = root / "train" / "formal" / "splits" / "sft_train.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in bad_rows) + "\n")
    dev = json.loads((path.parent / "sft_dev.jsonl").read_text())
    test = json.loads((path.parent / "sft_test.jsonl").read_text())
    (root / "train" / "formal" / config.FORMAL_SFT_PATH.name).write_text(
        "".join(json.dumps(r) + "\n" for r in [*bad_rows, dev, test]))
    report_path = path.parent / "SPLIT_REPORT.json"
    split_report = json.loads(report_path.read_text())
    split_report["n_train"] = len(bad_rows)
    report_path.write_text(json.dumps(split_report) + "\n")

    report, _, excluded = _export(root, monkeypatch)
    assert report["n_train"] == 0 and report["n_excluded"] == len(bad_rows)
    all_reasons = {reason for item in excluded
                   for reason in item.get("reasons", [item["reason"]])}
    assert {
        "formal_dataset_not_true",
        "unknown_probe_point_id",
        "prediction_source_not_probe_transition",
        "undo_source_not_probe_point_id",
        "is_mock_not_explicitly_false",
        "prompt_bundle_missing_or_invalid",
    } <= all_reasons
    assert any(reason.startswith("probe_provenance_mismatch:state_id")
               for reason in all_reasons)
    # Source files are audit assets and must remain byte-for-byte present.
    assert len(path.read_text().splitlines()) == len(bad_rows)
    mock_item = next(item for item in excluded
                     if item["id"] == "mock-explicit__request")
    assert "is_mock_not_explicitly_false" in mock_item.get(
        "reasons", [mock_item["reason"]])
    assert row["sample_id"] != bad_rows[0]["sample_id"]


def test_teacher_and_dpo_cannot_rebind_or_change_source_input(root, monkeypatch):
    row, _ = _seed(root)
    distilled = deepcopy(row)
    distilled["messages"][0]["content"] = "different system"
    (root / "train" / "formal" /
     config.FORMAL_DISTILLED_SFT_PATH.name).write_text(
        json.dumps(distilled) + "\n")
    pair = {
        "pair_id": row["sample_id"] + "__wrong_prompt",
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": "different input"}],
        "chosen": ASSISTANT,
        "rejected": ASSISTANT.replace("EXECUTE", "AVOID").replace(
            "<answer> click('3')", "<answer> click('4')"),
        "meta": dict(row["meta"], pair_type="over_block",
                     negative_source="legal_candidate",
                     negative_candidate_id="negative-candidate-1",
                     negative_candidate_snapshot_hash=snapshot_sha256(AXTREE),
                     legal_at_snapshot=True),
    }
    (root / "train" / "formal" / "splits" / "dpo_train.jsonl").write_text(
        json.dumps(pair) + "\n")

    report, directory, excluded = _export(root, monkeypatch)
    assert report["ok"] is False and directory is None
    reasons = {reason for item in excluded
               for reason in item.get("reasons", [item["reason"]])}
    assert "prompt_bundle_system_mismatch" in reasons

    (root / "train" / "formal" /
     config.FORMAL_DISTILLED_SFT_PATH.name).unlink()
    report, _, excluded = _export(root, monkeypatch, "dpo-rebind")
    assert report["ok"] is True and report["n_dpo"] == 0
    reasons = {reason for item in excluded
               for reason in item.get("reasons", [item["reason"]])}
    assert "derived_input_messages_mismatch" in reasons


def test_manifest_corruption_invalidates_entire_formal_source(root, monkeypatch):
    _seed(root)
    manifest = root / "grounded" / "POINT_MANIFEST.jsonl"
    payloads = [json.loads(line) for line in manifest.read_text().splitlines()]
    payloads[0]["record_sha256"] = "0" * 64
    manifest.write_text("".join(json.dumps(item) + "\n" for item in payloads))

    report, directory, excluded = _export(root, monkeypatch)
    assert report["n_train"] == 0
    assert "formal_grounding_artifact_invalid" in excluded[0].get(
        "reasons", [excluded[0]["reason"]])
    assert directory is None
    assert report["formal_grounding_error"]


def test_trainer_point_join_rejects_decorative_or_tampered_ids(root):
    row, _ = _seed(root)
    assert formal_point_join_problems([row], root) == []
    forged = deepcopy(row)
    forged["meta"]["state_id"] = "state-forged"
    problems = formal_point_join_problems([forged], root)
    assert any("probe_provenance_mismatch:state_id" in item
               for item in problems)


def test_formal_dpo_source_ratio_failure_quarantines_whole_family(
        root, monkeypatch):
    row, _ = _seed(root)
    synthetic = {
        "pair_id": row["sample_id"] + "__synthetic",
        "prompt": row["messages"][:-1],
        "chosen": ASSISTANT,
        "rejected": ASSISTANT.replace("EXECUTE", "AVOID"),
        "meta": dict(row["meta"], pair_type="over_block",
                     negative_source="synthetic_flip"),
    }
    (root / "train" / "formal" / "splits" / "dpo_train.jsonl").write_text(
        json.dumps(synthetic) + "\n")

    report, directory, excluded = _export(root, monkeypatch, "dpo-gate")
    gate = report["formal_dpo_source_gate"]
    assert gate["passes"] is False
    assert gate["legal_or_on_policy_share"] == 0.0
    assert report["ok"] is False and directory is None
    assert any(item["id"] == synthetic["pair_id"] and
               item["reason"].startswith(
                   "formal_dpo_legal_or_on_policy_share_below_0.500")
               for item in excluded)


def test_formal_dpo_source_ratio_is_release_wide(root, monkeypatch):
    row, _ = _seed(root)
    synthetic = {
        "pair_id": row["sample_id"] + "__synthetic",
        "prompt": row["messages"][:-1], "chosen": ASSISTANT,
        "rejected": ASSISTANT.replace("EXECUTE", "AVOID"),
        "meta": dict(row["meta"], pair_type="over_block",
                     negative_source="synthetic_flip"),
    }
    legal = deepcopy(synthetic)
    legal["pair_id"] = row["sample_id"] + "__legal"
    legal["rejected"] = ASSISTANT.replace("EXECUTE", "AVOID").replace(
        "<answer> click('3')", "<answer> click('4')")
    legal["meta"].update({
        "negative_source": "legal_candidate",
        "negative_candidate_id": "negative-candidate-1",
        "legal_at_snapshot": True,
        "negative_candidate_snapshot_hash": snapshot_sha256(AXTREE),
    })
    split = root / "train" / "formal" / "splits"
    (split / "dpo_train.jsonl").write_text(json.dumps(synthetic) + "\n")
    (split / "dpo_train_multiturn.jsonl").write_text(json.dumps(legal) + "\n")

    report, _, excluded = _export(root, monkeypatch, "dpo-wide-gate")
    gate = report["formal_dpo_source_gate"]
    assert gate["passes"] is True
    assert gate["n_pairs"] == 2
    assert gate["legal_or_on_policy_share"] == .5
    assert report["n_dpo"] == report["n_multiturn_dpo"] == 1
    assert excluded == []


def test_formal_export_includes_valid_single_and_multiturn_families(
        root, monkeypatch):
    _seed(root)
    formal_dir = root / "train" / "formal"
    split_dir = formal_dir / "splits"
    multiturn_rows = []
    for side in ("train", "dev", "test"):
        single = json.loads((split_dir / f"sft_{side}.jsonl").read_text())
        multi = deepcopy(single)
        multi["sample_id"] = single["sample_id"] + "__multiturn"
        multi["meta"]["kind"] = "multiturn"
        multiturn_rows.append(multi)
        (split_dir / f"sft_{side}_multiturn.jsonl").write_text(
            json.dumps(multi) + "\n")
    (formal_dir / config.FORMAL_MULTITURN_SFT_PATH.name).write_text(
        "".join(json.dumps(row) + "\n" for row in multiturn_rows))

    split_report_path = split_dir / "SPLIT_REPORT.json"
    split_report = json.loads(split_report_path.read_text())
    split_report.update({"n_train": 2, "n_dev": 2, "n_test": 2})
    split_report_path.write_text(json.dumps(split_report) + "\n")

    report, directory, excluded = _export(
        root, monkeypatch, "single-and-multiturn")
    assert report["ok"] is True
    assert report["n_train"] == report["n_val"] == report["n_test"] == 1
    assert report["n_multiturn_train"] == 1
    assert report["n_multiturn_val"] == 1
    assert report["n_multiturn_test"] == 1
    assert excluded == []
    for name in (
            "sft_train_multiturn.jsonl", "sft_val_multiturn.jsonl",
            "sft_test_multiturn.jsonl"):
        rows = [line for line in (directory / name).read_text().splitlines()
                if line.strip()]
        assert len(rows) == 1
    stats = json.loads((directory / "stats.json").read_text())
    assert stats["volumes"]["sft_samples"] == 6


def test_dataset_export_dry_run_executes_gates_without_writes(root, monkeypatch):
    _seed(root)
    output_root = root / "would-be-exports"
    monkeypatch.setattr("revact.server.export.EXPORTS_DIR", output_root)
    report = export_dataset(DataStore(root), {
        "name": "dry-run", "formal": True, "dry_run": True,
        "prefer_distilled": False,
    })
    assert report["ok"] is True and report["dry_run"] is True
    assert report["output_dir"] is None
    assert report["n_train"] == report["n_val"] == report["n_test"] == 1
    assert "stats.json" in report["would_write"]
    assert not output_root.exists()
