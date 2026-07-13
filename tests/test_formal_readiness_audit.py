"""Regression tests for the non-vacuous publication-readiness audit."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from revact.data.candidates import (CANDIDATE_SCHEMA_VERSION, Candidate,
                                    save_candidate_set, snapshot_sha256)
from revact.prompt_store import store_bundle


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_formal_readiness.py"
SPEC = importlib.util.spec_from_file_location("audit_formal_readiness", SCRIPT)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _run(root: Path, *extra: str) -> tuple[subprocess.CompletedProcess, dict]:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--data-root", str(root), *extra],
        cwd=ROOT, text=True, capture_output=True, check=False)
    return result, json.loads(result.stdout)


def test_empty_formal_assets_fail_non_vacuous_gate(tmp_path):
    result, report = _run(tmp_path)
    assert result.returncode == 1
    assert report["ready"] is False
    assert report["non_vacuous"] is False
    assert report["non_vacuous_gates"] == {
        "evaluation_truth_records": False,
        "formal_dpo": False,
        "formal_dpo_train_split": False,
        "formal_dev_split": False,
        "formal_sft": False,
        "formal_test_split": False,
        "formal_train_split": False,
        "grounding_points": False,
    }
    assert report["grounding"]["n_points"] == 0
    assert report["teacher"]["coverage"] is None

    allowed, same_report = _run(tmp_path, "--allow-blocked")
    assert allowed.returncode == 0
    assert same_report["ready"] is False
    assert same_report["non_vacuous"] is False


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _formal_collection_fixture(root: Path, *, origin: str = "webarena") -> dict:
    run_id, trajectory_id, state_id = "run-1", "traj-1", "state-1"
    _write_jsonl(root / "raw" / "trajectories_meta.jsonl", [
        # Frozen legacy dirt remains visible to the historical diagnostic but
        # must not permanently block a clean, exact formal run closure.
        {"trajectory_id": "legacy-duplicate"},
        {"trajectory_id": "legacy-duplicate"},
        {
            "run_id": run_id, "trajectory_id": trajectory_id,
            "environment_origin": origin, "success": True,
            "collector_success": True,
        },
    ])
    _write_jsonl(root / "raw" / "trajectories" / f"{trajectory_id}.jsonl", [
        {"run_id": run_id, "trajectory_id": trajectory_id, "step_id": 0},
    ])
    _write_jsonl(root / "raw" / "state_bank" / "shopping_key_states.jsonl", [
        {
            "run_id": run_id, "trajectory_id": trajectory_id,
            "state_id": state_id, "environment_origin": origin,
            "collector_success": True, "traj_success": True,
        },
    ])
    manifest = root / "manifests" / "collection_runs" / f"{run_id}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "schema_version": "iris.collection-run.v2",
        "run_id": run_id,
        "status": "COMPLETE",
        "trajectories": [{
            "run_id": run_id, "trajectory_id": trajectory_id,
            "environment_origin": origin, "success": True,
            "collector_success": True,
        }],
    }))
    return {
        "sample_id": "formal-sample",
        "meta": {
            "formal_dataset": True,
            "run_id": run_id,
            "trajectory_id": trajectory_id,
            "state_id": state_id,
            "environment_origin": origin,
            "collector_success": True,
        },
    }


def test_formal_lineage_uses_exact_run_closure_not_frozen_legacy_inventory(
        tmp_path):
    formal = _formal_collection_fixture(tmp_path)
    scoped = AUDIT.audit_formal_collection_lineage(tmp_path, [formal])
    historical = AUDIT.audit_collection_lineage(tmp_path)

    assert scoped["ok"] is True
    assert scoped["legacy_quarantine_excluded_from_gate"] is True
    assert scoped["n_formal_rows"] == 1
    assert historical["ok"] is False
    assert historical["duplicate_meta_trajectory_ids"] == ["legacy-duplicate"]
    assert len(historical["unknown_environment_origin_rows"]) == 2


def test_formal_lineage_rejects_unknown_environment_origin(tmp_path):
    formal = _formal_collection_fixture(tmp_path, origin="unknown")
    report = AUDIT.audit_formal_collection_lineage(tmp_path, [formal])
    assert report["ok"] is False
    assert report["issues"]["unknown_environment_origin"] == [
        "formal-sample",
        "run-1/traj-1/state-1:manifest",
        "run-1/traj-1/state-1:meta",
        "run-1/traj-1/state-1:state",
    ]


def test_dpo_deployment_source_share_is_non_vacuous_and_release_wide():
    assert AUDIT.deployment_source_share([])["passed"] is False
    half = AUDIT.deployment_source_share([
        {"meta": {"negative_source": "synthetic_flip"}},
        {"meta": {"negative_source": "legal_candidate"}},
    ])
    assert half["legal_or_on_policy_share"] == .5
    assert half["passed"] is True
    synthetic = AUDIT.deployment_source_share([
        {"meta": {"negative_source": "synthetic_flip"}},
    ])
    assert synthetic["legal_or_on_policy_share"] == 0.0
    assert synthetic["passed"] is False


def test_evaluation_truth_body_requires_matching_manifest(tmp_path):
    _write_jsonl(tmp_path / "eval" / "truth.jsonl", [{"incomplete": True}])
    context = AUDIT.formal_release_context(tmp_path)
    report = AUDIT.evaluation_truth_audit(tmp_path, context)
    assert report["n_records"] == 1
    assert report["integrity"] is False
    assert "must either both exist or both be absent" in report["error"]


def test_formal_dpo_gate_requires_exact_train_split_and_deployment_source(
        tmp_path, monkeypatch):
    source = {
        "sample_id": "sample-1",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": "chosen"},
        ],
        "meta": {"formal_dataset": True},
    }
    pair = {
        "pair_id": "sample-1__error-1",
        "prompt": source["messages"][:-1],
        "chosen": "chosen",
        "rejected": "rejected",
        "meta": {
            "formal_dataset": True,
            "source_sample_id": "sample-1",
            "negative_source": "legal_candidate",
        },
    }
    _write_jsonl(
        tmp_path / "train" / "formal" / "iris_dpo_point_v1.jsonl",
        [pair])
    monkeypatch.setattr(AUDIT, "validate_dpo_rows", lambda *args, **kwargs: {
        "n_rows": 1, "n_problems": 0, "problems": [],
        "negative_source_dist": {"legal_candidate": 1},
        "deployment_negative_ratio": 1.0,
    })
    monkeypatch.setattr(AUDIT, "_release_failures", lambda *args: {})
    monkeypatch.setattr(AUDIT, "formal_derivation_reasons", lambda *args: [])

    missing = AUDIT.formal_dpo_audit(
        tmp_path, [source], [source], [], object())
    assert missing["n_body_pairs"] == 1
    assert missing["n_train_pairs"] == 0
    assert missing["split_integrity"] is False
    assert missing["issues"]["split_missing_expected_pair"] == [
        "train:sample-1__error-1"]
    assert missing["source_share_passed"] is False

    _write_jsonl(
        tmp_path / "train" / "formal" / "splits" / "dpo_train.jsonl",
        [pair])
    complete = AUDIT.formal_dpo_audit(
        tmp_path, [source], [source], [], object())
    assert complete["issues"] == {}
    assert complete["integrity"] is True
    assert complete["split_integrity"] is True
    assert complete["source_share_passed"] is True


def test_bid_audit_checks_pinned_risky_action_not_only_answer():
    row = {
        "sample_id": "sample-1",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": (
                "<goal>g</goal>\n<history></history>\n"
                "<observation>\n[3] button 'safe'</observation>")},
            {"role": "assistant", "content": "<answer> click('3')"},
        ],
        "meta": {"risky_raw_action": "click('9')"},
    }
    report = AUDIT.bid_visibility([row])
    assert report["assistant_answers"]["passed"] is True
    assert report["assistant_answers"]["visibility_rate"] == 1.0
    assert report["risky_raw_actions"]["passed"] is False
    assert report["risky_raw_actions"]["missing_ids"] == [
        "sample-1:risky_raw_action:bid-9"]
    assert report["passed"] is False


def test_candidate_audit_requires_manifest_and_replays_snapshot_legality(tmp_path):
    axtree = "[7] button 'Do it'"
    state_dir = tmp_path / "raw" / "state_bank"
    state_dir.mkdir(parents=True)
    (state_dir / "fixture.jsonl").write_text(json.dumps({
        "state_id": "state-1", "axtree_snapshot": axtree,
    }) + "\n")
    path = tmp_path / "raw" / "candidates" / "iris_candidates.v3.jsonl"
    save_candidate_set([Candidate(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        candidate_id="candidate-1", state_id="state-1", bid="7",
        canonical_action="click:button:do-it", category="expert_action",
        source="expert", legal_at_snapshot=True,
        proposer_model="fixture", proposer_version="v1",
        snapshot_hash=snapshot_sha256(axtree),
    )], path)

    report = AUDIT.candidate_audit(tmp_path)
    assert report["integrity"] is True
    assert report["n_snapshot_matched_and_legal"] == 1

    path.with_name("CANDIDATE_MANIFEST.jsonl").unlink()
    broken = AUDIT.candidate_audit(tmp_path)
    assert broken["integrity"] is False
    assert "must both exist" in broken["error"]


def test_prompt_content_gate_and_canonical_distilled_path_are_audited(tmp_path):
    bundle = store_bundle(
        {"agent_system": "expected immutable system"},
        root=tmp_path, author="readiness-test")
    row = {
        "sample_id": "sample-1",
        "messages": [
            {"role": "system", "content": "tampered system"},
            {"role": "user", "content": "<observation>\n[3] button 'do'"},
            {"role": "assistant", "content": "<answer> click('3')"},
        ],
        "meta": {
            "formal_dataset": True,
            "prompts_fp": bundle.stem,
            "risky_raw_action": "click('3')",
        },
    }
    formal = tmp_path / "train" / "formal"
    formal.mkdir(parents=True)
    encoded = json.dumps(row) + "\n"
    (formal / "iris_sft_point_v1.jsonl").write_text(encoded)
    # A stale legacy filename must not be counted as formal teacher coverage.
    (formal / "revact_sft_distilled.jsonl").write_text(encoded)

    _, no_teacher = _run(tmp_path, "--allow-blocked")
    failures = no_teacher["formal_training"]["release_gate_failures"]
    assert "prompt_bundle_system_mismatch" in failures["sample-1"]
    content_failures = no_teacher["formal_training"]["prompt_content_failures"]
    assert "prompt_bundle_system_mismatch" in content_failures["sample-1"]
    assert any(reason.startswith("completion_")
               for reason in content_failures["sample-1"])
    assert no_teacher["teacher"]["n_rows"] == 0
    assert no_teacher["teacher"]["coverage"] == 0.0

    canonical = formal / "iris_sft_distilled_point_v1.jsonl"
    canonical.write_text(encoded)
    _, with_teacher = _run(tmp_path, "--allow-blocked")
    assert with_teacher["teacher"]["path"] == str(canonical)
    assert with_teacher["teacher"]["n_rows"] == 1
    assert with_teacher["teacher"]["coverage"] == 1.0
    assert "prompt_bundle_system_mismatch" in with_teacher["teacher"][
        "release_or_prompt_failures"]["sample-1"]
