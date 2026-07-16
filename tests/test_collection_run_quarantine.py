"""Collection-run quarantine is exact, reversible, and fail-closed."""
from __future__ import annotations

import hashlib
import json

import pytest

from scripts.quarantine_collection_run import (
    CollectionRunQuarantineError, quarantine_collection_run,
    rollback_collection_run)


def _write_fixture(root, run_id="run-1"):
    data = root / "data"
    manifest = data / "manifests" / "collection_runs" / f"{run_id}.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({
        "schema_version": "iris.collection-run.v2", "run_id": run_id,
        "status": "COMPLETE", "trajectories": [],
    }) + "\n")
    raw = data / "raw" / "trajectories" / f"task__run_{run_id}.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_text(json.dumps({"run_id": run_id}) + "\n")
    meta = data / "raw" / "trajectories_meta.jsonl"
    meta.write_text("".join(json.dumps(row) + "\n" for row in [
        {"run_id": "keep", "trajectory_id": "keep"},
        {"run_id": run_id, "trajectory_id": "task__run_run-1",
         "raw_artifact": f"raw/trajectories/{raw.name}"},
    ]))
    state = data / "raw" / "state_bank" / "shopping_key_states.jsonl"
    state.parent.mkdir(parents=True)
    state.write_text("".join(json.dumps(row) + "\n" for row in [
        {"run_id": run_id, "state_id": "remove"},
        {"run_id": "keep", "state_id": "keep"},
    ]))
    return manifest, raw, meta, state


def test_collection_run_quarantine_dry_run_writes_nothing(tmp_path):
    manifest, raw, meta, state = _write_fixture(tmp_path)
    before = {path: path.read_bytes() for path in (manifest, raw, meta, state)}
    result = quarantine_collection_run(
        tmp_path, "run-1", "bad provenance", execute=False)
    assert result["execute"] is False
    assert sum(item["removed_rows"] for item in result["shared_files"]) == 2
    assert all(path.read_bytes() == body for path, body in before.items())
    assert not (tmp_path / "data" / "raw" / "quarantine").exists()


def test_collection_run_quarantine_and_rollback_are_byte_exact(tmp_path):
    manifest, raw, meta, state = _write_fixture(tmp_path)
    before = {path: path.read_bytes() for path in (manifest, raw, meta, state)}
    result = quarantine_collection_run(
        tmp_path, "run-1", "bad provenance", execute=True)
    assert result["execute"] is True
    assert not manifest.exists() and not raw.exists()
    assert "run-1" not in meta.read_text()
    assert "remove" not in state.read_text()

    rollback = rollback_collection_run(tmp_path, "run-1")
    assert rollback["rolled_back"] is True
    assert all(path.read_bytes() == body for path, body in before.items())


def test_collection_run_rollback_refuses_post_migration_edits(tmp_path):
    _manifest, _raw, meta, _state = _write_fixture(tmp_path)
    quarantine_collection_run(
        tmp_path, "run-1", "bad provenance", execute=True)
    meta.write_text(meta.read_text() + json.dumps({"run_id": "new"}) + "\n")
    with pytest.raises(CollectionRunQuarantineError,
                       match="active shared file changed"):
        rollback_collection_run(tmp_path, "run-1")


def test_quarantine_report_hashes_match_artifacts(tmp_path):
    _write_fixture(tmp_path)
    result = quarantine_collection_run(
        tmp_path, "run-1", "bad provenance", execute=True)
    report = json.loads((tmp_path / result["report_path"]).read_text())
    for move in report["moved_artifacts"]:
        target = tmp_path / move["quarantine_path"]
        assert hashlib.sha256(target.read_bytes()).hexdigest() == move["sha256"]
