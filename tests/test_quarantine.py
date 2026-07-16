import hashlib
import json
import subprocess
import sys
from pathlib import Path

from revact.data.quarantine import build_quarantine_index, write_quarantine_index


def test_quarantine_is_index_only_and_flags_mock(tmp_path):
    data = tmp_path / "data"
    source = data / "train" / "sft" / "revact_sft_multiturn.jsonl"
    source.parent.mkdir(parents=True)
    row = {"sample_id": "legacy-1", "messages": [], "meta": {
        "trajectory_id": "mock.example_seed0", "history_source": "trajectory",
    }}
    source.write_text(json.dumps(row) + "\n")
    before = hashlib.sha256(source.read_bytes()).hexdigest()

    report = write_quarantine_index(data)
    assert report["n_quarantined"] == 1
    assert report["n_formal_eligible"] == 0
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    item = json.loads((data / "train" / "quarantine" /
                       "legacy_training_manifest.jsonl").read_text())
    assert {"mock_environment", "formal_dataset_not_true",
            "missing_probe_point_id"} <= set(item["reasons"])
    assert item["record_sha256"]


def test_current_formal_shape_can_pass_index(tmp_path):
    data = tmp_path / "data"
    source = data / "train" / "sft" / "revact_sft.jsonl"
    source.parent.mkdir(parents=True)
    row = {"sample_id": "formal-1", "messages": [], "meta": {
        "formal_dataset": True, "probe_point_id": "point-1",
        "state_id": "state-1", "action_instance_id": "action-1",
        "history_source": "trajectory", "collector_success": True,
        "prediction_source": "probe_transition", "undo_source": "probe_point_id",
    }}
    source.write_text(json.dumps(row) + "\n")
    report = build_quarantine_index(data, sources=("sft/revact_sft.jsonl",))
    assert report["n_formal_eligible"] == 1
    assert report["entries"][0]["reasons"] == []


def test_legacy_trajectory_quarantine_preserves_sources(tmp_path):
    data = tmp_path / "data"
    meta = data / "raw" / "trajectories_meta.jsonl"
    meta.parent.mkdir(parents=True)
    meta.write_text(
        "".join(json.dumps({
            "trajectory_id": "logical-1", "success": index == 0}) + "\n"
            for index in range(2)) +
        json.dumps({
            "trajectory_id": "logical-2__run_abc", "run_id": "abc",
            "environment_origin": "webarena", "success": True,
        }) + "\n")
    before = hashlib.sha256(meta.read_bytes()).hexdigest()
    script = Path(__file__).resolve().parents[1] / "scripts" / \
        "quarantine_legacy_lineage.py"
    result = subprocess.run(
        [sys.executable, str(script), "--data-root", str(data)],
        text=True, capture_output=True, check=False)
    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["n_meta_rows"] == 3
    assert report["n_legacy_rows"] == 2
    assert report["n_transactional_rows_excluded"] == 1
    assert report["n_unique_trajectory_ids"] == 2
    assert report["n_duplicate_trajectory_ids"] == 1
    assert hashlib.sha256(meta.read_bytes()).hexdigest() == before
    rows = [json.loads(line) for line in
            (data / "raw" / "quarantine" /
             "legacy_lineage_rows.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["formal_eligible"] is False for row in rows)
    assert all("duplicate_trajectory_id" in row["reason_codes"] for row in rows)
