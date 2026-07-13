#!/usr/bin/env python3
"""Read-only integrity audit for the IRIS remediation baseline.

The script deliberately uses only the standard library.  It never mutates the
dataset and emits one deterministic JSON document that can be pasted into an
audit report or compared after a migration.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any


CLICK_RE = re.compile(r"click\(['\"]([^'\"]+)['\"]")
HISTORY_RE = re.compile(
    r"<history>\s*\n(.*?)\n\s*\n<observation>", re.DOTALL
)


def jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def count(values) -> dict[str, int]:
    return dict(sorted(collections.Counter(str(v) for v in values).items()))


def history_len(row: dict[str, Any]) -> int:
    messages = row.get("messages") or []
    if len(messages) < 2:
        return 0
    match = HISTORY_RE.search(messages[1].get("content", ""))
    if not match or match.group(1).strip() == "(none)":
        return 0
    return len([line for line in match.group(1).splitlines() if line.strip()])


def decision_observation(row: dict[str, Any]) -> str:
    users = [m.get("content", "") for m in row.get("messages", [])
             if m.get("role") == "user"]
    return users[-1] if users else ""


def gold_bid_visible(row: dict[str, Any]) -> bool:
    raw_action = (row.get("meta") or {}).get("risky_raw_action", "")
    match = CLICK_RE.search(raw_action)
    return bool(match and f"[{match.group(1)}]" in decision_observation(row))


def state_group(sample_id: str) -> str:
    for suffix in ("__constraint", "__request"):
        if sample_id.endswith(suffix):
            return sample_id[:-len(suffix)]
    return sample_id.rsplit("__", 1)[0]


def split_overlap(root: Path) -> dict[str, Any]:
    split_dir = root / "data" / "train" / "splits"
    rows = {
        name: jsonl(split_dir / f"sft_{name}.jsonl")
        for name in ("train", "test")
    }
    groups = {
        name: {state_group(r.get("sample_id", "")) for r in values}
        for name, values in rows.items()
    }
    overlap = sorted(groups["train"] & groups["test"])
    return {
        "train_rows": len(rows["train"]),
        "test_rows": len(rows["test"]),
        "state_group_overlap_count": len(overlap),
        "state_group_overlap_examples": overlap[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()

    paths = {
        "key_states": root / "data/raw/state_bank/shopping_key_states.jsonl",
        "grounded": root / "data/grounded/reversibility.jsonl",
        "manifest": root / "data/grounded/MANIFEST.jsonl",
        "sft": root / "data/train/sft/revact_sft.jsonl",
        "sft_multiturn": root / "data/train/sft/revact_sft_multiturn.jsonl",
        "sft_distilled": root / "data/train/sft/revact_sft_distilled.jsonl",
        "dpo": root / "data/train/dpo/revact_dpo.jsonl",
        "dpo_multiturn": root / "data/train/dpo/revact_dpo_multiturn.jsonl",
        "trajectory_meta": root / "data/raw/trajectories_meta.jsonl",
    }
    data = {name: jsonl(path) for name, path in paths.items()}
    ks = data["key_states"]
    grounded = data["grounded"]
    manifest = data["manifest"]
    sft = data["sft"]
    multi = data["sft_multiturn"]
    meta = data["trajectory_meta"]

    manifest_ids = {r.get("probe_id") for r in manifest if r.get("probe_id")}
    grounded_ids = {r.get("probe_id") for r in grounded if r.get("probe_id")}
    multi_visible = [gold_bid_visible(r) for r in multi]

    report = {
        "root": str(root),
        "artifact_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in paths.items() if path.exists()
        },
        "line_counts": {name: len(rows) for name, rows in data.items()},
        "key_states": {
            "unique_state_ids": len({r.get("state_id") for r in ks}),
            "unique_trajectory_ids": len({r.get("trajectory_id") for r in ks}),
            "afforded_action_types": count(
                action for row in ks for action in row.get("afforded_action_types", [])
            ),
            "trajectory_success": count(r.get("traj_success") for r in ks),
        },
        "grounding": {
            "labels": count(r.get("label") for r in grounded),
            "sites": count(r.get("site") for r in grounded),
            "missing_state_id": sum(not r.get("state_id") for r in grounded),
            "missing_probe_id": sum(not r.get("probe_id") for r in grounded),
            "manifest_commit_mode": count(r.get("commit_mode") for r in manifest),
            "grounded_without_manifest": sorted(grounded_ids - manifest_ids),
            "manifest_without_grounded": sorted(manifest_ids - grounded_ids),
            "legacy_row_numbers": [
                index for index, row in enumerate(grounded, 1)
                if not row.get("probe_id")
            ],
        },
        "single_sft": {
            "sites": count((r.get("meta") or {}).get("site") for r in sft),
            "actions": count((r.get("meta") or {}).get("action_type") for r in sft),
            "labels": count((r.get("meta") or {}).get("reversibility") for r in sft),
            "decisions": count((r.get("meta") or {}).get("decision") for r in sft),
            "history_sources": count((r.get("meta") or {}).get("history_source") for r in sft),
            "history_line_counts": count(history_len(r) for r in sft),
        },
        "multiturn_sft": {
            "mock_rows": sum("mock." in (r.get("meta") or {}).get("trajectory_id", "")
                             for r in multi),
            "webarena_rows": sum("webarena." in (r.get("meta") or {}).get("trajectory_id", "")
                                 for r in multi),
            "actions": count((r.get("meta") or {}).get("action_type") for r in multi),
            "decision_steps": count((r.get("meta") or {}).get("decision_step") for r in multi),
            "history_line_counts": count(history_len(r) for r in multi),
            "gold_bid_visible": sum(multi_visible),
            "gold_bid_missing": len(multi_visible) - sum(multi_visible),
        },
        "trajectories": {
            "meta_rows": len(meta),
            "unique_trajectory_ids": len({r.get("trajectory_id") for r in meta}),
            "success_rows": sum(bool(r.get("success")) for r in meta),
            "unique_ids_with_any_success": len({r.get("trajectory_id") for r in meta
                                                if r.get("success")}),
            "duplicate_id_counts": count(
                collections.Counter(r.get("trajectory_id") for r in meta).values()
            ),
        },
        "splits": split_overlap(root),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
