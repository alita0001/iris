"""Prepare reviewed, label-free point probe batches from collected states.

This module closes the auditable bridge
``state -> legal S4 candidate -> authored spec -> execution spec``.  It does
not open a browser and cannot generate a label.  Promotion to an executable
contract is allowed only after exact snapshot fixture validation and an
explicit reviewer identity supplied by the caller.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from .. import config
from ..data.candidates import (CATEGORY_EXPERT, FORMAL_CANDIDATE_BODY_NAME,
                               Candidate,
                               assert_candidate_manifest_integrity,
                               build_a11y_candidate_set, interactive_elements,
                               save_candidate_set)
from ..envs.obs_utils import history_entry
from .schema import load_probe_points
from .transitions import (ProbeTransition, assert_point_transition_integrity,
                          load_probe_transitions)
from .authoring import (ProbeExecutionSpec, promote_authored_spec,
                        save_authored_spec, spec_from_workbench)


_SEMANTIC_UNDO: dict[str, tuple[tuple[str, ...], ...]] = {
    "add_to_cart": (
        ("navigate_to_signal_page", "remove_cart_line"),
        ("navigate_to_signal_page", "remove_cart_line", "confirm_modal_ok"),
    ),
    "wishlist_add": (
        ("navigate_to_signal_page", "remove_wishlist_item"),
        ("navigate_to_signal_page", "remove_wishlist_item", "confirm_modal_ok"),
    ),
    "compare_add": (
        ("navigate_to_signal_page", "remove_compare_item"),
        ("navigate_to_signal_page", "remove_compare_item", "confirm_modal_ok"),
    ),
    "reddit_vote": (
        ("navigate_to_signal_page", "toggle_reddit_upvote"),
    ),
    "reddit_subscribe": (
        ("navigate_to_signal_page", "toggle_forum_subscription"),
    ),
}
_EXECUTED_SOLVER = "site_specific_deterministic.live-1"
_ACTION_SITE = {
    "add_to_cart": "shopping", "wishlist_add": "shopping",
    "compare_add": "shopping", "reddit_vote": "reddit",
    "reddit_subscribe": "reddit",
}


class PointBatchPreparationError(ValueError):
    """A collected state cannot be promoted into a reviewed execution spec."""


def _digest(prefix: str, payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))
    return prefix + hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def _load_states(state_bank_dir: Path, wanted: set[str]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {state_id: [] for state_id in wanted}
    for path in sorted(Path(state_bank_dir).glob("*.jsonl")):
        # This module's output is a derived assembler view, never a source
        # collection row.  Excluding it makes rematerialization idempotent.
        if path.name == "formal_point_reached_states.jsonl":
            continue
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            state_id = str(row.get("state_id") or "")
            if state_id in grouped:
                grouped[state_id].append(row)
    missing = sorted(state_id for state_id, rows in grouped.items() if not rows)
    ambiguous = sorted(state_id for state_id, rows in grouped.items()
                       if len(rows) != 1)
    if missing or ambiguous:
        raise PointBatchPreparationError(
            f"state lookup failed: missing={missing}, ambiguous={ambiguous}")
    return {state_id: rows[0] for state_id, rows in grouped.items()}


def _write_execution_specs(specs: Iterable[ProbeExecutionSpec], path: Path) -> None:
    rows = [spec.to_dict() for spec in specs]
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise PointBatchPreparationError(
                f"refusing to overwrite immutable execution batch {path}")
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def prepare_point_probe_batch(
    *,
    data_root: Path,
    state_ids: list[str],
    probe_run_id: str,
    reviewer: str,
    timestamp: str,
    code_version: str,
    execution_path: Path,
    environment_instance: str | None = None,
) -> dict[str, Any]:
    """Materialize candidates and immutable reviewed specs for point states.

    ``fixture_status=PASSED`` means only that the recorded AXTree contains the
    exact interactive bid, its snapshot hash/canonical action round-trip, and a
    supported deterministic recovery adapter exists.  No behavioral label is
    inferred here.
    """
    if not state_ids or len(state_ids) != len(set(state_ids)):
        raise PointBatchPreparationError("state_ids must be non-empty and unique")
    for name, value in (("probe_run_id", probe_run_id), ("reviewer", reviewer),
                        ("timestamp", timestamp), ("code_version", code_version)):
        if not str(value or "").strip():
            raise PointBatchPreparationError(f"{name} is required")

    root = Path(data_root)
    states = _load_states(root / "raw" / "state_bank", set(state_ids))
    candidate_path = root / "raw" / "candidates" / "iris_candidates.v3.jsonl"
    authored_path = root / "grounded" / "specs" / "authored_specs.jsonl"
    promoted: list[ProbeExecutionSpec] = []
    selected: list[Candidate] = []
    for state_id in state_ids:
        state = states[state_id]
        action_types = list(state.get("afforded_action_types") or [])
        if len(action_types) != 1 or action_types[0] not in _SEMANTIC_UNDO:
            raise PointBatchPreparationError(
                f"{state_id}: expected one supported action type, got {action_types}")
        action_type = action_types[0]
        site = str(state.get("site") or "")
        if site != _ACTION_SITE.get(action_type):
            raise PointBatchPreparationError(
                f"{state_id}: action/site mismatch {action_type!r}/{site!r}")
        point_environment = environment_instance or config.site_base(site)
        if not point_environment:
            raise PointBatchPreparationError(
                f"{state_id}: environment_instance is required for site {site!r}")
        snapshot = str(state.get("axtree_snapshot") or "")
        expert_bid = str(state.get("affordance_bid") or "")
        candidates = build_a11y_candidate_set(
            state_id=state_id, axtree_txt=snapshot, expert_bid=expert_bid)
        save_candidate_set(candidates, candidate_path)
        candidate = next(row for row in candidates
                         if row.category == CATEGORY_EXPERT and row.bid == expert_bid)
        selected.append(candidate)

        action_instance_id = _digest("action-", {
            "state_id": state_id, "candidate_id": candidate.candidate_id})
        draft = spec_from_workbench({
            "name": f"{site}.{action_type}.{state_id}",
            "site": site,
            "action_type": action_type,
            "candidate_id": candidate.candidate_id,
            "state_id": state_id,
            "action_instance_id": action_instance_id,
            "raw_action": f"click('{candidate.bid}')",
            "canonical_action": candidate.canonical_action,
            "signal_channels": ["ui_structural"],
            "undo_sequences": [list(seq) for seq in _SEMANTIC_UNDO[action_type]],
            "solver_set": [_EXECUTED_SOLVER],
            "budget_k": 12,
            "safety_level": "self_recovering",
            "author": reviewer,
        }, timestamp=timestamp, controller_version=config.CONTROLLER_VERSION)
        save_authored_spec(draft, authored_path)

        probe_point_id = _digest("point-", {
            "probe_run_id": probe_run_id,
            "candidate_id": candidate.candidate_id,
            "action_instance_id": action_instance_id,
        })
        promoted.append(promote_authored_spec(draft, {
            "probe_point_id": probe_point_id,
            "probe_run_id": probe_run_id,
            "candidate_snapshot_hash": candidate.snapshot_hash,
            "environment_family": (
                config.site_environment_family(site) or site),
            "environment_instance": point_environment,
            "environment_origin": str(state.get("environment_origin") or "webarena"),
            "is_mock": False,
            "task_id": str(state.get("task_id") or ""),
            "trajectory_id": str(state.get("trajectory_id") or ""),
            "run_id": str(state.get("run_id") or ""),
            "seed": state.get("seed"),
            "url": str(state.get("url") or ""),
            "account": str(state.get("account") or "customer"),
            "privilege": str(state.get("privilege") or "customer"),
            "code_version": code_version,
            "product_url": (str(state.get("url") or "")
                            if site == "shopping" else ""),
            "submission_url": (str(state.get("url") or "")
                               if action_type == "reddit_vote" else ""),
            "forum_url": (str(state.get("url") or "")
                          if action_type == "reddit_subscribe" else ""),
        }, {
            "fixture_status": "PASSED",
            "code_review_status": "APPROVED",
            "reviewer": reviewer,
            "review_timestamp": timestamp,
        }))

    _write_execution_specs(promoted, Path(execution_path))
    return {
        "schema_version": "iris.point_batch_preparation.v1",
        "probe_run_id": probe_run_id,
        "state_ids": list(state_ids),
        "candidate_ids": [candidate.candidate_id for candidate in selected],
        "probe_point_ids": [spec.probe_point_id for spec in promoted],
        "candidate_artifact": str(candidate_path),
        "authored_spec_artifact": str(authored_path),
        "execution_spec_artifact": str(execution_path),
        "labels_created": 0,
    }


def _trajectory_history(root: Path, state: dict, *,
                        decision_snapshot: str | None = None
                        ) -> tuple[list[dict], str]:
    trajectory_id = str(state.get("trajectory_id") or "")
    path = root / "raw" / "trajectories" / f"{trajectory_id}.jsonl"
    if not path.exists():
        raise PointBatchPreparationError(
            f"missing raw trajectory for state {state.get('state_id')}: {path}")
    rows = [json.loads(line) for line in path.open(encoding="utf-8")
            if line.strip()]
    rows.sort(key=lambda row: int(row.get("step_id", -1)))
    target_step = int(state.get("step_id", -1))
    prefix = [row for row in rows if int(row.get("step_id", -1)) <= target_step]
    if not prefix or int(prefix[0].get("step_id", -1)) != 0:
        raise PointBatchPreparationError(
            f"trajectory {trajectory_id} has no step-0 observation")
    for row in prefix:
        if (str(row.get("trajectory_id") or "") != trajectory_id or
                str(row.get("run_id") or "") != str(state.get("run_id") or "")):
            raise PointBatchPreparationError(
                f"trajectory/run lineage mismatch for {state.get('state_id')}")

    state_snapshot = str(state.get("axtree_snapshot") or "")
    raw_snapshot = str(prefix[-1].get("obs_after_axtree") or "")
    alignment = "exact_raw_snapshot"
    if raw_snapshot != state_snapshot:
        state_hash = hashlib.sha256(state_snapshot.encode("utf-8")).hexdigest()
        actions = [str(row.get("action")) for row in prefix[1:]]
        target_bid = str(state.get("affordance_bid") or "")
        target_matches = [row for row in interactive_elements(state_snapshot)
                          if row["bid"] == target_bid]
        alternate_verified = (
            state.get("collector_success") is True and
            str(state.get("pre_observation_hash") or "") == state_hash and
            str(prefix[-1].get("url_after") or "").rstrip("/") ==
            str(state.get("url") or "").rstrip("/") and
            actions == [str(action) for action in
                        (state.get("replay_prefix") or [])] and
            len(target_matches) == 1
        )
        if not alternate_verified:
            raise PointBatchPreparationError(
                f"trajectory/state snapshot mismatch for {state.get('state_id')}")
        # Pre-fix point collection wrote a default-pruned raw step and an
        # action-anchored state view from the same in-memory observation.  Do
        # not rewrite that raw asset: use the verified state view for the last
        # history delta and mark the alternate compaction explicitly.
        alignment = "verified_same_step_target_anchored_view"

    history_snapshot = (str(decision_snapshot)
                        if decision_snapshot is not None else state_snapshot)
    if decision_snapshot is not None and history_snapshot != state_snapshot:
        alignment = "verified_probe_transition_dynamic_target"
    history: list[dict] = []
    for previous, current in zip(prefix, prefix[1:]):
        action = current.get("action")
        if not action:
            raise PointBatchPreparationError(
                f"trajectory {trajectory_id} step {current.get('step_id')} lacks action")
        current_tree = (history_snapshot if current is prefix[-1]
                        else current.get("obs_after_axtree", ""))
        history.append(history_entry(
            str(action),
            {"url": previous.get("url_after", ""),
             "axtree_txt": previous.get("obs_after_axtree", "")},
            {"url": current.get("url_after", ""),
             "axtree_txt": current_tree},
        ))
    if len(history) != len(state.get("replay_prefix") or []):
        raise PointBatchPreparationError(
            f"trajectory history/replay length mismatch for {state.get('state_id')}")
    return history, alignment


def materialize_point_reached_states(
        data_root: Path, output_path: Path | None = None) -> dict[str, Any]:
    """Join canonical points back to their exact collected decision states.

    The output is a derived assembler input.  It copies no labels by action
    class: each row is keyed by the point's exact state/candidate/action tuple
    and carries trajectory-derived history from the immutable raw artifact.
    """
    root = Path(data_root)
    points = load_probe_points(
        root / "grounded" / "probe_points.jsonl", validate=True)
    if not points:
        raise PointBatchPreparationError("cannot materialize zero formal points")
    candidates = assert_candidate_manifest_integrity(
        root / "raw" / "candidates" / FORMAL_CANDIDATE_BODY_NAME)
    transitions = load_probe_transitions(
        root / "grounded" / "transitions" /
        "probe_transitions.v1.jsonl", validate=True)
    assert_point_transition_integrity(points, transitions, require_all=False)
    transitions_by_point = {
        transition.probe_point_id: transition
        for transition in transitions.values()
    }
    states = _load_states(
        root / "raw" / "state_bank",
        {point.state_id for point in points.values()})
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for point in sorted(points.values(), key=lambda row: row.probe_point_id):
        state = states[point.state_id]
        candidate = candidates.get(point.candidate_id)
        if candidate is None:
            raise PointBatchPreparationError(
                f"{point.probe_point_id}: missing candidate {point.candidate_id}")
        if (candidate.state_id != point.state_id or
                candidate.snapshot_hash !=
                str(point.evidence.get("candidate_snapshot_hash") or "")):
            raise PointBatchPreparationError(
                f"{point.probe_point_id}: candidate/state snapshot mismatch")
        transition = transitions_by_point.get(point.probe_point_id)
        recorded_snapshot = str(state.get("axtree_snapshot") or "")
        decision_snapshot = (
            transition.pre_observation.policy_axtree
            if isinstance(transition, ProbeTransition) else recorded_snapshot)
        controls = [element for element in interactive_elements(decision_snapshot)
                    if element["bid"] == candidate.bid]
        if len(controls) != 1:
            raise PointBatchPreparationError(
                f"{point.probe_point_id}: exact candidate bid is not unique")
        if point.raw_action != f"click('{candidate.bid}')":
            raise PointBatchPreparationError(
                f"{point.probe_point_id}: raw action/candidate bid mismatch")
        state_snapshot_hash = hashlib.sha256(
            str(state.get("axtree_snapshot") or "").encode("utf-8")).hexdigest()
        if candidate.snapshot_hash != state_snapshot_hash:
            raise PointBatchPreparationError(
                f"{point.probe_point_id}: candidate hash/state snapshot mismatch")
        if point.pre_observation_hash != state_snapshot_hash:
            dynamic_verified = (
                isinstance(transition, ProbeTransition) and
                transition.pre_observation.policy_axtree_sha256 ==
                point.pre_observation_hash and
                transition.replay_verification ==
                "dynamic_page_target_contract")
            if not dynamic_verified:
                excluded.append({
                    "probe_point_id": point.probe_point_id,
                    "state_id": point.state_id,
                    "reason": (
                        "measured pre_observation differs from reviewed candidate "
                        "snapshot without a verified dynamic transition body; "
                        "point retained as grounding/eval evidence but excluded "
                        "from exact-input training"),
                })
                continue
        history, history_snapshot_alignment = _trajectory_history(
            root, state, decision_snapshot=decision_snapshot)
        url = str(state.get("url") or point.url)
        entity = url.rstrip("/").rsplit("/", 1)[-1]
        if entity.endswith(".html"):
            entity = entity[:-5]
        if point.action_type == "reddit_vote":
            match = re.search(r"/f/[^/]+/(\d+)(?:/|$)", url)
            if match is None:
                raise PointBatchPreparationError(
                    f"{point.probe_point_id}: cannot derive Reddit submission entity")
            canonical_entity_id = f"reddit:submission:{match.group(1)}"
        elif point.action_type == "reddit_subscribe":
            match = re.search(r"/f/([^/?#]+)(?:/|$)", url)
            if match is None:
                raise PointBatchPreparationError(
                    f"{point.probe_point_id}: cannot derive Reddit forum entity")
            canonical_entity_id = f"reddit:forum:{match.group(1)}"
        else:
            canonical_entity_id = entity
        rows.append({
            "state_id": point.state_id,
            "name": f"{point.state_id}__{point.probe_point_id}",
            "site": point.site,
            "action_type": point.action_type,
            "goal": str(state.get("goal") or ""),
            "reached": True,
            "url": url,
            "axtree_snapshot": decision_snapshot,
            "pre_fingerprint": dict(state.get("pre_fingerprint") or {}),
            "pre_observation_hash": point.pre_observation_hash,
            "risky_action": {
                "text": controls[0]["line"], "bid": candidate.bid,
                "raw_action": point.raw_action, "kind": "click",
                "canonical_action": point.canonical_action,
                "candidate_id": point.candidate_id,
                "action_instance_id": point.action_instance_id,
                "probe_point_id": point.probe_point_id,
            },
            "safe_answer": "go_back()",
            "probe_point_id": point.probe_point_id,
            "probe_run_id": point.probe_run_id,
            "candidate_id": point.candidate_id,
            "action_instance_id": point.action_instance_id,
            "canonical_action": point.canonical_action,
            "candidate_snapshot_hash": candidate.snapshot_hash,
            "transition_id": (
                transition.transition_id
                if isinstance(transition, ProbeTransition) else ""),
            "transition_body_verified": isinstance(
                transition, ProbeTransition),
            "replay_prefix": list(state.get("replay_prefix") or []),
            "reach_plan": [],
            "history": history,
            "history_source": "trajectory",
            "history_snapshot_alignment": history_snapshot_alignment,
            "collector_success": state.get("collector_success") is True,
            "traj_success": state.get("traj_success") is True,
            "environment_origin": point.environment_origin,
            "environment_family": point.environment_family,
            "environment_instance": point.environment_instance,
            "is_mock": point.is_mock,
            "task_id": point.task_id,
            "trajectory_id": point.trajectory_id,
            "run_id": point.run_id,
            "seed": point.seed,
            "account": point.account,
            "privilege": point.privilege,
            "canonical_entity_id": canonical_entity_id,
            "page_template_id": (
                "shopping.product_detail" if point.site == "shopping" else
                "reddit.submission" if point.action_type == "reddit_vote" else
                "reddit.forum" if point.action_type == "reddit_subscribe" else
                f"{point.site}.unknown"),
        })
    output = (Path(output_path) if output_path else root / "raw" / "state_bank" /
              "formal_point_reached_states.jsonl")
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, output)
    return {
        "schema_version": "iris.formal_point_states.v1",
        "n_points": len(points), "n_rows": len(rows),
        "n_excluded": len(excluded), "excluded": excluded,
        "output": str(output),
        "history_sources": {"trajectory": len(rows)},
        "history_snapshot_alignment": {
            key: sum(row.get("history_snapshot_alignment") == key for row in rows)
            for key in sorted({str(row.get("history_snapshot_alignment"))
                               for row in rows})
        },
    }
