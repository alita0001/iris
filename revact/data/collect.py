"""S2: trajectory collection + task-independent mutation-candidate states.

Runs a policy inside RevActEnv, logs step-level raw trajectories, and mines
candidate states whenever a page exposes legal interactive controls.  The old
English keyword table survives only as a ranking hint.  Persistent mutation is
decided later by :mod:`revact.data.mutation_miner` through execute→signal-diff→
reset; collection never treats a keyword hit as a behavioral label.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .. import config
from ..envs.harness import RevActEnv
from ..envs.obs_utils import extract_interactive_bids, history_entry, prune_axtree_txt
from ..policies import is_terminal_action


# --------------------------------------------------------------------------- #
# Key-state detection
# --------------------------------------------------------------------------- #
def afforded_action_types(obs_view: dict) -> list[str]:
    """Legacy pilot action-class *ranking hints* (never a discovery gate)."""
    text = (obs_view.get("axtree_txt", "") or "").lower()
    url = (obs_view.get("url", "") or "").lower()
    hits: list[str] = []
    for atype, keywords in config.PILOT_ACTION_KEYWORDS.items():
        matched = any(kw in text for kw in keywords)
        if atype == "delete_address":
            # only on an address-book-like page
            matched = matched and any(
                h in url or h in text for h in config.ADDRESS_URL_HINTS
            )
        if matched:
            hits.append(atype)
    return hits


@dataclass
class KeyStateRecord:
    state_id: str
    task_id: str
    site: str
    goal: str
    trajectory_id: str
    step_id: int
    afforded_action_types: list[str]
    replay_prefix: list[str]
    seed: int
    url: str
    axtree_snapshot: str
    interactive_bids: list[dict]
    pre_fingerprint: dict
    backend_state: Any = None
    traj_success: bool = False  # did the trajectory that reached this state succeed?
    run_id: str = ""
    logical_trajectory_id: str = ""
    environment_origin: str = "unknown"
    environment_family: str = ""
    is_mock: bool = False
    collector_success: bool = False
    discovery_method: str = "interactive_control_enumeration"
    mutation_probe_status: str = "UNPROBED"
    keyword_ranking_hints: list[str] | None = None


# --------------------------------------------------------------------------- #
# Trajectory collection
# --------------------------------------------------------------------------- #
def _environment_origin(task_id: str) -> str:
    task = str(task_id).lower()
    if task.startswith("mock"):
        return "mock"
    if task.startswith("webarena") or task.startswith("browsergym/webarena"):
        return "webarena"
    if task.startswith("workarena") or "servicenow" in task:
        return "workarena"
    return "unknown"


def collect_trajectory(
    renv: RevActEnv,
    policy,
    seed: int,
    trajectory_id: str,
    run_id: str = "",
    logical_trajectory_id: str = "",
    max_steps: int = 30,
) -> tuple[list[KeyStateRecord], dict]:
    """Roll out one trajectory; return (key_states, summary)."""
    if hasattr(policy, "reset"):
        policy.reset()

    _obs, _info, view = renv.reset(
        seed=seed, trajectory_id=trajectory_id, run_id=run_id)
    goal = renv.goal
    key_states: list[KeyStateRecord] = []
    running_history: list[dict] = []

    def _maybe_key_state(step_id: int):
        interactive = extract_interactive_bids(view.get("axtree_txt", ""))
        # Task-independent candidate discovery: enumerate controls first.
        # Whether any control mutates persistent state is measured downstream.
        if not interactive:
            return
        types = afforded_action_types(view)
        fp = renv.current_fingerprint()
        key_states.append(
            KeyStateRecord(
                state_id=f"{trajectory_id}_s{step_id}",
                task_id=renv.task_id,
                site=renv.site,
                goal=goal,
                trajectory_id=trajectory_id,
                step_id=step_id,
                afforded_action_types=types,
                replay_prefix=list(renv.history),
                seed=seed,
                url=view.get("url", ""),
                axtree_snapshot=prune_axtree_txt(view.get("axtree_txt", "")),
                interactive_bids=interactive,
                pre_fingerprint=fp.to_dict(),
                backend_state=view.get("backend_state"),
                run_id=run_id,
                logical_trajectory_id=logical_trajectory_id or trajectory_id,
                environment_origin=_environment_origin(renv.task_id),
                environment_family=renv.site,
                is_mock=renv.task_id.startswith("mock"),
                keyword_ranking_hints=list(types),
            )
        )

    # step 0 (post-reset) can itself be a key state
    _maybe_key_state(step_id=0)

    terminated = truncated = False
    max_reward = 0.0
    for _ in range(max_steps):
        action = policy.act(view, goal=goal, history=running_history)
        if not action:
            break
        prev_view = view
        _obs, reward, terminated, truncated, _info, view = renv.step(action)
        # P2 history entry: action + observed delta + change flag (no grounded
        # labels here — the policy must see only deployment-computable facts).
        running_history.append(history_entry(action, prev_view, view))
        max_reward = max(max_reward, float(reward))
        _maybe_key_state(step_id=renv.step_id)
        if terminated or truncated:
            break
        # a terminal action (final answer / give-up) was executed and scored by
        # WebArena on this step -> end the trajectory here.
        if is_terminal_action(action):
            break

    success = max_reward >= 1.0
    for k in key_states:  # success is only known after the whole rollout
        k.traj_success = success
        k.collector_success = success

    final_response = getattr(policy, "last_raw_response", "") or ""
    summary = {
        "trajectory_id": trajectory_id,
        "logical_trajectory_id": logical_trajectory_id or trajectory_id,
        "run_id": run_id,
        "task_id": renv.task_id,
        "seed": seed,
        "n_steps": renv.step_id,
        "terminated": terminated,
        "truncated": truncated,
        "max_reward": max_reward,
        "success": success,
        "collector_success": success,
        "environment_origin": _environment_origin(renv.task_id),
        "environment_family": renv.site,
        "is_mock": renv.task_id.startswith("mock"),
        "n_key_states": len(key_states),
        "final_model_response": final_response[:500],
        "final_finish_reason": getattr(policy, "last_finish_reason", ""),
    }
    return key_states, summary


def append_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def new_run_id() -> str:
    """Globally unique, path-safe collection batch identifier."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _safe_run_id(run_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]", "-", run_id).strip("-.")
    if not value or value != run_id:
        raise ValueError("run_id must be a non-empty path-safe identifier")
    return value


def _state_key(rec: dict) -> tuple:
    return (rec["trajectory_id"], rec["step_id"], rec.get("url", ""),
            tuple(rec.get("afforded_action_types", [])))


def _load_seen_state_keys(path: Path) -> set:
    seen = set()
    if path.exists():
        for r in (json.loads(ln) for ln in path.open(encoding="utf-8") if ln.strip()):
            seen.add(_state_key(r))
    return seen


def _atomic_json(path: Path, payload: dict, *, exclusive: bool = False) -> None:
    """Durably replace one run-status record without exposing partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive and path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_collection(
    env_factory,
    policy_factory,
    task_ids: list[str],
    seeds: list[int],
    out_dir: Optional[Path] = None,
    max_steps: int = 30,
    only_success: bool = False,
    save_screenshots: bool = False,
    run_id: str | None = None,
) -> dict:
    """Collect over (task_id, seed) pairs.

    env_factory(task_id) -> a gym-like env;
    policy_factory(task_id, seed) -> a fresh policy.
    only_success: if True, only write key states from successful trajectories
                  (the expert set); raw trajectories + meta are always written.
    save_screenshots: persist a PNG per step (see RevActEnv) for `revact viz`.
    """
    config.ensure_dirs()
    run_id = _safe_run_id(run_id or new_run_id())
    if out_dir:
        base = Path(out_dir)
        artifact_root = base
        raw_dir = base / "trajectories"
        state_bank_path = base / "state_bank" / f"{config.SITE}_key_states.jsonl"
        meta_path = base / "trajectories_meta.jsonl"
    else:
        artifact_root = config.DATA_ROOT
        raw_dir = config.RAW_TRAJ_DIR
        state_bank_path = config.STATE_BANK_DIR / f"{config.SITE}_key_states.jsonl"
        meta_path = config.RAW_DIR / "trajectories_meta.jsonl"

    # The immutable run record is allocated *before* raw/meta/key-state writes.
    # A process crash therefore leaves an auditable IN_PROGRESS transaction,
    # never an apparently complete collection with no run manifest.
    manifest_dir = artifact_root / "manifests" / "collection_runs"
    manifest_path = manifest_dir / f"{run_id}.json"
    started_at = datetime.now(timezone.utc).isoformat()
    _atomic_json(manifest_path, {
        "schema_version": "iris.collection-run.v2",
        "run_id": run_id, "status": "IN_PROGRESS",
        "started_at": started_at, "completed_at": None,
        "trajectories": [],
    }, exclusive=True)

    # New physical trajectory ids contain run_id, so independent attempts stay
    # independent; this set only prevents duplicate writes within one artifact.
    seen_states = _load_seen_state_keys(state_bank_path)
    summaries = []
    n_success = 0
    for task_id in task_ids:
        env = env_factory(task_id)
        renv = RevActEnv(env, task_id=task_id, save_screenshots=save_screenshots)
        try:
            for seed in seeds:
                logical_traj_id = f"{task_id}_seed{seed}"
                # The run suffix prevents raw JSONL replacement across retries.
                # ``logical_trajectory_id`` remains available for grouping.
                traj_id = f"{logical_traj_id}__run_{run_id}"
                policy = policy_factory(task_id, seed)
                key_states, summary = collect_trajectory(
                    renv, policy, seed=seed, trajectory_id=traj_id,
                    run_id=run_id, logical_trajectory_id=logical_traj_id,
                    max_steps=max_steps
                )
                raw_path = raw_dir / f"{traj_id}.jsonl"
                renv.logger.to_jsonl(raw_path)
                try:
                    summary["raw_artifact"] = str(raw_path.relative_to(artifact_root))
                except ValueError:
                    summary["raw_artifact"] = str(raw_path)

                rows = [asdict(k) for k in key_states]
                if only_success and not summary["success"]:
                    rows = []
                fresh = []
                for r in rows:
                    k = _state_key(r)
                    if k in seen_states:
                        continue
                    seen_states.add(k)
                    fresh.append(r)
                append_jsonl(state_bank_path, fresh)
                summary["n_key_states_written"] = len(fresh)
                summary["key_state_ids"] = [r["state_id"] for r in fresh]
                # Meta is the final per-trajectory commit record: it is written
                # only after raw and key-state artifacts exist.
                append_jsonl(meta_path, [summary])

                n_success += int(summary["success"])
                summaries.append(summary)
                print(
                    f"[collect] {traj_id}: steps={summary['n_steps']} "
                    f"success={summary['success']} reward={summary['max_reward']} "
                    f"key_states(+new)={len(fresh)}/{len(key_states)}"
                )
        finally:
            renv.close()
    def _artifact_name(path: Path) -> str:
        try:
            return str(path.relative_to(artifact_root))
        except ValueError:
            return str(path)
    _atomic_json(manifest_path, {
        "schema_version": "iris.collection-run.v2",
        "run_id": run_id, "status": "COMPLETE",
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "trajectories": summaries,
        "state_bank": _artifact_name(state_bank_path),
        "meta": _artifact_name(meta_path),
    })
    print(
        f"[collect] done. trajectories={len(summaries)} success={n_success} "
        f"raw_dir={raw_dir} state_bank={state_bank_path} meta={meta_path}"
    )
    return {"run_id": run_id, "summaries": summaries, "n_success": n_success,
            "collection_manifest": str(manifest_path),
            "state_bank": str(state_bank_path), "meta": str(meta_path)}
