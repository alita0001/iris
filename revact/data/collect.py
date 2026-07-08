"""S2: trajectory collection + key-state extraction.

Runs a policy inside RevActEnv, logs step-level raw trajectories, and mines
"key states" (pages that afford one of the pilot target actions). Each key-state
record carries the `replay_prefix` needed by S5 to reproduce the state and probe
reversibility.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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
    """Which pilot action types this page affords (shallow keyword match)."""
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


# --------------------------------------------------------------------------- #
# Trajectory collection
# --------------------------------------------------------------------------- #
def collect_trajectory(
    renv: RevActEnv,
    policy,
    seed: int,
    trajectory_id: str,
    max_steps: int = 30,
) -> tuple[list[KeyStateRecord], dict]:
    """Roll out one trajectory; return (key_states, summary)."""
    if hasattr(policy, "reset"):
        policy.reset()

    _obs, _info, view = renv.reset(seed=seed, trajectory_id=trajectory_id)
    goal = renv.goal
    key_states: list[KeyStateRecord] = []
    running_history: list[dict] = []

    def _maybe_key_state(step_id: int):
        types = afforded_action_types(view)
        if not types:
            return
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
                interactive_bids=extract_interactive_bids(view.get("axtree_txt", "")),
                pre_fingerprint=fp.to_dict(),
                backend_state=view.get("backend_state"),
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

    final_response = getattr(policy, "last_raw_response", "") or ""
    summary = {
        "trajectory_id": trajectory_id,
        "task_id": renv.task_id,
        "seed": seed,
        "n_steps": renv.step_id,
        "terminated": terminated,
        "truncated": truncated,
        "max_reward": max_reward,
        "success": success,
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


def _state_key(rec: dict) -> tuple:
    return (rec["trajectory_id"], rec["step_id"], rec.get("url", ""),
            tuple(rec.get("afforded_action_types", [])))


def _load_seen_state_keys(path: Path) -> set:
    seen = set()
    if path.exists():
        for r in (json.loads(ln) for ln in path.open(encoding="utf-8") if ln.strip()):
            seen.add(_state_key(r))
    return seen


def run_collection(
    env_factory,
    policy_factory,
    task_ids: list[str],
    seeds: list[int],
    out_dir: Optional[Path] = None,
    max_steps: int = 30,
    only_success: bool = False,
    save_screenshots: bool = False,
) -> dict:
    """Collect over (task_id, seed) pairs.

    env_factory(task_id) -> a gym-like env;
    policy_factory(task_id, seed) -> a fresh policy.
    only_success: if True, only write key states from successful trajectories
                  (the expert set); raw trajectories + meta are always written.
    save_screenshots: persist a PNG per step (see RevActEnv) for `revact viz`.
    """
    config.ensure_dirs()
    if out_dir:
        base = Path(out_dir)
        raw_dir = base / "trajectories"
        state_bank_path = base / "state_bank" / f"{config.SITE}_key_states.jsonl"
        meta_path = base / "trajectories_meta.jsonl"
    else:
        raw_dir = config.RAW_TRAJ_DIR
        state_bank_path = config.STATE_BANK_DIR / f"{config.SITE}_key_states.jsonl"
        meta_path = config.RAW_DIR / "trajectories_meta.jsonl"

    seen_states = _load_seen_state_keys(state_bank_path)  # cross-run de-dup
    summaries = []
    n_success = 0
    for task_id in task_ids:
        env = env_factory(task_id)
        renv = RevActEnv(env, task_id=task_id, save_screenshots=save_screenshots)
        try:
            for seed in seeds:
                traj_id = f"{task_id}_seed{seed}"
                policy = policy_factory(task_id, seed)
                key_states, summary = collect_trajectory(
                    renv, policy, seed=seed, trajectory_id=traj_id, max_steps=max_steps
                )
                renv.logger.to_jsonl(raw_dir / f"{traj_id}.jsonl")
                append_jsonl(meta_path, [summary])

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

                n_success += int(summary["success"])
                summaries.append(summary)
                print(
                    f"[collect] {traj_id}: steps={summary['n_steps']} "
                    f"success={summary['success']} reward={summary['max_reward']} "
                    f"key_states(+new)={len(fresh)}/{len(key_states)}"
                )
        finally:
            renv.close()
    print(
        f"[collect] done. trajectories={len(summaries)} success={n_success} "
        f"raw_dir={raw_dir} state_bank={state_bank_path} meta={meta_path}"
    )
    return {"summaries": summaries, "n_success": n_success,
            "state_bank": str(state_bank_path), "meta": str(meta_path)}
