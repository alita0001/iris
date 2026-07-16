"""S2: trajectory collection + task-independent mutation-candidate states.

Runs a policy inside RevActEnv, logs step-level raw trajectories, and mines
candidate states whenever a page exposes legal interactive controls.  The old
English keyword table survives only as a ranking hint.  Persistent mutation is
decided later by :mod:`revact.data.mutation_miner` through execute→signal-diff→
reset; collection never treats a keyword hit as a behavioral label.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

from .. import config, prompts
from ..envs.harness import PolicyAttemptRecord, RevActEnv
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
    environment_instance: str = ""
    is_mock: bool = False
    collector_success: bool = False
    discovery_method: str = "interactive_control_enumeration"
    mutation_probe_status: str = "UNPROBED"
    keyword_ranking_hints: list[str] | None = None
    # Commit/worktree identifier supplied by the collection orchestrator.
    # Legacy rows legitimately omit it; new point-state collection records it.
    code_version: str = ""


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


def _site_for_url(url: str, fallback: str) -> str:
    """Resolve the concrete WebArena site from the observed origin.

    BrowserGym task ids are not site-encoded, so stamping the global pilot
    default silently mislabeled Reddit trajectories as shopping.  The reset
    observation is authoritative and every configured WebArena mirror has a
    distinct origin in this deployment.
    """
    target = urlsplit(str(url or ""))
    matches = []
    for name in config.SITES:
        base = urlsplit(config.site_base(name))
        if (target.scheme, target.netloc) == (base.scheme, base.netloc) and \
                target.scheme and target.netloc:
            matches.append(name)
    return matches[0] if len(matches) == 1 else fallback


def _policy_provenance(policy) -> dict[str, Any]:
    """Serialize model/decode identity without serializing a credential."""
    guarded = (hasattr(policy, "policy") and
               policy.__class__.__name__ == "ReadOnlyPolicyGuard")
    inner = policy.policy if guarded else policy
    return {
        "wrapper": policy.__class__.__name__,
        "implementation": inner.__class__.__name__,
        "provider": str(getattr(inner, "provider", "") or "local_or_scripted"),
        "model": str(getattr(inner, "model", "") or ""),
        "base_url": str(getattr(inner, "base_url", "") or ""),
        "api_key_env": str(getattr(inner, "api_key_env", "") or ""),
        "credential_value_stored": False,
        "decode": {
            "temperature": getattr(inner, "temperature", None),
            "top_p": getattr(inner, "top_p", None),
            "max_tokens": getattr(inner, "max_tokens", None),
            "max_history": getattr(inner, "max_history", None),
        },
        "read_only_guard": guarded,
    }


def _judge_provenance() -> dict[str, Any]:
    """Capture the configured WebArena judge route, never its key value."""
    return {
        "mode": os.environ.get("REVACT_WA_JUDGE", "off"),
        "provider": os.environ.get("REVACT_WA_JUDGE_PROVIDER", ""),
        "model": os.environ.get("REVACT_WA_JUDGE_MODEL", ""),
        "base_url": os.environ.get("REVACT_WA_JUDGE_BASE_URL", ""),
        "api_key_env": os.environ.get("REVACT_WA_JUDGE_API_KEY_ENV", ""),
        "credential_value_stored": False,
    }


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def _episode_observation_source(view: dict) -> dict[str, Any]:
    """Portable exact observation for later, fail-closed episode authoring."""
    axtree = str(view.get("axtree_txt") or "")
    persistent = view.get("persistent_signal")
    if persistent is None:
        persistent = view.get("backend_state")
    return {
        "url": str(view.get("url") or ""),
        "title": str(view.get("title") or ""),
        "axtree_txt": axtree,
        "axtree_sha256": hashlib.sha256(axtree.encode("utf-8")).hexdigest(),
        "persistent_signal": persistent,
    }


def collect_trajectory(
    renv: RevActEnv,
    policy,
    seed: int,
    trajectory_id: str,
    run_id: str = "",
    logical_trajectory_id: str = "",
    max_steps: int = 30,
    code_version: str = "",
) -> tuple[list[KeyStateRecord], dict]:
    """Roll out one trajectory; return (key_states, summary)."""
    if hasattr(policy, "reset"):
        policy.reset()

    _obs, _info, view = renv.reset(
        seed=seed, trajectory_id=trajectory_id, run_id=run_id,
        code_version=code_version)
    renv.site = _site_for_url(view.get("url", ""), renv.site)
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
                environment_family=(
                    config.site_environment_family(renv.site) or renv.site),
                environment_instance=config.site_base(renv.site),
                is_mock=renv.task_id.startswith("mock"),
                keyword_ranking_hints=list(types),
                code_version=code_version,
            )
        )

    # step 0 (post-reset) can itself be a key state
    _maybe_key_state(step_id=0)

    terminated = truncated = False
    max_reward = 0.0
    for attempt_index in range(max_steps):
        action = policy.act(view, goal=goal, history=running_history)
        # Capture the exact model call before the environment moves.  A
        # scripted policy legitimately has no call trace and remains explicit
        # rather than being assigned fabricated messages/completion.
        policy_messages = getattr(policy, "last_request_messages", []) or []
        policy_messages = json.loads(json.dumps(
            policy_messages, ensure_ascii=False)) if policy_messages else []
        assistant_completion = str(
            getattr(policy, "last_raw_response", "") or "")
        proposed_action = str(
            getattr(policy, "last_proposed_action", "") or action or "")
        proposed_completion = str(
            getattr(policy, "last_proposed_completion", "") or
            assistant_completion)
        finish_reason = str(
            getattr(policy, "last_finish_reason", "") or "")
        guarded = finish_reason == "read_only_guard"
        executed_action = str(action or "")
        executed_completion = assistant_completion if action else ""
        inner_policy = getattr(policy, "policy", policy)
        renv.policy_attempt_logger.add(PolicyAttemptRecord(
            schema_version="iris.policy-attempt.v1",
            run_id=run_id,
            trajectory_id=trajectory_id,
            task_id=renv.task_id,
            attempt_index=attempt_index,
            step_id_before=renv.step_id,
            url=str(view.get("url") or ""),
            policy_input_messages=policy_messages,
            policy_input_messages_sha256=(
                _canonical_json_sha256(policy_messages)
                if policy_messages else ""),
            proposed_action=proposed_action,
            proposed_completion=proposed_completion,
            proposed_completion_sha256=(
                hashlib.sha256(proposed_completion.encode("utf-8")).hexdigest()
                if proposed_completion else ""),
            executed_action=executed_action,
            executed_completion=executed_completion,
            executed_completion_sha256=(
                hashlib.sha256(executed_completion.encode("utf-8")).hexdigest()
                if executed_completion else ""),
            execution_status=("NO_ACTION" if not action else
                              "GUARDED" if guarded else "EXECUTED"),
            finish_reason=finish_reason,
            provider=str(getattr(inner_policy, "provider", "") or ""),
            model=str(getattr(inner_policy, "model", "") or ""),
            code_version=code_version,
        ))
        if not action:
            break
        prev_view = view
        _obs, reward, terminated, truncated, _info, view = renv.step(action)
        # P2 history entry: action + observed delta + change flag (no grounded
        # labels here — the policy must see only deployment-computable facts).
        observed_entry = history_entry(action, prev_view, view)
        running_history.append(observed_entry)
        raw_step = renv.logger.records[-1]
        raw_step.policy_call_captured = bool(
            policy_messages and assistant_completion)
        raw_step.policy_input_messages = policy_messages
        raw_step.policy_input_messages_sha256 = (
            _canonical_json_sha256(policy_messages) if policy_messages else "")
        raw_step.assistant_completion = assistant_completion
        raw_step.assistant_completion_sha256 = (
            hashlib.sha256(assistant_completion.encode("utf-8")).hexdigest()
            if assistant_completion else "")
        raw_step.pre_observation = _episode_observation_source(prev_view)
        raw_step.post_observation = _episode_observation_source(view)
        raw_step.observed_history_entry = dict(observed_entry)
        raw_step.policy_guarded = guarded
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
        "site": renv.site,
        "environment_family": (
            config.site_environment_family(renv.site) or renv.site),
        "is_mock": renv.task_id.startswith("mock"),
        "n_key_states": len(key_states),
        "final_model_response": final_response[:500],
        "final_finish_reason": getattr(policy, "last_finish_reason", ""),
        "read_only_guard_rejections": list(
            getattr(policy, "guard_rejections", []) or []),
        "policy_provenance": _policy_provenance(policy),
        "judge_provenance": _judge_provenance(),
        "goal": goal,
        "code_version": code_version,
        "episode_source_capture": {
            "schema_version": "iris.raw_episode_source.v1",
            "n_action_steps": renv.step_id,
            "n_exact_policy_calls": sum(
                record.policy_call_captured for record in renv.logger.records),
            "counts_as_formal_supervision": False,
        },
        "policy_attempt_source_capture": {
            "schema_version": "iris.policy-attempt.v1",
            "n_policy_attempts": len(renv.policy_attempt_logger.records),
            "n_unexecuted_policy_attempts": sum(
                record.execution_status == "NO_ACTION"
                for record in renv.policy_attempt_logger.records),
            "counts_as_environment_transitions": False,
            "counts_as_formal_supervision": False,
        },
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


def _collection_failure(exc: Exception, *, stage: str) -> dict[str, str]:
    """Return bounded, credential-redacted terminal failure provenance.

    A recoverable Python exception is different from a process crash.  Leaving
    both as ``IN_PROGRESS`` made it impossible to distinguish an abandoned
    transaction from one that was still running.  We retain a short diagnostic
    while refusing to serialize an exception payload verbatim: provider errors
    can contain request headers or credential-shaped values.
    """
    message = str(exc).replace("\n", " ").replace("\r", " ")
    message = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "<redacted>", message)
    message = re.sub(
        r"(?i)(api[_-]?key|authorization|bearer)(\s*[=:]\s*)\S+",
        r"\1\2<redacted>", message)
    return {
        "stage": str(stage or "unknown"),
        "error_type": type(exc).__name__,
        "message": message[:500],
    }


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
    code_version: str = "",
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
        policy_attempt_dir = base / "policy_attempts"
        state_bank_dir = base / "state_bank"
        meta_path = base / "trajectories_meta.jsonl"
    else:
        artifact_root = config.DATA_ROOT
        raw_dir = config.RAW_TRAJ_DIR
        policy_attempt_dir = config.RAW_DIR / "policy_attempts"
        state_bank_dir = config.STATE_BANK_DIR
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
        "code_version": code_version,
        "trajectories": [],
    }, exclusive=True)

    # New physical trajectory ids contain run_id, so independent attempts stay
    # independent; this set only prevents duplicate writes within one artifact.
    seen_states_by_path: dict[Path, set] = {}
    state_bank_paths: dict[str, Path] = {}
    summaries = []
    n_success = 0
    for task_id in task_ids:
        try:
            env = env_factory(task_id)
        except Exception as exc:
            _atomic_json(manifest_path, {
                "schema_version": "iris.collection-run.v2",
                "run_id": run_id, "status": "FAILED",
                "started_at": started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "code_version": code_version,
                "trajectories": summaries,
                "failure": _collection_failure(exc, stage="env_factory"),
            })
            raise
        renv = RevActEnv(env, task_id=task_id, save_screenshots=save_screenshots)
        try:
            for seed in seeds:
                logical_traj_id = f"{task_id}_seed{seed}"
                # The run suffix prevents raw JSONL replacement across retries.
                # ``logical_trajectory_id`` remains available for grouping.
                traj_id = f"{logical_traj_id}__run_{run_id}"
                policy = policy_factory(task_id, seed)
                # Freeze the exact effective registry before the first call;
                # taking this snapshot after rollout would permit a concurrent
                # workbench edit to misattribute every captured message.
                policy_provenance = _policy_provenance(policy)
                prompt_provenance = prompts.snapshot_generation(
                    root=artifact_root,
                    producer="revact.data.collect",
                    author=f"collection:{run_id}",
                    model={
                        "provider": policy_provenance.get("provider"),
                        "name": policy_provenance.get("model") or
                                policy_provenance.get("implementation"),
                        "wrapper": policy_provenance.get("wrapper"),
                        "implementation": policy_provenance.get(
                            "implementation"),
                    },
                    decode_config={
                        **dict(policy_provenance.get("decode") or {}),
                        "message_topology": "stateless",
                        "policy_builder": "revact.prompts.build_policy_messages",
                    },
                )
                key_states, summary = collect_trajectory(
                    renv, policy, seed=seed, trajectory_id=traj_id,
                    run_id=run_id, logical_trajectory_id=logical_traj_id,
                    max_steps=max_steps, code_version=code_version,
                )
                summary.update(prompt_provenance)
                summary["message_topology"] = "stateless"
                summary["policy_builder"] = \
                    "revact.prompts.build_policy_messages"
                raw_path = raw_dir / f"{traj_id}.jsonl"
                renv.logger.to_jsonl(raw_path)
                policy_attempt_path = policy_attempt_dir / f"{traj_id}.jsonl"
                renv.policy_attempt_logger.to_jsonl(policy_attempt_path)
                try:
                    summary["raw_artifact"] = str(raw_path.relative_to(artifact_root))
                    summary["policy_attempt_artifact"] = str(
                        policy_attempt_path.relative_to(artifact_root))
                except ValueError:
                    summary["raw_artifact"] = str(raw_path)
                    summary["policy_attempt_artifact"] = str(
                        policy_attempt_path)

                rows = [asdict(k) for k in key_states]
                if only_success and not summary["success"]:
                    rows = []
                site = str(summary.get("site") or config.SITE)
                state_bank_path = state_bank_dir / f"{site}_key_states.jsonl"
                state_bank_paths[site] = state_bank_path
                if state_bank_path not in seen_states_by_path:
                    seen_states_by_path[state_bank_path] = \
                        _load_seen_state_keys(state_bank_path)
                seen_states = seen_states_by_path[state_bank_path]
                fresh = []
                for r in rows:
                    k = _state_key(r)
                    if k in seen_states:
                        continue
                    seen_states.add(k)
                    fresh.append(r)
                append_jsonl(state_bank_path, fresh)
                try:
                    summary["state_bank_artifact"] = str(
                        state_bank_path.relative_to(artifact_root))
                except ValueError:
                    summary["state_bank_artifact"] = str(state_bank_path)
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
        except Exception as exc:
            _atomic_json(manifest_path, {
                "schema_version": "iris.collection-run.v2",
                "run_id": run_id, "status": "FAILED",
                "started_at": started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "code_version": code_version,
                "trajectories": summaries,
                "failure": _collection_failure(
                    exc, stage=f"trajectory:{task_id}"),
            })
            raise
        finally:
            renv.close()
    def _artifact_name(path: Path) -> str:
        try:
            return str(path.relative_to(artifact_root))
        except ValueError:
            return str(path)
    state_bank_artifacts = {
        site: _artifact_name(path)
        for site, path in sorted(state_bank_paths.items())
    }
    legacy_state_bank = (
        next(iter(state_bank_artifacts.values()))
        if len(state_bank_artifacts) == 1 else "")
    _atomic_json(manifest_path, {
        "schema_version": "iris.collection-run.v2",
        "run_id": run_id, "status": "COMPLETE",
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "code_version": code_version,
        "trajectories": summaries,
        "state_bank": legacy_state_bank,
        "state_banks": state_bank_artifacts,
        "meta": _artifact_name(meta_path),
    })
    print(
        f"[collect] done. trajectories={len(summaries)} success={n_success} "
        f"raw_dir={raw_dir} state_banks={state_bank_artifacts} meta={meta_path}"
    )
    return {"run_id": run_id, "summaries": summaries, "n_success": n_success,
            "collection_manifest": str(manifest_path),
            "state_bank": (str(next(iter(state_bank_paths.values())))
                           if len(state_bank_paths) == 1 else ""),
            "state_banks": {site: str(path)
                            for site, path in sorted(state_bank_paths.items())},
            "meta": str(meta_path)}
