"""Deterministic point-state collection with full, immutable lineage.

Unlike task-driven trajectory collection (``collect.py``), this reaches a
*reproducible decision state* by fixed navigation and records one key state per
``(state, action)`` closure.  ``collector_success`` here means the reviewed
affordance was actually reached (not a WebArena task reward), which is the
correct success criterion for reversibility probing: the state must be
replayable, the affordance must be present.

Each closure writes the same transactional lineage the formal audit checks:
``COMPLETE run manifest -> one meta row -> one raw trajectory -> one key
state``, with matching run/trajectory/state identities and a webarena origin.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .. import config
from ..envs.harness import RevActEnv
from ..envs.obs_utils import (extract_interactive_bids, find_bid_by_text,
                              prune_axtree_txt)
from .collect import (KeyStateRecord, _atomic_json, _collection_failure,
                     _safe_run_id, append_jsonl, new_run_id)


@dataclass(frozen=True)
class PointReachSpec:
    """One reproducible decision state and the affordance it must expose."""
    state_id: str
    action_type: str
    nav_plan: list[str]          # exact BrowserGym actions replayed from reset
    affordance_terms: list[str]  # substrings identifying the target control
    site: str = "shopping"
    account: str = "customer"
    privilege: str = "customer"


_STATE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def version_point_reach_specs(
        specs: list[PointReachSpec], suffix: str) -> list[PointReachSpec]:
    """Give a deliberate recollection a fresh, visibly versioned state ID.

    Dynamic pages may stop reproducing an old snapshot.  Recollection must
    never rebase that immutable row: it creates a new state/candidate lineage.
    """
    from dataclasses import replace

    revision = str(suffix or "").strip()
    if not revision:
        return list(specs)
    if not _STATE_VERSION.fullmatch(revision):
        raise ValueError(
            "state_id_suffix must match [A-Za-z0-9][A-Za-z0-9._-]*")
    versioned = [
        replace(spec, state_id=f"{spec.state_id}__{revision}")
        for spec in specs
    ]
    if len({spec.state_id for spec in versioned}) != len(versioned):
        raise ValueError("versioned point-state ids are not unique")
    return versioned


def _sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _resolve_step(renv: RevActEnv, action: str) -> Optional[str]:
    """Resolve a dynamic ``CLICK:<term>`` nav step to a concrete click action.

    Returns the executed action string (recorded in ``renv.history``), or None
    if the affordance is absent (the reach then fails, never fabricated)."""
    if not action.startswith("CLICK:"):
        renv.step(action)
        return action
    term = action.split(":", 1)[1]
    el = find_bid_by_text(renv._last_obs_view, [term])
    if el is None:
        return None
    resolved = f"click('{el['bid']}')"
    renv.step(resolved)
    return resolved


def _reach_one(renv: RevActEnv, spec: PointReachSpec, seed: int, run_id: str,
               trajectory_id: str, code_version: str = ""
               ) -> tuple[Optional[dict], dict]:
    """Replay the fixed nav plan; return (key_state_row | None, summary)."""
    # The raw StepRecord is part of the same immutable lineage closure as the
    # state/meta/manifest.  Omitting code_version here left the raw reset and
    # navigation rows unpinned even though every derived row claimed a code
    # snapshot.
    renv.reset(seed=seed, trajectory_id=trajectory_id, run_id=run_id,
               code_version=code_version)
    plan_ok = True
    for action in spec.nav_plan:
        if _resolve_step(renv, action) is None:
            plan_ok = False
            break
    view = renv._last_obs_view
    full_axtree = view.get("axtree_txt", "") or ""
    # Locate the reviewed affordance in the full observation first, then anchor
    # compaction on its exact bid.  Prefix truncation made long Postmill pages
    # appear to lack controls that were genuinely present near the tail.
    full_el = find_bid_by_text({"axtree_txt": full_axtree}, spec.affordance_terms)
    anchor_bids = [str(full_el["bid"])] if full_el is not None else []
    pruned = prune_axtree_txt(full_axtree, anchor_bids=anchor_bids)
    matches = [row for row in extract_interactive_bids(pruned)
               if full_el is not None and row["bid"] == str(full_el["bid"])]
    el = matches[0] if len(matches) == 1 else None
    reached = plan_ok and el is not None
    fp = renv.current_fingerprint()
    step_id = renv.step_id
    row = None
    if reached:
        ks = KeyStateRecord(
            state_id=spec.state_id,
            task_id=renv.task_id,
            site=renv.site,
            goal="",
            trajectory_id=trajectory_id,
            step_id=step_id,
            afforded_action_types=[spec.action_type],
            replay_prefix=list(renv.history),
            seed=seed,
            url=view.get("url", ""),
            axtree_snapshot=pruned,
            interactive_bids=extract_interactive_bids(view.get("axtree_txt", "")),
            pre_fingerprint=fp.to_dict(),
            backend_state=view.get("backend_state"),
            traj_success=True,
            run_id=run_id,
            logical_trajectory_id=spec.state_id,
            environment_origin="webarena",
            environment_family="webarena",
            environment_instance=config.site_base(renv.site),
            is_mock=False,
            collector_success=True,
            keyword_ranking_hints=[spec.action_type],
            code_version=code_version,
        )
        row = asdict(ks)
        # Extra formal-join fields consumed by the point runner + assemblers.
        row["pre_observation_hash"] = _sha256(pruned)
        row["affordance_bid"] = el["bid"]
        row["account"] = spec.account
        row["privilege"] = spec.privilege
        row["snapshot_compaction"] = "target_anchored.v1"
        # The raw trajectory and key-state must describe the same model input.
        # StepLogger's generic prefix compaction can omit a tail control even
        # though the point collector retained it with an anchor.  Store the
        # anchored view in the final raw step at collection time; the full
        # interactive element inventory above remains available for audit.
        if renv.logger.records:
            renv.logger.records[-1].obs_after_axtree = pruned
    summary = {
        "trajectory_id": trajectory_id,
        "logical_trajectory_id": spec.state_id,
        "run_id": run_id,
        "task_id": renv.task_id,
        "seed": seed,
        "state_id": spec.state_id,
        "action_type": spec.action_type,
        "n_steps": step_id,
        "success": reached,
        "collector_success": reached,
        "traj_success": reached,
        "environment_origin": "webarena",
        "environment_family": "webarena",
        "environment_instance": config.site_base(renv.site),
        "site": renv.site,
        "is_mock": False,
        "n_key_states": 1 if reached else 0,
        "code_version": code_version,
        "snapshot_compaction": "target_anchored.v1" if reached else "default",
    }
    return row, summary


def collect_point_states(specs: list[PointReachSpec], *, seed: int = 0,
                         out_dir: Optional[Path] = None,
                         env_factory=None, run_id: str | None = None,
                         code_version: str = "") -> dict:
    """Reach every spec, writing one immutable lineage closure per success."""
    config.ensure_dirs()
    root = Path(out_dir) if out_dir else config.DATA_ROOT
    raw_dir = root / "raw" / "trajectories"
    sites = {spec.site for spec in specs}
    if len(sites) != 1:
        raise ValueError(
            f"one point collection run must target exactly one site, got {sorted(sites)}")
    site = next(iter(sites))
    if site not in config.SITES:
        raise ValueError(f"unknown point collection site {site!r}")
    state_bank_dir = root / "raw" / "state_bank"
    state_bank_path = state_bank_dir / f"{site}_key_states.jsonl"
    meta_path = root / "raw" / "trajectories_meta.jsonl"

    requested = [spec.state_id for spec in specs]
    duplicate_request = sorted({sid for sid in requested
                                if requested.count(sid) > 1})
    existing: set[str] = set()
    for existing_path in sorted(state_bank_dir.glob("*.jsonl")):
        if existing_path.name == "formal_point_reached_states.jsonl":
            continue
        for line in existing_path.open(encoding="utf-8"):
            if line.strip():
                existing.add(str(json.loads(line).get("state_id") or ""))
    collisions = sorted(set(requested) & existing)
    if duplicate_request or collisions:
        raise ValueError(
            "point-state ids must be immutable and unique; "
            f"duplicate_request={duplicate_request}, already_exists={collisions}. "
            "Use new versioned state_ids for a deliberate recollection")

    run_id = _safe_run_id(run_id or new_run_id())
    manifest_path = root / "manifests" / "collection_runs" / f"{run_id}.json"

    started_at = datetime.now(timezone.utc).isoformat()
    _atomic_json(manifest_path, {
        "schema_version": "iris.collection-run.v2", "run_id": run_id,
        "status": "IN_PROGRESS", "started_at": started_at,
        "completed_at": None, "trajectories": [],
        "code_version": code_version,
    }, exclusive=True)

    from ..envs.harness import make_env
    factory = env_factory or (lambda task_id: make_env(task_id, headless=True))

    summaries: list[dict] = []
    written: list[str] = []
    session_task = config.SITES[site].session_task
    renv: RevActEnv | None = None
    try:
        env = factory(str(session_task))
        renv = RevActEnv(env, task_id=str(session_task), site=site)
        for spec in specs:
            trajectory_id = f"{spec.state_id}__run_{run_id}"
            row, summary = _reach_one(
                renv, spec, seed, run_id, trajectory_id, code_version)
            # Raw trajectory (reset + nav steps) is written for every attempt.
            raw_path = raw_dir / f"{trajectory_id}.jsonl"
            renv.logger.to_jsonl(raw_path)
            try:
                summary["raw_artifact"] = str(raw_path.relative_to(root))
            except ValueError:
                summary["raw_artifact"] = str(raw_path)
            if row is not None:
                append_jsonl(state_bank_path, [row])
                written.append(spec.state_id)
            append_jsonl(meta_path, [summary])
            summaries.append(summary)
            print(f"[point-collect] {spec.state_id:34s} action={spec.action_type:20s} "
                  f"reached={summary['success']} steps={summary['n_steps']}")
    except Exception as exc:
        _atomic_json(manifest_path, {
            "schema_version": "iris.collection-run.v2", "run_id": run_id,
            "status": "FAILED", "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "trajectories": summaries, "code_version": code_version,
            "site": site,
            "failure": _collection_failure(exc, stage="point_reach"),
        })
        raise
    finally:
        if renv is not None:
            renv.close()

    def _artifact(path: Path) -> str:
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)

    _atomic_json(manifest_path, {
        "schema_version": "iris.collection-run.v2", "run_id": run_id,
        "status": "COMPLETE", "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "trajectories": summaries,
        "code_version": code_version,
        "site": site,
        "state_bank": _artifact(state_bank_path), "meta": _artifact(meta_path),
    })
    print(f"[point-collect] done. reached={len(written)}/{len(specs)} "
          f"run_id={run_id}")
    return {"run_id": run_id, "n_reached": len(written),
            "state_ids": written, "summaries": summaries,
            "collection_manifest": str(manifest_path),
            "state_bank": str(state_bank_path), "meta": str(meta_path)}


def default_shopping_specs(product_urls: list[str], base: str, *,
                           n_cart: int = 10, n_wishlist: int = 8,
                           n_compare: int = 8) -> list[PointReachSpec]:
    """A spread of reproducible non-destructive shopping decision states."""
    def slug(url: str) -> str:
        tail = url.rstrip("/").rsplit("/", 1)[-1]
        return tail[:-5] if tail.endswith(".html") else tail

    specs: list[PointReachSpec] = []
    for url in product_urls[:n_cart]:
        specs.append(PointReachSpec(
            state_id=f"pt_add_to_cart__{slug(url)[:40]}",
            action_type="add_to_cart", nav_plan=[f"goto('{url}')"],
            affordance_terms=["add to cart"]))
    for url in product_urls[:n_wishlist]:
        specs.append(PointReachSpec(
            state_id=f"pt_wishlist_add__{slug(url)[:40]}",
            action_type="wishlist_add", nav_plan=[f"goto('{url}')"],
            affordance_terms=["add to wish list"]))
    for url in product_urls[:n_compare]:
        specs.append(PointReachSpec(
            state_id=f"pt_compare_add__{slug(url)[:40]}",
            action_type="compare_add", nav_plan=[f"goto('{url}')"],
            affordance_terms=["add to compare"]))
    return specs


def destructive_place_order_specs(product_urls: list[str], base: str, *,
                                  n: int = 3) -> list[PointReachSpec]:
    """Reach the checkout Place-Order affordance (destructive; double-gated)."""
    checkout = base + config.SHOPPING_PATHS["checkout"]
    specs = []
    for url in product_urls[:n]:
        tail = url.rstrip("/").rsplit("/", 1)[-1]
        s = tail[:-5] if tail.endswith(".html") else tail
        specs.append(PointReachSpec(
            state_id=f"pt_place_order__{s[:40]}",
            action_type="place_order",
            nav_plan=[f"goto('{url}')", "CLICK:add to cart",
                      f"goto('{checkout}')", "CLICK:next"],
            affordance_terms=["place order"]))
    return specs


def default_reddit_specs(submission_urls: list[str], forum_urls: list[str],
                         *, n_vote: int = 3, n_subscribe: int = 3
                         ) -> list[PointReachSpec]:
    """Reproducible, self-recovering Postmill decision states.

    These specs only reach the visible toggle.  Behavioral labels remain the
    output of the later execute--then--undo point runner.
    """
    def tail(url: str) -> str:
        parts = [part for part in url.split("?")[0].rstrip("/").split("/") if part]
        numeric = next((part for part in reversed(parts) if part.isdigit()), "")
        return numeric or (parts[-1] if parts else "root")

    specs: list[PointReachSpec] = []
    for url in submission_urls[:n_vote]:
        specs.append(PointReachSpec(
            state_id=f"pt_reddit_vote__{tail(url)[:40]}",
            action_type="reddit_vote", nav_plan=[f"goto('{url}')"],
            affordance_terms=["upvote"], site="reddit"))
    for url in forum_urls[:n_subscribe]:
        specs.append(PointReachSpec(
            state_id=f"pt_reddit_subscribe__{tail(url)[:40]}",
            action_type="reddit_subscribe", nav_plan=[f"goto('{url}')"],
            affordance_terms=["subscribe"], site="reddit"))
    return specs
