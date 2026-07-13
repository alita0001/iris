"""S1: environment harness.

Provides:
  * make_env(task_id)            - lazily build a real BrowserGym WebArena env.
  * RevActEnv                    - thin wrapper owning step-level logging and the
                                   action history needed for replay-to-state.
  * StepLogger / StepRecord      - JSONL step logging.
  * replay_to_state(...)         - reset(seed) + replay a prefix, verifying we
                                   land on the recorded fingerprint.

Works over ANY env exposing the gymnasium contract used by BrowserGym and
MockShoppingEnv:
    obs, info = env.reset(seed=...)
    obs, reward, terminated, truncated, info = env.step(action_str)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .. import config
from .fingerprint import StateFingerprint, fingerprint, is_restored, state_distance
from .obs_utils import prune_axtree_txt, to_obs_view


# --------------------------------------------------------------------------- #
# Real WebArena env (lazy import)
# --------------------------------------------------------------------------- #
def _task_gym_id(task_id: str | int) -> str:
    s = str(task_id)
    if s.startswith(config.WEBARENA_GYM_PREFIX):
        return s
    if s.startswith("webarena."):
        return "browsergym/" + s
    return f"{config.WEBARENA_GYM_PREFIX}{s}"


def make_env(task_id: str | int, headless: bool = True, **gym_kwargs):
    """Create a real BrowserGym WebArena env. Requires the `agentlab` conda env."""
    import gymnasium as gym  # lazy
    import browsergym.webarena  # noqa: F401  (registers tasks)

    # Reward uses an LLM judge on some tasks; not needed for collection.
    # REVACT_WA_JUDGE selects: off (default, no key) | deepseek | openai.
    from . import webarena_patch

    webarena_patch.configure_reward_judge()
    return gym.make(_task_gym_id(task_id), headless=headless, **gym_kwargs)


# --------------------------------------------------------------------------- #
# Step logging
# --------------------------------------------------------------------------- #
@dataclass
class StepRecord:
    task_id: str
    site: str
    trajectory_id: str
    step_id: int
    action: Optional[str]
    url_before: str
    url_after: str
    reward: float
    terminated: bool
    truncated: bool
    obs_after_axtree: str  # pruned snapshot
    backend_after: Any = None
    replay_prefix: list[str] = field(default_factory=list)
    screenshot: str = ""   # path relative to DATA_ROOT, "" when not captured
    # Immutable collection-attempt identifier.  Legacy records legitimately
    # omit it; newly collected trajectories stamp the same value into the raw
    # steps, trajectory manifest, and key-state rows.
    run_id: str = ""


class StepLogger:
    def __init__(self) -> None:
        self.records: list[StepRecord] = []

    def add(self, rec: StepRecord) -> None:
        self.records.append(rec)

    def to_jsonl(self, path: Path) -> None:
        """Write one immutable raw trajectory artifact.

        Collection used to overwrite ``<task>_seed<n>.jsonl`` while appending a
        second metadata row.  Exclusive creation makes such lineage corruption
        fail loudly.  Callers that intentionally regenerate an artifact must
        choose a new run/trajectory id rather than mutating history.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# RevActEnv wrapper
# --------------------------------------------------------------------------- #
class RevActEnv:
    """Owns the env loop, step logging, and the running action history.

    With ``save_screenshots=True`` every step's ``obs["screenshot"]`` (HxWx3
    array provided by BrowserGym) is saved as PNG under
    ``DATA_ROOT/raw/screenshots/<trajectory_id>/step_<id>.png`` and its
    relative path recorded in the StepRecord — the raw material for
    `revact.cli viz`. Off by default (screenshots are the bulkiest part of a
    trajectory and most pipeline runs don't need them).
    """

    def __init__(self, env, task_id: str, site: str = config.SITE,
                 save_screenshots: bool = False):
        self.env = env
        self.task_id = str(task_id)
        self.site = site
        self.save_screenshots = save_screenshots
        self.logger = StepLogger()
        self.history: list[str] = []          # executed action strings (replay prefix)
        self.step_id = 0
        self.goal = ""
        self.trajectory_id = ""
        self.run_id = ""
        self._last_obs_view: dict = {}

    def _save_screenshot(self, obs) -> str:
        """Persist obs['screenshot'] when enabled; returns DATA_ROOT-relative
        path ('' when unavailable). Never raises — screenshots are best-effort
        and must not break collection or probing."""
        if not self.save_screenshots or not isinstance(obs, dict):
            return ""
        shot = obs.get("screenshot")
        if shot is None:
            return ""
        try:
            import numpy as np
            from PIL import Image  # lazy; the agentlab env ships Pillow

            img = Image.fromarray(np.asarray(shot)[..., :3])
            rel = Path("raw") / "screenshots" / self.trajectory_id / f"step_{self.step_id:03d}.png"
            out = config.DATA_ROOT / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            img.save(out)
            return str(rel)
        except Exception:
            return ""

    def reset(self, seed: int = 0, trajectory_id: Optional[str] = None,
              run_id: str = ""):
        obs, info = self.env.reset(seed=seed)
        self.history = []
        self.logger.records.clear()   # one JSONL per trajectory, not cumulative
        self.step_id = 0
        self.trajectory_id = trajectory_id or f"{self.task_id}_seed{seed}"
        self.run_id = run_id
        view = to_obs_view(obs)
        self.goal = obs.get("goal", "") if isinstance(obs, dict) else ""
        self._last_obs_view = view
        shot_path = self._save_screenshot(obs)
        # step 0 = the initial observation (action=None). Multi-turn sample
        # assembly needs the obs BEFORE the first action; without this record
        # a trajectory's first turn had no observation to render.
        self.logger.add(
            StepRecord(
                task_id=self.task_id, site=self.site,
                trajectory_id=self.trajectory_id, step_id=0, action=None,
                url_before="", url_after=view.get("url", ""), reward=0.0,
                terminated=False, truncated=False,
                obs_after_axtree=prune_axtree_txt(view.get("axtree_txt", "")),
                backend_after=view.get("backend_state"), replay_prefix=[],
                screenshot=shot_path, run_id=self.run_id,
            )
        )
        return obs, info, view

    def step(self, action: str):
        url_before = self._last_obs_view.get("url", "")
        obs, reward, terminated, truncated, info = self.env.step(action)
        view = to_obs_view(obs)
        self.history.append(action)
        self.step_id += 1
        shot_path = self._save_screenshot(obs)
        self.logger.add(
            StepRecord(
                task_id=self.task_id,
                site=self.site,
                trajectory_id=self.trajectory_id,
                step_id=self.step_id,
                action=action,
                url_before=url_before,
                url_after=view.get("url", ""),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                obs_after_axtree=prune_axtree_txt(view.get("axtree_txt", "")),
                backend_after=view.get("backend_state"),
                replay_prefix=list(self.history),
                screenshot=shot_path,
                run_id=self.run_id,
            )
        )
        self._last_obs_view = view
        return obs, reward, terminated, truncated, info, view

    def current_fingerprint(self) -> StateFingerprint:
        return fingerprint(self._last_obs_view)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Replay-to-state (foundation for S5 undo probing)
# --------------------------------------------------------------------------- #
@dataclass
class ReplayResult:
    ok: bool
    obs_view: dict
    fingerprint: dict
    distance: Optional[dict] = None


def replay_to_state(
    env,
    seed: int,
    replay_prefix: list[str],
    expected_fp: Optional[StateFingerprint] = None,
    tol: config.FingerprintTolerance = config.DEFAULT_TOL,
) -> ReplayResult:
    """Reset(seed) then replay the action prefix; verify we reproduce the state.

    Returns ok=False (with a distance report) if an expected fingerprint was
    given and the replayed state does not match it — such states are not
    reproducible and must be dropped before S5 probing.
    """
    obs, _ = env.reset(seed=seed)
    view = to_obs_view(obs)
    for action in replay_prefix:
        obs, _, _, _, _ = env.step(action)
        view = to_obs_view(obs)
    fp = fingerprint(view)
    if expected_fp is None:
        return ReplayResult(ok=True, obs_view=view, fingerprint=fp.to_dict())
    ok = is_restored(expected_fp, fp, tol)
    return ReplayResult(
        ok=ok,
        obs_view=view,
        fingerprint=fp.to_dict(),
        distance=state_distance(expected_fp, fp),
    )
