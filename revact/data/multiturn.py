"""P1: multi-turn trajectory samples (a full trajectory = one chat sample).

Conversation layout (see revact/prompts.py):

    system   prompts.SYSTEM
    user     render_user(goal, history=<older steps>, obs_before_first_kept_step)
    asst     "<answer> a_1"                    (routine step)
    user     render_followup(obs_1)
    asst     "<answer> a_2"
    ...
    user     render_followup(obs_k)            (the risk-affording page)
    asst     full <think> block + <answer>     (grounded label + oracle decision)

Sources are the REAL pipeline artifacts: raw step trajectories (harness logs a
step-0 record with the initial observation), key states (which step affords
which pilot action type), and grounded reversibility labels. Goals are
injected per (key state, variant) exactly like assemble (constraint/request),
so the decision label stays oracle-clean; the trajectory's own goal is NOT
used as a label source.

Long trajectories keep the last MAX_LIVE_TURNS steps as real turns and fold
older steps into the first user turn's <history> block — mirroring the rollout
policy's max_history compaction, so training and deployment see the same shape.

Loss: train.sft masks everything except assistant turns (all of them).
"""
from __future__ import annotations

import json
from pathlib import Path

from .. import config, prompts
from ..envs.obs_utils import find_bid_by_text
from .assemble import (
    ACTION_KW,
    ACTION_META,
    SYSTEM,
    _dpo_pairs_for,
    build_fields,
    build_goal,
    render_assistant,
    site_of,
)

MAX_LIVE_TURNS = 5          # real user/assistant turn pairs before the decision
OBS_CHARS = 3000            # per-turn observation budget (multi-turn is long)


def _clip(text: str, n: int = OBS_CHARS) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n] + "\n… (truncated)"


def _load_steps(path: Path) -> list[dict]:
    steps = [json.loads(ln) for ln in path.open(encoding="utf-8") if ln.strip()]
    steps.sort(key=lambda s: s.get("step_id", 0))
    return steps


def _one_line(step: dict) -> str:
    return (step.get("url_after") or "")[:120]


def _risky_action_at(obs_txt: str, action_type: str) -> dict | None:
    """Locate the risky control on the decision page (real bid, real action)."""
    kw = ACTION_KW.get(action_type)
    if not kw:
        return None
    el = find_bid_by_text({"axtree_txt": obs_txt}, [kw])
    if not el:
        return None
    return {"text": el["line"], "bid": el["bid"],
            "raw_action": f"click('{el['bid']}')", "kind": "click"}


def build_conversation(steps: list[dict], k: int, goal: str) -> list[dict] | None:
    """Messages up to (and including) the user turn showing obs at step k.

    steps[0] must be the step-0 record (initial observation). Returns None when
    the trajectory lacks it (pre-P0 logs)."""
    if not steps or steps[0].get("step_id") != 0:
        return None
    live_from = max(0, k - MAX_LIVE_TURNS)
    older = [{"action": s.get("action", ""), "obs": _one_line(s)}
             for s in steps[1:live_from + 1] if s.get("action")]
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompts.render_user(
                goal, _clip(steps[live_from].get("obs_after_axtree", "")), older)}]
    for s in steps[live_from + 1:k + 1]:
        if not s.get("action"):
            return None                       # gap in the log -> unusable
        msgs.append({"role": "assistant", "content": f"<answer> {s['action']}"})
        msgs.append({"role": "user", "content": prompts.render_followup(
            _clip(s.get("obs_after_axtree", "")))})
    return msgs


def assemble_multiturn(traj_dir: Path, key_states_path: Path, rev: dict,
                       out_dir: Path) -> dict:
    """Emit multi-turn SFT samples + DPO pairs for every usable key state."""
    key_by_traj: dict[str, list[dict]] = {}
    if key_states_path.exists():
        for ln in key_states_path.open(encoding="utf-8"):
            r = json.loads(ln)
            key_by_traj.setdefault(r["trajectory_id"], []).append(r)

    sft, dpo, skipped = [], [], []
    for tid, kss in sorted(key_by_traj.items()):
        traj_path = traj_dir / f"{tid}.jsonl"
        if not traj_path.exists():
            continue
        steps = _load_steps(traj_path)
        # deepest key state per action type whose snapshot still SHOWS the
        # risky control (pruned axtrees can truncate it away) -> richest
        # usable history without losing the sample
        by_type: dict[str, list[dict]] = {}
        for ks in kss:
            for at in ks.get("afforded_action_types", []):
                if at in rev and at in ACTION_META:
                    by_type.setdefault(at, []).append(ks)
        for at, cands in sorted(by_type.items()):
            ks = risky = None
            for cand in sorted(cands, key=lambda x: -x["step_id"]):
                obs = next((s.get("obs_after_axtree", "") for s in steps
                            if s.get("step_id") == cand["step_id"]), "")
                risky = _risky_action_at(obs, at)
                if risky is not None:
                    ks = cand
                    break
            if ks is None:
                skipped.append(f"{tid}/{at}: risky control not found at any "
                               f"of {len(cands)} key state(s)")
                continue
            k = ks["step_id"]
            state = {"name": f"{tid}_s{k}", "risky_action": risky,
                     "safe_answer": "go_back()", "url": ks.get("url", "")}
            for vname, violates, requested in [("constraint", True, False),
                                               ("request", False, True)]:
                g = build_goal(at, vname, state["name"])
                msgs = build_conversation(steps, k, g["goal"])
                if msgs is None:
                    skipped.append(f"{tid}/s{k}/{at}: no step-0 record")
                    break
                f = build_fields(state, at, rev[at], g["goal"], violates, requested)
                chosen = render_assistant(f)
                sample_id = f"mt__{tid}__s{k}__{at}__{vname}"
                sft.append({
                    "sample_id": sample_id,
                    "messages": msgs + [{"role": "assistant", "content": chosen}],
                    "meta": {"kind": "multiturn", "action_type": at,
                             "site": site_of(at), "reversibility": rev[at],
                             "decision": f["decision"], "variant": vname,
                             "constraint_style": g["style"],
                             "goal_template": g["template_id"],
                             "reversibility_grounded": True,
                             "history_source": "trajectory",
                             "risky_raw_action": risky["raw_action"],
                             "trajectory_id": tid, "decision_step": k,
                             "n_turns": (len(msgs) + 1) // 2},
                })
                for pair_type, rejected in _dpo_pairs_for(f, state, violates, requested):
                    dpo.append({
                        "pair_id": f"{sample_id}__{pair_type}",
                        "prompt": msgs, "chosen": chosen, "rejected": rejected,
                        "meta": {"kind": "multiturn", "action_type": at,
                                 "site": site_of(at), "reversibility": rev[at],
                                 "variant": vname, "pair_type": pair_type,
                                 "constraint_style": g["style"],
                                 "history_source": "trajectory",
                                 "risky_raw_action": risky["raw_action"]},
                    })

    sft_path = out_dir / "train" / "sft" / "revact_sft_multiturn.jsonl"
    dpo_path = out_dir / "train" / "dpo" / "revact_dpo_multiturn.jsonl"
    for path, rows in [(sft_path, sft), (dpo_path, dpo)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"n_sft": len(sft), "n_dpo": len(dpo), "n_skipped": len(skipped),
            "skipped": skipped[:10], "sft_path": str(sft_path),
            "dpo_path": str(dpo_path)}


def run(out_dir: Path | None = None) -> dict:
    from ..grounding.base import load_reversibility

    root = Path(out_dir) if out_dir else config.DATA_ROOT
    rev = load_reversibility(root / "grounded" / "reversibility.jsonl")
    return assemble_multiturn(
        root / "raw" / "trajectories",
        root / "raw" / "state_bank" / f"{config.SITE}_key_states.jsonl",
        rev, root)
