"""Trajectory-conditioned, stateless policy samples.

Every new formal sample has exactly the same input topology as collection and
IrisPolicy deployment (see ``prompts.build_policy_messages``):

    system   prompts.agent_system
    user     <goal> + last K real history deltas + current <observation>
    asst     full <think> block + <answer>

Sources are the REAL pipeline artifacts: raw step trajectories (harness logs a
step-0 record with the initial observation), key states (which step affords
which pilot action type), and grounded reversibility labels. Goals are
injected per (key state, variant) exactly like assemble (constraint/request),
so the decision label stays oracle-clean; the trajectory's own goal is NOT
used as a label source.

The historical chat-shaped materializer used a different topology and a
different history budget from deployment.  It also supervised routine actions
from failed collector traces.  Formal mode now accepts only non-mock,
unambiguously successful trajectories and uses the shared K-step serializer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .. import config, prompts
from ..eval.truth import (EvaluationTruthRecord, assert_truth_manifest_integrity,
                          load_truth_records, truth_by_point_variant)
from ..envs.obs_utils import find_bid_by_text, history_entry
from ..grounding.schema import GroundingPoint, load_probe_points
from ..train.validators import parse_action
from .assemble import (
    ACTION_KW,
    ACTION_META,
    _dpo_pairs_for,
    build_fields,
    build_goal,
    render_assistant,
    site_of,
    _point_info,
)

def _load_steps(path: Path) -> list[dict]:
    steps = [json.loads(ln) for ln in path.open(encoding="utf-8") if ln.strip()]
    steps.sort(key=lambda s: s.get("step_id", 0))
    return steps


def _step_view(step: dict) -> dict:
    """Obs view reconstructed from a logged step (for delta computation)."""
    return {"url": step.get("url_after", ""),
            "axtree_txt": step.get("obs_after_axtree", "")}


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


def trajectory_history(steps: list[dict], k: int) -> list[dict] | None:
    """Real observed-delta history for actions 1..k, or ``None`` on any gap."""
    by_id = {s.get("step_id"): s for s in steps}
    if 0 not in by_id or k not in by_id:
        return None
    out: list[dict] = []
    for i in range(1, k + 1):
        previous, current = by_id.get(i - 1), by_id.get(i)
        if previous is None or current is None or not current.get("action"):
            return None
        out.append(history_entry(current["action"], _step_view(previous),
                                 _step_view(current)))
    return out


def build_conversation(steps: list[dict], k: int, goal: str, *,
                       required_actions: list[str] | None = None,
                       max_history: int | None = None) -> list[dict] | None:
    """Canonical stateless messages for the decision observation at step ``k``.

    Despite the compatibility name, this no longer emits an alternating chat.
    It calls the exact serializer used by ``LLMActionPolicy``/``IrisPolicy``.
    ``required_actions`` activates the supervised-bid hard gate.
    """
    by_id = {s.get("step_id"): s for s in steps}
    history = trajectory_history(steps, k)
    if history is None:
        return None
    return prompts.build_policy_messages(
        goal,
        by_id[k].get("obs_after_axtree", ""),
        history,
        system_prompt=prompts.get("agent_system"),
        max_history=max_history,
        required_actions=required_actions,
    )


def _trajectory_origin(tid: str, records: list[dict]) -> tuple[str, bool]:
    task_ids = {str(r.get("task_id", "")) for r in records}
    is_mock = tid.startswith("mock.") or any(t.startswith("mock.") for t in task_ids)
    if is_mock:
        return "mock", True
    if tid.startswith("webarena.") or any(t.startswith("webarena.") for t in task_ids):
        return "webarena", False
    return "unknown", False


def assemble_multiturn(traj_dir: Path, key_states_path: Path, rev: dict,
                       out_dir: Path, *, formal: bool = True,
                       truth_records: dict[tuple[str, str],
                                           EvaluationTruthRecord] | None = None) -> dict:
    """Emit trajectory-conditioned SFT/DPO.

    ``formal=True`` is deliberately conservative: mock trajectories, failed
    trajectories, missing success metadata, and duplicated trajectory IDs with
    conflicting success values are quarantined.  ``formal=False`` remains an
    explicit development-only escape hatch and marks all emitted rows so they
    cannot be mistaken for the formal dataset.
    """
    key_by_traj: dict[str, list[dict]] = {}
    if key_states_path.exists():
        for ln in key_states_path.open(encoding="utf-8"):
            r = json.loads(ln)
            key_by_traj.setdefault(r["trajectory_id"], []).append(r)

    sft, dpo, skipped = [], [], []
    truth_records = truth_records or {}
    excluded = {"mock": 0, "failed": 0, "missing_success": 0,
                "ambiguous_success": 0, "missing_point": 0,
                "point_mismatch": 0}
    prompt_provenance = prompts.snapshot_generation(
        root=out_dir, author="assemble-multiturn",
        producer="revact.data.multiturn",
        model={"provider": "local", "name": "deterministic-template",
               "revision": "iris-fields-v1"},
        decode_config={
            "strategy": "deterministic", "sampling": False,
            "format": ("iris.v3" if formal else "iris.v2-legacy"),
            "message_topology": "stateless",
            "policy_history_steps": config.POLICY_HISTORY_STEPS,
            "snapshot_max_chars": config.MAX_AXTREE_CHARS_SNAPSHOT,
        })
    prompts_fp = prompt_provenance["prompts_fp"]
    prompt_generation_fp = prompt_provenance["prompt_generation_fp"]
    for tid, kss in sorted(key_by_traj.items()):
        environment_origin, is_mock = _trajectory_origin(tid, kss)
        statuses = {r.get("traj_success") for r in kss
                    if r.get("traj_success") is not None}
        if formal and is_mock:
            excluded["mock"] += 1
            skipped.append(f"{tid}: mock trajectory excluded from formal data")
            continue
        if formal and not statuses:
            excluded["missing_success"] += 1
            skipped.append(f"{tid}: collector success metadata missing")
            continue
        if formal and len(statuses) > 1:
            excluded["ambiguous_success"] += 1
            skipped.append(f"{tid}: conflicting collector success records")
            continue
        if formal and statuses != {True}:
            excluded["failed"] += 1
            skipped.append(f"{tid}: failed collector trajectory excluded")
            continue
        collector_success = statuses == {True}
        traj_path = traj_dir / f"{tid}.jsonl"
        if not traj_path.exists():
            continue
        steps = _load_steps(traj_path)
        # deepest key state per action type whose snapshot still SHOWS the
        # risky control (pruned axtrees can truncate it away) -> richest
        # usable history without losing the sample
        by_type: dict[str, list[dict]] = {}
        for ks in kss:
            if formal:
                point_id = ks.get("probe_point_id")
                point = rev.get(point_id) if point_id else None
                if not isinstance(point, GroundingPoint):
                    excluded["missing_point"] += 1
                    skipped.append(
                        f"{tid}/s{ks.get('step_id')}: missing formal probe_point_id")
                    continue
                state_id = ks.get("state_id")
                if not state_id or state_id != point.state_id or point.is_mock:
                    excluded["point_mismatch"] += 1
                    skipped.append(
                        f"{tid}/s{ks.get('step_id')}: point state/mock mismatch")
                    continue
                candidate = dict(ks)
                candidate["_point"] = point
                if point.action_type in ACTION_META:
                    by_type.setdefault(point.action_type, []).append(candidate)
            else:
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
            point = ks.get("_point") if formal else None
            if formal:
                action_instance_id = (ks.get("action_instance_id") or
                                      (ks.get("risky_action") or {}).get(
                                          "action_instance_id"))
                canonical_action = (ks.get("canonical_action") or
                                    (ks.get("risky_action") or {}).get(
                                        "canonical_action"))
                state_hash = (ks.get("pre_observation_hash") or
                              (ks.get("pre_fingerprint") or {}).get("axtree_hash"))
                mismatches = []
                if risky["raw_action"] != point.raw_action:
                    mismatches.append("raw_action")
                if action_instance_id != point.action_instance_id:
                    mismatches.append("action_instance_id")
                if canonical_action != point.canonical_action:
                    mismatches.append("canonical_action")
                if state_hash != point.pre_observation_hash:
                    mismatches.append("pre_observation_hash")
                if mismatches:
                    excluded["point_mismatch"] += 1
                    skipped.append(
                        f"{tid}/s{k}/{at}: formal point mismatch {mismatches}")
                    continue
                info = _point_info(point)
            else:
                normative_risk = None
                info = rev[at] if isinstance(rev[at], dict) else {"label": rev[at]}
            state = {"name": f"{tid}_s{k}", "risky_action": risky,
                     "safe_answer": "go_back()", "url": ks.get("url", "")}
            for vname, violates, requested in [("constraint", True, False),
                                               ("request", False, True)]:
                truth_record = (truth_records.get((point.probe_point_id, vname))
                                if point else None)
                if formal and truth_record is None:
                    excluded["point_mismatch"] += 1
                    skipped.append(
                        f"{tid}/s{k}/{at}: missing evaluation truth for {vname}")
                    continue
                normative_risk = (truth_record.normative_risk
                                  if truth_record else None)
                if formal and (
                        truth_record.violates_constraint != violates or
                        truth_record.action_required_for_goal != requested):
                    excluded["point_mismatch"] += 1
                    skipped.append(
                        f"{tid}/s{k}/{at}: evaluation truth variant mismatch")
                    continue
                g = build_goal(at, vname, state["name"])
                f = build_fields(
                    state, at, info, g["goal"], violates, requested,
                    normative_risk=normative_risk)
                try:
                    msgs = build_conversation(
                        steps, k, g["goal"],
                        required_actions=[risky["raw_action"], f["answer"]],
                    )
                except ValueError as exc:
                    skipped.append(f"{tid}/s{k}/{at}: {exc}")
                    break
                if msgs is None:
                    skipped.append(f"{tid}/s{k}/{at}: incomplete trajectory history")
                    break
                chosen = render_assistant(f)
                policy_observation = prompts.parse_observation_message(
                    msgs[-1]["content"])
                policy_input_observation_hash = hashlib.sha256(
                    policy_observation.encode("utf-8")).hexdigest()
                if formal and f["decision"] != truth_record.expected_decision:
                    excluded["point_mismatch"] += 1
                    skipped.append(
                        f"{tid}/s{k}/{at}: expected decision truth mismatch")
                    continue
                sample_id = f"mt__{tid}__s{k}__{at}__{vname}"
                sft.append({
                    "sample_id": sample_id,
                    "messages": msgs + [{"role": "assistant", "content": chosen}],
                    "meta": {"kind": "multiturn", "action_type": at,
                             "site": point.site if point else (
                                 ks.get("site") or site_of(at)),
                             "environment_origin": (point.environment_origin
                                                    if point else environment_origin),
                             "environment_family": (point.environment_family
                                                    if point else environment_origin),
                             "environment_instance": (point.environment_instance
                                                      if point else ""),
                             "is_mock": is_mock,
                             "collector_success": collector_success,
                             "formal_dataset": formal,
                             "dataset_tier": ("formal_point" if formal else
                                              "legacy_quarantine"),
                             "probe_point_id": point.probe_point_id if point else "",
                             "probe_run_id": point.probe_run_id if point else "",
                             "state_id": point.state_id if point else ks.get("state_id", ""),
                             "candidate_id": point.candidate_id if point else "",
                             "action_instance_id": (point.action_instance_id
                                                    if point else ""),
                             "effect_status": point.effect_status if point else None,
                             "recovery_status": (point.recovery_status
                                                 if point else None),
                             "reversibility": info["label"],
                             "undo_cost_steps": info.get("undo_steps"),
                             "prediction_source": f["prediction_source"],
                             "undo_source": f["undo_source"],
                             "undo_source_probe_point_id":
                                 f["undo_source_probe_point_id"],
                             "pre_observation_hash": (point.pre_observation_hash
                                                      if point else ""),
                             "post_observation_hash": (point.post_observation_hash
                                                       if point else ""),
                             "post_signal_diff": ({
                                 "pre_signal": point.pre_signal,
                                 "post_signal": point.post_signal,
                             } if point else None),
                             "undo_actions": (list(point.undo_actions)
                                              if point else []),
                             "undo_semantic_actions": (
                                 list(point.undo_semantic_actions)
                                 if point else []),
                             "undo_observation_hashes": (
                                 list(point.undo_observation_hashes)
                                 if point else []),
                             "residual_diff": (point.residual_diff
                                               if point else None),
                             "budget_k": point.budget_k if point else None,
                             "solver_set": (list(point.solver_set)
                                            if point else []),
                             "candidate_snapshot_hash": (
                                 str((point.evidence or {}).get(
                                     "candidate_snapshot_hash") or "")
                                 if point else ""),
                             "policy_input_observation_hash":
                                 policy_input_observation_hash,
                             "evidence": dict(point.evidence) if point else {},
                             "normative_risk": normative_risk,
                             "policy_constraint_truth": (
                                 truth_record.policy_constraint_truth
                                 if truth_record else None),
                             "evaluation_case_id": (
                                 truth_record.evaluation_case_id
                                 if truth_record else ""),
                             "normative_truth_source": (
                                 truth_record.truth_source if truth_record else ""),
                             "normative_policy_id": (
                                 truth_record.policy_id if truth_record else ""),
                             "normative_policy_version": (
                                 truth_record.policy_version if truth_record else ""),
                             "decision": f["decision"], "variant": vname,
                             "constraint_style": g["style"],
                             "goal_template": g["template_id"],
                             "reversibility_grounded": formal,
                             "history_source": "trajectory",
                             "risky_raw_action": risky["raw_action"],
                             "risky_action": ((parsed.to_dict()
                                               if (parsed := parse_action(
                                                   risky["raw_action"])) else None)
                                              if formal else None),
                             "canonical_action": (point.canonical_action
                                                  if point else ""),
                             "legal_at_snapshot": True if formal else None,
                             "action_legal": True if formal else None,
                             "backend_commit": ((point.evidence or {}).get(
                                 "backend_commit") if point else None),
                             "violates_constraint": violates,
                             "action_required_for_goal": requested,
                             "task_id": point.task_id if point else ks.get("task_id", ""),
                             "trajectory_id": (point.trajectory_id if point else tid),
                             "run_id": point.run_id if point else ks.get("run_id", ""),
                             "seed": point.seed if point else ks.get("seed"),
                             "account": point.account if point else ks.get("account", ""),
                             "privilege": (point.privilege if point else
                                           ks.get("privilege", "")),
                             "url": point.url if point else ks.get("url", ""),
                             "canonical_entity_id": ks.get("canonical_entity_id", ""),
                             "page_template_id": ks.get("page_template_id", ""),
                             "decision_step": k,
                             "history_steps_total": k,
                             "history_steps_kept": min(k, config.POLICY_HISTORY_STEPS),
                             "message_topology": "stateless",
                             "turn_type": "decision",
                             "assistant_turn_types": ["decision"],
                             "n_turns": 1,
                             "format": ("iris.v3" if formal else
                                        "iris.v2-legacy"),
                             "prompts_fp": prompts_fp,
                             "prompt_generation_fp": prompt_generation_fp},
                })
                for pair_type, rejected in _dpo_pairs_for(f, state, violates, requested):
                    dpo.append({
                        "pair_id": f"{sample_id}__{pair_type}",
                        "prompt": msgs, "chosen": chosen, "rejected": rejected,
                        "meta": {"kind": "multiturn", "action_type": at,
                                 "site": ks.get("site") or site_of(at),
                                 "environment_origin": environment_origin,
                                 "is_mock": is_mock,
                                 "collector_success": collector_success,
                                 "formal_dataset": False,
                                 "dataset_tier": "synthetic_ablation",
                                 "turn_type": "decision",
                                 # _dpo_pairs_for only mutates the gold text;
                                 # it does not propose a distinct legal action.
                                 "negative_source": "synthetic_flip",
                                 "probe_point_id": point.probe_point_id if point else "",
                                 "state_id": point.state_id if point else "",
                                 "effect_status": point.effect_status if point else None,
                                 "recovery_status": (point.recovery_status
                                                     if point else None),
                                 "reversibility": info["label"],
                                 "variant": vname, "pair_type": pair_type,
                                 "constraint_style": g["style"],
                                 "history_source": "trajectory",
                                 "risky_raw_action": risky["raw_action"],
                                 "format": ("iris.v3" if formal else
                                            "iris.v2-legacy"),
                                 "prompts_fp": prompts_fp,
                                 "prompt_generation_fp": prompt_generation_fp},
                    })

    if formal:
        sft_path = (out_dir / "train" / "formal" /
                    "iris_sft_multiturn_point_v1.jsonl")
        dpo_path = (out_dir / "train" / "ablation" /
                    "iris_dpo_multiturn_synthetic_v1.jsonl")
    else:
        # Development escape hatches never overwrite historical/formal assets.
        sft_path = (out_dir / "train" / "quarantine" /
                    "legacy_sft_multiturn.jsonl")
        dpo_path = (out_dir / "train" / "quarantine" /
                    "legacy_dpo_multiturn.jsonl")
    for path, rows in [(sft_path, sft), (dpo_path, dpo)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"n_sft": len(sft), "n_dpo": len(dpo), "n_skipped": len(skipped),
            "excluded": excluded, "formal": formal,
            "formal_dpo_eligible": False,
            "prompts_fp": prompts_fp,
            "prompt_generation_fp": prompt_generation_fp,
            "skipped": skipped[:10], "sft_path": str(sft_path),
            "dpo_path": str(dpo_path)}


def run(out_dir: Path | None = None, *, formal: bool = True) -> dict:
    root = Path(out_dir) if out_dir else config.DATA_ROOT
    if formal:
        rev = load_probe_points(root / "grounded" / "probe_points.jsonl")
        truth_path = root / "eval" / "truth.jsonl"
        truth_manifest = root / "eval" / "TRUTH_MANIFEST.jsonl"
        truth = {}
        if truth_path.exists() and truth_manifest.exists():
            assert_truth_manifest_integrity(truth_path, truth_manifest, rev)
            truth = truth_by_point_variant(
                load_truth_records(truth_path, points=rev).values())
    else:
        from ..grounding.base import load_reversibility_details
        rev = load_reversibility_details(
            root / "grounded" / "reversibility.jsonl")
    return assemble_multiturn(
        root / "raw" / "trajectories",
        root / "raw" / "state_bank" / f"{config.SITE}_key_states.jsonl",
        rev, root, formal=formal,
        truth_records=truth if formal else None)
