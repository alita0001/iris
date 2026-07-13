"""Read-only loaders + lineage joins over the existing pipeline artifacts.

Nothing here writes to ``data/``. The join keys are the pipeline's natural
keys (see ``revact/data/assemble.py``):

  reached state ``name``  --(x variant)-->  sft ``sample_id`` = name__variant
  sft sample_id           --(x pair_type)-> dpo ``pair_id``  = sample_id__pair_type
The old ``reversibility.jsonl`` action-class table is exposed only as a
``legacy_class_smoke`` asset for inspection.  Formal supervision is exposed
separately from the exact 1:1 ``probe_points.jsonl`` / ``POINT_MANIFEST.jsonl``
pair and is never produced through an action-type fallback.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .. import config, prompts
from ..data.assemble import (
    ACTION_KW,
    ACTION_META,
    _dpo_pairs_for,
    build_fields,
    build_goal,
)
from ..data.candidates import (CandidateValidationError,
                               build_a11y_candidate_set, interactive_elements,
                               save_candidate_set)
from ..data.governance import formal_release_context
from ..envs.obs_utils import action_bid
from ..grounding.base import load_reversibility, load_reversibility_details

# --------------------------------------------------------------------------- #
# Dataset-card schema (HF dataset-card style; rendered by the browser tab).
# Descriptions mirror revact/data/assemble.py — update both when the format
# changes.
# --------------------------------------------------------------------------- #
CARD_GRANULARITY = (
    "单步文件（revact_sft.jsonl）：每条样本是一个「决策点」，输入 = system 指令 + user"
    "（<goal> + <history> 编号历史行（动作+[变化标记]+关键delta）+ <observation> 当前页 "
    "axtree 快照）；监督目标 = assistant 一条（<think> 七字段 + <answer> 动作）。"
    "轨迹条件文件（revact_sft_multiturn.jsonl）：每条仍是同样的 system + 单个 user + "
    "assistant 决策点；区别是 <history> 来自连续真实轨迹。新 formal 产物不再使用交替"
    "chat topology，并默认排除 mock 与失败 collector 轨迹。该格式与 rollout 循环"
    "（policies.py）共用 prompt builder（revact/prompts.py）。"
)

CARD_MESSAGE_FLOW = [
    {"role": "system", "desc": "安全 agent 指令（prompt registry: agent_system，训练与部署同一条，"
     "工作台可编辑）：动作空间 + 常规步直接 <answer>、涉状态变更步先 <think> 七字段再 <answer>"},
    {"role": "user", "desc": "<goal>\\n{目标}\\n\\n<history>\\n{编号行：action -> [flag] delta"
     "，flag ∈ nav/state-change/update/no-effect |(none)}\\n\\n<observation>\\n{axtree 快照}；"
     "single 与轨迹条件 family 均为一个 stateless user 输入"},
    {"role": "assistant", "desc": "监督目标：常规步 `<answer> 动作`；风险步 <think><observation>.."
     "<reasoning>..<prediction>..<rev_check>..<reversibility>..<undo>..<decision>..</think>"
     "<answer>.."},
]

CARD_SFT_SCHEMA = [
    ("sample_id", "str", "state 名 + '__' + 变体（constraint|request）；与 DPO pair_id、"
     "lineage 的自然连接键"),
    ("messages", "list", "正式产物固定3条：system + stateless user + assistant"),
    ("messages[0].content", "str", "system：安全 agent 指令（registry agent_system，可编辑）"),
    ("messages[1].content", "str", "user 首轮：<goal> + <history>（编号的 动作+[变化标记]+关键delta "
     "行）+ <observation>"),
    ("messages[-1].content", "str", "assistant 决策轮：结构化 <think> + <answer>"),
    ("meta", "dict", "审计/切分字段，训练时不喂入模型（见 meta 表）"),
]

CARD_META_SCHEMA = [
    ("probe_point_id", "正式集唯一 grounding join key；必须存在于 point body + 1:1 manifest"),
    ("action_type", "动作类（分层统计字段，不是正式标签绑定主键）"),
    ("site", "站点（shopping / reddit / …），cross_site split 依据"),
    ("effect_status / recovery_status", "正交 point-level 实测轴；负例仅表示预算内未恢复"),
    ("decision", "oracle 决策：EXECUTE / VERIFY / CONFIRM / AVOID"),
    ("variant", "constraint（目标禁止该动作）| request（目标请求该动作）"),
    ("constraint_style", "explicit_constraint / implicit_constraint / request"),
    ("goal_template", "目标模板 id（审计：防止决策可由单一表面形式预测）"),
    ("reversibility_grounded", "正式集必须为 true；legacy/quarantine 为 false 或缺失"),
    ("undo_cost_steps", "probe 实录的 nullable undo 步数；不编码进离散类别"),
    ("prose_source", "template | teacher（蒸馏只升级 observation/reasoning/prediction/rev_check 措辞，结论 pin 死）"),
    ("rev_check_source", "teacher | template（仅蒸馏样本：标记 <rev_check> 措辞是否由 teacher 写；早于 rev_check 字段的历史蒸馏产物迁移时记 template）"),
    ("history_source", "plan（记录的 reach 计划）| canonical（按动作类合成）| none | "
     "trajectory（多轮：真实轨迹）"),
    ("risky_raw_action", "该状态风险控件的可执行动作（GRPO 约束违反奖励用）"),
    ("format", "正式样本 iris.v3；iris.v2 legacy 只可浏览/非正式导出"),
    ("prompts_fp", "生成时 prompt 集的内容指纹；全文存于 manifests/prompts/{fp}.json"),
]

CARD_ASSISTANT_FORMAT = [
    ("<observation>", "当前处境复述（含目标）"),
    ("<reasoning>", "以可逆性结论为条件的决策推理"),
    ("<prediction>", "动作后果预测（前向世界模型字段）"),
    ("<rev_check>", "逆向世界模型 reasoning：在标签之前检查站内 undo 机制是否存在"
     "（事实由探针 undo 实录钉死，teacher 只改措辞）"),
    ("<reversibility>", "RECOVERED / PARTIALLY_RECOVERED / "
     "NOT_RECOVERED_WITHIN_BUDGET / UNKNOWN（NO_EFFECT 属 effect_status）"),
    ("<undo>", "undo 计划概要 + 实测步数（如 'remove the item from the cart (1 step)'）"
     "或 none available / unverified"),
    ("<decision>", "EXECUTE / VERIFY / CONFIRM / AVOID + risk 分数"),
    ("<answer>", "最终动作：raw_action / send_msg_to_user(...) / go_back()"),
]

CARD_DPO_SCHEMA = [
    ("pair_id", "str", "sample_id + '__' + pair_type"),
    ("prompt", "list[2]", "system + user，与同 sample_id 的 SFT prompt 完全一致"),
    ("chosen", "str", "正确 assistant 序列（与 SFT 监督目标相同）"),
    ("rejected", "str", "反事实 builder 生成的错误序列（标签/决策按错误方式自洽）"),
    ("meta.pair_type", "str", "false_safe / over_block / goal_violation / wrong_reversibility"),
]


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for ln in path.open(encoding="utf-8"):
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return out


def _clip(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n] + f"\n… (+{len(text) - n} chars)"


def _goal_obs(user_content: str) -> tuple[str, str, str]:
    p = prompts.parse_user(user_content)
    return p["goal"], p["obs"], p["history"]


def _answer_of(assistant: str) -> str:
    m = re.search(r"<answer>\s*(.+)", assistant)
    return m.group(1).strip() if m else ""


def _field_of(assistant: str, tag: str) -> str:
    m = re.search(rf"<{tag}>\s*([^\n]+)", assistant)
    return m.group(1).strip() if m else ""


class DataStore:
    """One instance per request family; everything loads lazily from disk so
    the workbench always reflects the current files (pipeline runs append)."""

    def __init__(self, root: Path | None = None):
        self.root = root or config.DATA_ROOT

    # ---------------------------------------------------------------- raw -- #
    def trajectories_meta(self) -> list[dict]:
        return _jsonl(self.root / "raw" / "trajectories_meta.jsonl")

    def trajectory_ids(self) -> list[str]:
        d = self.root / "raw" / "trajectories"
        return sorted(p.stem for p in d.glob("*.jsonl")) if d.exists() else []

    def trajectory_index(self) -> list[dict]:
        metas = {m.get("trajectory_id"): m for m in self.trajectories_meta()}
        out = []
        for tid in self.trajectory_ids():
            m = metas.get(tid, {})
            out.append({
                "trajectory_id": tid, "task_id": m.get("task_id", tid.split("_seed")[0]),
                "run_id": m.get("run_id", ""),
                "logical_trajectory_id": m.get("logical_trajectory_id", tid),
                "environment_origin": m.get("environment_origin", "unknown"),
                "is_mock": bool(m.get("is_mock", str(m.get("task_id", "")).startswith("mock"))),
                "collector_success": bool(m.get("collector_success", m.get("success"))),
                "success": bool(m.get("success")), "n_steps": m.get("n_steps"),
                "max_reward": m.get("max_reward", 0), "seed": m.get("seed"),
                "terminated": m.get("terminated"), "truncated": m.get("truncated"),
                "final": _clip(m.get("final_model_response", ""), 400),
                "anomalous": bool(m.get("truncated")) and not m.get("success"),
            })
        return out

    def trajectory(self, tid: str) -> dict | None:
        f = self.root / "raw" / "trajectories" / f"{tid}.jsonl"
        if not f.exists():
            return None
        steps = []
        for s in _jsonl(f):
            steps.append({
                "step_id": s.get("step_id"), "action": s.get("action", ""),
                "url_before": s.get("url_before", ""), "url_after": s.get("url_after", ""),
                "reward": s.get("reward", 0), "terminated": s.get("terminated"),
                "truncated": s.get("truncated"),
                "axtree": _clip(s.get("obs_after_axtree", ""), 4000),
                "screenshot": s.get("screenshot", ""),
            })
        matches = [m for m in self.trajectories_meta()
                   if m.get("trajectory_id") == tid]
        meta = matches[-1] if matches else {}
        key_states = [k for k in self.key_states() if k["trajectory_id"] == tid]
        return {"trajectory_id": tid, "meta": meta, "steps": steps,
                "key_states": key_states, "n_meta_attempts": len(matches)}

    def key_states(self) -> list[dict]:
        rows = _jsonl(self.root / "raw" / "state_bank" / f"{config.SITE}_key_states.jsonl")
        out = []
        for r in rows:
            out.append({
                "state_id": r.get("state_id"), "task_id": r.get("task_id"),
                "trajectory_id": r.get("trajectory_id"), "step_id": r.get("step_id"),
                "run_id": r.get("run_id", ""),
                "logical_trajectory_id": r.get("logical_trajectory_id", r.get("trajectory_id")),
                "environment_origin": r.get("environment_origin", "unknown"),
                "environment_family": r.get("environment_family", r.get("site", "")),
                "is_mock": bool(r.get("is_mock", str(r.get("task_id", "")).startswith("mock"))),
                "collector_success": bool(r.get("collector_success", r.get("traj_success"))),
                "goal": r.get("goal", ""),
                "afforded_action_types": r.get("afforded_action_types", []),
                "url": r.get("url", ""), "replay_prefix": r.get("replay_prefix", []),
                "axtree": _clip(r.get("axtree_snapshot", ""), 3000),
            })
        return out

    def reached_states(self) -> list[dict]:
        """Risk-affording states (S3 pilot + scale), deduped latest-per-name."""
        latest: dict[str, dict] = {}
        for fname, source in [("pilot_reached_states.jsonl", "pilot"),
                              ("scaled_reached_states.jsonl", "scaled")]:
            for r in _jsonl(self.root / "raw" / "state_bank" / fname):
                if not r.get("reached"):
                    continue
                r["_source"] = source
                latest[r.get("name", "")] = r
        out = []
        for name, r in sorted(latest.items()):
            risky = r.get("risky_action") or {}
            out.append({
                "name": name, "state_id": r.get("state_id", ""),
                "action_type": r.get("action_type", ""), "source": r["_source"],
                "url": r.get("url", ""), "constraint_goal": r.get("constraint_goal", ""),
                "risky_text": risky.get("text", ""),
                "risky_raw_action": risky.get("raw_action", ""),
                "safe_answer": r.get("safe_answer", ""),
                "grounded_action_type": self._bind_action_type(risky.get("text", "")),
                "axtree": _clip(r.get("axtree_snapshot", ""), 3000),
            })
        return out

    def _bind_action_type(self, risky_text: str) -> str | None:
        """Same keyword binding assemble() uses to attach a grounded label."""
        low = (risky_text or "").lower()
        return next((a for a, kw in ACTION_KW.items()
                     if a in ACTION_META and kw in low), None)

    # ----------------------------------------------------------- grounded -- #
    def grounded_runs(self) -> list[dict]:
        """Legacy class-level smoke rows; never a formal supervision source."""
        rows = _jsonl(self.root / "grounded" / "reversibility.jsonl")
        effective = self._effective_rows(rows)
        out = []
        for i, r in enumerate(rows):
            ev = dict(r.get("evidence") or {})
            sdir = ev.get("screenshots_dir", "")
            shots = []
            if sdir and (self.root / sdir).is_dir():
                shots = [str(p.relative_to(self.root))
                         for p in sorted((self.root / sdir).glob("*.png"))[:12]]
            out.append({
                "asset_tier": "legacy_class_smoke", "formal_supervision": False,
                "shots": shots,
                "row": i, "action_type": r.get("action_type", "?"),
                "label": r.get("label", "?"), "grounding": r.get("grounding", ""),
                "destructive": r.get("destructive", False),
                "commit_mode": r.get("commit_mode", False),
                "probe_id": r.get("probe_id", ""), "probe_name": r.get("probe_name", ""),
                "timestamp": r.get("timestamp", ""), "site": r.get("site", ""),
                "undo_steps": ev.get("undo_steps"),
                "undo_actions": ev.get("undo_actions", []),
                "residual_diff": ev.get("residual_diff"),
                "screenshots_dir": ev.get("screenshots_dir", ""),
                "evidence": {k: v for k, v in ev.items()
                             if k not in ("undo_actions", "screenshots_dir")},
                "effective": effective.get(r.get("action_type")) is r,
            })
        return out

    @staticmethod
    def _effective_rows(rows: list[dict]) -> dict[str, dict]:
        latest: dict[str, dict] = {}
        for r in rows:
            at = r.get("action_type", "?")
            if r.get("label") != "UNKNOWN" or at not in latest:
                latest[at] = r
        return latest

    def effective_labels(self) -> dict[str, str]:
        """Legacy latest-per-action display labels (not a formal join API)."""
        return load_reversibility(self.root / "grounded" / "reversibility.jsonl")

    def manifest(self) -> list[dict]:
        """Legacy smoke-probe manifest retained for compatibility."""
        return _jsonl(self.root / "grounded" / "MANIFEST.jsonl")

    def formal_grounding(self) -> dict:
        """Canonical formal points with exact body↔manifest integrity status."""
        context = formal_release_context(self.root)
        items = [] if context.grounding_error else [
            {**point.to_dict(), "display_label": point.display_label,
             "asset_tier": "formal_point", "formal_supervision": True}
            for point in context.points.values()
        ]
        manifest = (_jsonl(self.root / "grounded" / "POINT_MANIFEST.jsonl")
                    if not context.grounding_error else [])
        return {
            "ok": not bool(context.grounding_error),
            "error": context.grounding_error,
            "n_points": len(items),
            "n_manifest": len(manifest),
            "one_to_one": (not context.grounding_error
                           and len(items) == len(manifest)),
            "items": items,
            "manifest": manifest,
        }

    def probe_specs(self) -> list[dict]:
        from ..grounding import list_probes
        from ..grounding import probes  # noqa: F401  (registers all probes)
        labels = self.effective_labels()
        return [{
            "name": s.name, "site": s.site, "action_type": s.action_type,
            "destructive": s.destructive, "grounding": s.grounding, "undo": s.undo,
            "expected_spectrum": s.expected_spectrum,
            "live_label": labels.get(s.action_type),
        } for s in list_probes()]

    # -------------------------------------------------------------- train -- #
    def sft(self, distilled: bool = False, family: str = "single",
            tier: str = "legacy") -> list[dict]:
        """Workbench view over one explicitly selected dataset tier.

        ``formal`` and ``legacy`` are never merged implicitly.  This keeps an
        empty formal release visibly empty while retaining the frozen pilot as
        an inspectable quarantine/development asset.
        """
        if family not in ("single", "multiturn", "all"):
            raise ValueError("family must be single, multiturn, or all")
        if tier not in ("formal", "legacy"):
            raise ValueError("tier must be formal or legacy")
        if distilled and family != "single":
            return []  # no multi-turn teacher artifact exists yet
        specs = []
        if tier == "formal":
            base = self.root / "train" / "formal"
            if family in ("single", "all"):
                specs.append(("single", base / (
                    "iris_sft_distilled_point_v1.jsonl" if distilled else
                    "iris_sft_point_v1.jsonl")))
            if family in ("multiturn", "all") and not distilled:
                specs.append(("multiturn", base /
                              "iris_sft_multiturn_point_v1.jsonl"))
            split_dir = base / "splits"
        else:
            base = self.root / "train" / "sft"
            if family in ("single", "all"):
                specs.append(("single", base / (
                    "revact_sft_distilled.jsonl" if distilled else
                    "revact_sft.jsonl")))
            if family in ("multiturn", "all") and not distilled:
                specs.append(("multiturn", base /
                              "revact_sft_multiturn.jsonl"))
            split_dir = self.root / "train" / "splits"
        rows: list[tuple[str, dict]] = []
        for fam, path in specs:
            rows.extend((fam, r) for r in _jsonl(path))
        test_ids = {
            r.get("sample_id")
            for name in ("sft_test.jsonl", "sft_test_multiturn.jsonl")
            for r in _jsonl(split_dir / name)}
        dev_ids = {
            r.get("sample_id")
            for name in ("sft_dev.jsonl", "sft_dev_multiturn.jsonl")
            for r in _jsonl(split_dir / name)}
        train_ids = {
            r.get("sample_id")
            for name in ("sft_train.jsonl", "sft_train_multiturn.jsonl")
            for r in _jsonl(split_dir / name)}
        out = []
        for fam, r in rows:
            goal, obs, hist = _goal_obs(r["messages"][1]["content"])
            m = r.get("meta", {})
            asst = r["messages"][-1]["content"]
            sample_id = r.get("sample_id", "")
            memberships = [side for side, ids in (
                ("train", train_ids), ("dev", dev_ids), ("test", test_ids))
                if sample_id in ids]
            split = (memberships[0] if len(memberships) == 1 else
                     "unassigned" if not memberships else "membership_error")
            out.append({
                "sample_id": sample_id,
                "asset_tier": tier,
                "family": fam,
                "site": m.get("site", ""),
                "action_type": m.get("action_type", ""), "variant": m.get("variant", ""),
                "constraint_style": m.get("constraint_style", ""),
                "goal_template": m.get("goal_template", ""),
                "decision": m.get("decision", ""), "reversibility": m.get("reversibility", ""),
                "prose_source": m.get("prose_source", "template"),
                "history": hist, "history_source": m.get("history_source", ""),
                "trajectory_id": m.get("trajectory_id", ""),
                "run_id": m.get("run_id", ""),
                "environment_origin": m.get("environment_origin", "unknown"),
                "is_mock": m.get("is_mock"),
                "collector_success": m.get("collector_success"),
                "formal_dataset": m.get("formal_dataset"),
                "probe_point_id": m.get("probe_point_id", ""),
                "probe_run_id": m.get("probe_run_id", ""),
                "state_id": m.get("state_id", ""),
                "candidate_id": m.get("candidate_id", ""),
                "action_instance_id": m.get("action_instance_id", ""),
                "effect_status": m.get("effect_status"),
                "recovery_status": m.get("recovery_status"),
                "undo_cost_steps": m.get("undo_cost_steps"),
                "prompts_fp": m.get("prompts_fp", ""),
                "prompt_generation_fp": m.get("prompt_generation_fp", ""),
                "teacher_prompts_fp": m.get("teacher_prompts_fp", ""),
                "teacher_prompt_generation_fp": m.get(
                    "teacher_prompt_generation_fp", ""),
                "split": split,
                "goal": goal, "obs": _clip(obs, 4000),
                "assistant": asst, "answer": _answer_of(asst),
                "undo_steps": m.get("undo_steps"),
                "observation": _field_of(asst, "observation"),
                "reasoning": _field_of(asst, "reasoning"),
                "prediction": _field_of(asst, "prediction"),
                "rev_check": _field_of(asst, "rev_check"),
                "undo": _field_of(asst, "undo"),
            })
        return out

    def dpo(self, family: str = "single", tier: str = "legacy") -> list[dict]:
        if family not in ("single", "multiturn", "all"):
            raise ValueError("family must be single, multiturn, or all")
        if tier not in ("formal", "legacy"):
            raise ValueError("tier must be formal or legacy")
        names = []
        if tier == "formal":
            base = self.root / "train" / "formal"
            if family in ("single", "all"):
                names.append(("single", base / "iris_dpo_point_v1.jsonl"))
            if family in ("multiturn", "all"):
                names.append(("multiturn", base /
                              "iris_dpo_multiturn_point_v1.jsonl"))
            split_dir = base / "splits"
        else:
            base = self.root / "train" / "dpo"
            if family in ("single", "all"):
                names.append(("single", base / "revact_dpo.jsonl"))
            if family in ("multiturn", "all"):
                names.append(("multiturn", base /
                              "revact_dpo_multiturn.jsonl"))
            split_dir = self.root / "train" / "splits"
        train_pair_ids = {
            str(r.get("pair_id") or "")
            for name in ("dpo_train.jsonl", "dpo_train_multiturn.jsonl")
            for r in _jsonl(split_dir / name)}
        dev_pair_ids = {
            str(r.get("pair_id") or "")
            for name in ("dpo_dev.jsonl", "dpo_dev_multiturn.jsonl")
            for r in _jsonl(split_dir / name)}
        out = []
        rows = ((fam, r) for fam, path in names for r in _jsonl(path))
        for fam, r in rows:
            goal = _goal_obs(r["prompt"][1]["content"])[0] if r.get("prompt") else ""
            m = r.get("meta", {})
            pair_id = r.get("pair_id", "")
            memberships = [side for side, ids in (
                ("train", train_pair_ids), ("dev", dev_pair_ids))
                if pair_id in ids]
            split = (memberships[0] if len(memberships) == 1 else
                     "unassigned" if not memberships else "membership_error")
            out.append({
                "pair_id": pair_id, "pair_type": m.get("pair_type", ""),
                "family": fam, "asset_tier": tier,
                "split": split,
                "action_type": m.get("action_type", ""), "variant": m.get("variant", ""),
                "constraint_style": m.get("constraint_style", ""),
                "reversibility": m.get("reversibility", ""),
                "goal": goal, "chosen": r.get("chosen", ""), "rejected": r.get("rejected", ""),
                "chosen_answer": _answer_of(r.get("chosen", "")),
                "rejected_answer": _answer_of(r.get("rejected", "")),
                "rejected_reversibility": _field_of(r.get("rejected", ""), "reversibility"),
                "rejected_decision": _field_of(r.get("rejected", ""), "decision"),
            })
        return out

    def splits_report(self) -> dict:
        names = ("sft_train", "sft_dev", "sft_test", "dpo_train", "dpo_dev",
                 "sft_train_multiturn", "sft_dev_multiturn",
                 "sft_test_multiturn", "dpo_train_multiturn",
                 "dpo_dev_multiturn")
        return {
            tier: {name: len(_jsonl(directory / f"{name}.jsonl"))
                   for name in names}
            for tier, directory in (
                ("formal", self.root / "train" / "formal" / "splits"),
                ("legacy", self.root / "train" / "splits"),
            )
        }

    def sample_raw(self, sample_id: str, tier: str = "auto") -> dict | None:
        """Full, unclipped JSONL rows behind one sample_id: the template SFT
        row, the distilled row (if any), and every DPO pair derived from it."""
        if tier not in ("auto", "formal", "legacy"):
            raise ValueError("tier must be auto, formal, or legacy")
        tiers = ("formal", "legacy") if tier == "auto" else (tier,)
        row = None
        family = "single"
        selected = ""
        for candidate_tier in tiers:
            for candidate_family in ("single", "multiturn"):
                raw_rows = self.sft(family=candidate_family,
                                    tier=candidate_tier)
                hit = next((item for item in raw_rows
                            if item["sample_id"] == sample_id), None)
                if hit is None:
                    continue
                selected = candidate_tier
                family = candidate_family
                base = self.root / "train" / (
                    "formal" if selected == "formal" else "sft")
                name = (("iris_sft_point_v1.jsonl" if family == "single" else
                         "iris_sft_multiturn_point_v1.jsonl") if selected == "formal"
                        else ("revact_sft.jsonl" if family == "single" else
                              "revact_sft_multiturn.jsonl"))
                row = next((r for r in _jsonl(base / name)
                            if r.get("sample_id") == sample_id), None)
                break
            if row is not None:
                break
        if row is None:
            return None
        distilled = None
        if family == "single":
            distill_path = self.root / "train" / (
                "formal/iris_sft_distilled_point_v1.jsonl"
                if selected == "formal" else
                "sft/revact_sft_distilled.jsonl")
            distilled = next((r for r in _jsonl(distill_path)
                              if r.get("sample_id") == sample_id), None)
        if selected == "formal":
            dpo_path = self.root / "train" / "formal" / (
                "iris_dpo_point_v1.jsonl" if family == "single" else
                "iris_dpo_multiturn_point_v1.jsonl")
        else:
            dpo_path = self.root / "train" / "dpo" / (
                "revact_dpo.jsonl" if family == "single" else
                "revact_dpo_multiturn.jsonl")
        dpo = [r for r in _jsonl(dpo_path)
               if r.get("pair_id", "").startswith(sample_id + "__")]
        split_dir = self.root / "train" / (
            "formal/splits" if selected == "formal" else "splits")
        suffix = "" if family == "single" else "_multiturn"
        test_ids = {r.get("sample_id") for r in _jsonl(
            split_dir / f"sft_test{suffix}.jsonl")}
        dev_ids = {r.get("sample_id") for r in _jsonl(
            split_dir / f"sft_dev{suffix}.jsonl")}
        train_ids = {r.get("sample_id") for r in _jsonl(
            split_dir / f"sft_train{suffix}.jsonl")}
        memberships = [side for side, ids in (
            ("train", train_ids), ("dev", dev_ids), ("test", test_ids))
            if sample_id in ids]
        split = (memberships[0] if len(memberships) == 1 else
                 "unassigned" if not memberships else "membership_error")
        return {"sample_id": sample_id,
                "family": family, "asset_tier": selected,
                "split": split,
                "sft": row, "distilled": distilled, "dpo": dpo,
                "n_chars": {m["role"]: len(m.get("content", ""))
                            for m in row.get("messages", [])}}

    def dataset_card(self) -> dict:
        """HF-dataset-card-style structured description of the training files:
        sample anatomy, field schema, and live counts/length stats."""
        lens: dict[str, list[int]] = {}
        for family, name in (("single", "revact_sft.jsonl"),
                             ("multiturn", "revact_sft_multiturn.jsonl")):
            for r in _jsonl(self.root / "train" / "sft" / name):
                for m in r.get("messages", []):
                    if m.get("role") in ("user", "assistant"):
                        lens.setdefault(f"{family}.{m['role']}", []).append(
                            len(m.get("content", "")))
        length_stats = {
            role: ({"n": len(v), "min": min(v), "avg": round(sum(v) / len(v)),
                    "max": max(v)} if v else {"n": 0})
            for role, v in lens.items()}
        for role in ("user", "assistant"):
            values = [n for key, vals in lens.items() if key.endswith("." + role)
                      for n in vals]
            length_stats[role] = ({"n": len(values), "min": min(values),
                                   "avg": round(sum(values) / len(values)),
                                   "max": max(values)} if values else {"n": 0})
        raw_sft = [r for name in ("revact_sft.jsonl",
                                  "revact_sft_multiturn.jsonl")
                   for r in _jsonl(self.root / "train" / "sft" / name)]
        prompt_fps = sorted({str((r.get("meta") or {}).get("prompts_fp"))
                             for r in raw_sft
                             if (r.get("meta") or {}).get("prompts_fp")})
        from ..prompt_store import bundle_dir
        prompt_bundle_status = {
            fp: (bundle_dir(self.root) / f"{fp}.json").exists()
            for fp in prompt_fps}
        formal_grounding = self.formal_grounding()
        legacy_smoke = self.grounded_runs()
        return {
            "summary": self.summary(),
            "granularity": CARD_GRANULARITY,
            "system_prompt": prompts.get("agent_system"),
            "prompts_fingerprint": prompts.fingerprint(),
            "sample_prompt_fingerprints": prompt_fps,
            "prompt_bundle_status": prompt_bundle_status,
            "grounding_assets": {
                "formal_point": {
                    "ok": formal_grounding["ok"],
                    "error": formal_grounding["error"],
                    "n_points": formal_grounding["n_points"],
                    "n_manifest": formal_grounding["n_manifest"],
                    "one_to_one": formal_grounding["one_to_one"],
                    "join_key": "probe_point_id",
                },
                "legacy_class_smoke": {
                    "n_rows": len(legacy_smoke),
                    "n_manifest": len(self.manifest()),
                    "formal_supervision": False,
                    "binding": "action_type latest non-UNKNOWN (display only)",
                },
            },
            "message_flow": CARD_MESSAGE_FLOW,
            "sft_schema": CARD_SFT_SCHEMA,
            "meta_schema": CARD_META_SCHEMA,
            "assistant_format": CARD_ASSISTANT_FORMAT,
            "dpo_schema": CARD_DPO_SCHEMA,
            "length_stats": length_stats,
        }

    # ---------------------------------------------- constraint / candidate -- #
    def constraint_templates(self) -> dict:
        return {"explicit": prompts.get_list("explicit_constraint_templates"),
                "implicit": prompts.get_list("implicit_constraint_templates"),
                "request": prompts.get_list("request_templates"),
                "action_meta": ACTION_META}

    def constraint_preview(self, state_name: str) -> dict | None:
        """Deterministic goals the assembler WILL produce for this state —
        the real injection logic (assemble.build_goal), not a mock."""
        st = next((s for s in self.reached_states() if s["name"] == state_name), None)
        if st is None or not st["grounded_action_type"]:
            return None
        at = st["grounded_action_type"]
        return {"state": state_name, "action_type": at,
                "variants": {v: build_goal(at, v, state_name)
                             for v in ("constraint", "request")}}

    def candidates_for(self, state_name: str) -> dict | None:
        """S4 legal candidates plus separately marked legacy counterfactuals.

        S4 depends only on the current snapshot and expert bid.  It never reads
        a recovery/safety label.  DPO template flips are retained in a separate
        section so the UI cannot confuse them with deployment-shaped actions.
        """
        raw = self._raw_reached(state_name)
        if raw is None:
            return None
        st = next(s for s in self.reached_states() if s["name"] == state_name)
        at = st["grounded_action_type"]
        labels = self.effective_labels()
        snapshot = raw.get("axtree_snapshot", "")
        expert_bid = ((raw.get("risky_action") or {}).get("bid") or
                      action_bid(st["risky_raw_action"]) or "")
        cands, s4_error = [], ""
        try:
            objects = build_a11y_candidate_set(
                state_id=str(raw.get("state_id") or state_name),
                axtree_txt=snapshot, expert_bid=str(expert_bid))
            lines = {row["bid"]: row["line"]
                     for row in interactive_elements(snapshot)}
            cands = [{
                **candidate.to_dict(),
                "kind": candidate.category,
                "text": lines.get(candidate.bid, ""),
                "raw_action": f"click('{candidate.bid}')",
                "grounded": False,
                "category_is_proposal": True,
            } for candidate in objects]
        except CandidateValidationError as exc:
            s4_error = str(exc)
        counterfactuals = []
        if at and at in labels:
            details = load_reversibility_details(
                self.root / "grounded" / "reversibility.jsonl")
            rev_info = details.get(at, labels[at])  # dict with undo evidence,
            for vname, violates, requested in [("constraint", True, False),  # else bare label
                                               ("request", False, True)]:
                g = build_goal(at, vname, state_name)
                f = build_fields(raw, at, rev_info, g["goal"], violates, requested)
                for pair_type, rejected in _dpo_pairs_for(f, raw, violates, requested):
                    counterfactuals.append({
                        "kind": "dpo_rejected", "pair_type": pair_type, "variant": vname,
                        "raw_action": _answer_of(rejected),
                        "reversibility_claimed": _field_of(rejected, "reversibility"),
                        "decision_claimed": _field_of(rejected, "decision"),
                        "rendered": rejected,
                    })
        return {"state": state_name, "state_id": raw.get("state_id") or state_name,
                "action_type": at, "s4_status": "ready" if cands else "blocked",
                "s4_error": s4_error, "snapshot_hash": (
                    cands[0]["snapshot_hash"] if cands else ""),
                "candidates": cands, "counterfactuals": counterfactuals}

    def materialize_candidates(self, state_name: str) -> dict:
        """Persist the pure S4 set; idempotent for the same state/snapshot."""
        result = self.candidates_for(state_name)
        if result is None:
            raise CandidateValidationError(f"unknown reached state {state_name!r}")
        if result["s4_status"] != "ready":
            raise CandidateValidationError(result.get("s4_error") or "S4 blocked")
        from ..data.candidates import Candidate
        objects = [Candidate.from_dict({
            key: value for key, value in row.items()
            if key in {"schema_version", "candidate_id", "state_id", "bid",
                       "canonical_action", "category", "source",
                       "legal_at_snapshot", "proposer_model",
                       "proposer_version", "snapshot_hash"}
        }) for row in result["candidates"]]
        path = self.root / "raw" / "candidates" / "iris_candidates.v3.jsonl"
        save_candidate_set(objects, path)
        return {**result, "artifact": str(path), "n": len(objects)}

    def _raw_reached(self, state_name: str) -> dict | None:
        for fname in ("scaled_reached_states.jsonl", "pilot_reached_states.jsonl"):
            for r in _jsonl(self.root / "raw" / "state_bank" / fname):
                if r.get("name") == state_name and r.get("reached"):
                    return r
        return None

    # ------------------------------------------------------------ lineage -- #
    def lineage(self, sample_id: str) -> dict | None:
        """task -> trajectory -> key state -> state -> constraint -> candidates
        -> grounded undo label -> teacher prose, joined by natural keys."""
        sample = next((s for tier in ("formal", "legacy")
                       for s in self.sft(family="all", tier=tier)
                       if s["sample_id"] == sample_id), None)
        if sample is None:
            return None
        tier = sample["asset_tier"]
        state_name = sample_id.rsplit("__", 1)[0]
        at = sample["action_type"]
        grounded = [g for g in self.grounded_runs() if g["action_type"] == at]
        point_id = str(sample.get("probe_point_id") or "")
        formal = self.formal_grounding()
        formal_point = next((p for p in formal["items"]
                             if p["probe_point_id"] == point_id), None)
        state = next((s for s in self.reached_states()
                      if ((formal_point and s.get("state_id") ==
                           formal_point.get("state_id")) or
                          (not formal_point and s["name"] == state_name))), None)
        candidate = None
        transition = None
        if formal_point:
            candidate_id = str(formal_point.get("candidate_id") or "")
            candidate = next((row for row in _jsonl(
                self.root / "raw" / "candidates" / "iris_candidates.v3.jsonl")
                              if row.get("candidate_id") == candidate_id), None)
            transition = {
                key: formal_point.get(key) for key in (
                    "probe_point_id", "state_id", "candidate_id",
                    "action_instance_id", "raw_action", "canonical_action",
                    "pre_observation_hash", "pre_signal",
                    "post_observation_hash", "post_signal", "undo_actions",
                    "undo_observation_hashes", "final_signal", "residual_diff")}
        pairs = [p for p in self.dpo(family="all", tier=tier)
                 if p["pair_id"].startswith(sample_id + "__")]
        distilled = next((d for d in self.sft(distilled=True, tier=tier)
                          if d["sample_id"] == sample_id), None)
        trajectory_id = (sample.get("trajectory_id") or "")
        related_ks = [k for k in self.key_states()
                      if (trajectory_id and k.get("trajectory_id") == trajectory_id)
                      or (not trajectory_id and at in k.get("afforded_action_types", []))][:8]
        return {"sample": sample, "asset_tier": tier,
                "split": sample.get("split"), "state": state,
                "candidate": candidate, "transition": transition,
                "formal_grounding_point": formal_point,
                "formal_grounding_integrity": {
                    key: formal[key] for key in ("ok", "error", "one_to_one")},
                "grounded_runs": grounded,
                "legacy_display_label": self.effective_labels().get(at),
                "legacy_display_label_tier": "legacy_class_smoke",
                # Backward-compatible UI alias, explicitly tiered below.
                "effective_label": self.effective_labels().get(at),
                "effective_label_tier": "legacy_class_smoke",
                "dpo_pairs": pairs, "distilled": distilled,
                "related_key_states": related_ks}

    # ------------------------------------------------------------ summary -- #
    def summary(self) -> dict:
        trajs = self.trajectory_index()
        sft = self.sft(tier="legacy")
        sft_multi = self.sft(family="multiturn", tier="legacy")
        formal_sft = self.sft(tier="formal")
        formal_sft_multi = self.sft(family="multiturn", tier="formal")
        labels = self.effective_labels()
        formal = self.formal_grounding()
        return {
            "site": config.SITE, "data_root": str(self.root),
            "n_traj": len(trajs), "n_traj_success": sum(t["success"] for t in trajs),
            "n_trajectory_meta_rows": len(self.trajectories_meta()),
            "n_key_states": len(self.key_states()),
            "n_reached_states": len(self.reached_states()),
            # Backward-compatible names are explicitly identified as legacy.
            "n_grounded_runs": len(self.grounded_runs()),
            "n_grounded_classes": len(labels),
            "effective_labels": labels,
            "n_legacy_class_smoke_rows": len(self.grounded_runs()),
            "n_formal_probe_points": formal["n_points"],
            "n_formal_point_manifest": formal["n_manifest"],
            "formal_grounding_ok": formal["ok"],
            "formal_grounding_error": formal["error"],
            "n_sft": len(sft), "n_sft_multiturn": len(sft_multi),
            "n_sft_total": len(sft) + len(sft_multi),
            "n_dpo": len(self.dpo(tier="legacy")),
            "n_dpo_multiturn": len(self.dpo(
                family="multiturn", tier="legacy")),
            "n_distilled": len(self.sft(distilled=True, tier="legacy")),
            "n_formal_sft": len(formal_sft),
            "n_formal_sft_multiturn": len(formal_sft_multi),
            "n_formal_dpo": len(self.dpo(tier="formal")),
            "n_formal_dpo_multiturn": len(self.dpo(
                family="multiturn", tier="formal")),
            "n_formal_distilled": len(self.sft(
                distilled=True, tier="formal")),
            "splits": self.splits_report(),
        }
