"""Read-only loaders + lineage joins over the existing pipeline artifacts.

Nothing here writes to ``data/``. The join keys are the pipeline's natural
keys (see ``revact/data/assemble.py``):

  reached state ``name``  --(x variant)-->  sft ``sample_id`` = name__variant
  sft sample_id           --(x pair_type)-> dpo ``pair_id``  = sample_id__pair_type
  state risky_action text --ACTION_KW-->    grounded ``action_type`` (latest
                                            non-UNKNOWN row wins, dry-run-safe)
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
    "多轮文件（revact_sft_multiturn.jsonl）：一条完整轨迹为一个样本，system + "
    "交替 user/assistant，途中步为 `<answer> 动作`，风险决策步为完整 <think> 块，"
    "loss 计在所有 assistant 轮。该格式与 rollout 循环（policies.py）共用一个"
    " prompt builder（revact/prompts.py），训练分布=部署分布。"
)

CARD_MESSAGE_FLOW = [
    {"role": "system", "desc": "安全 agent 指令（prompt registry: agent_system，训练与部署同一条，"
     "工作台可编辑）：动作空间 + 常规步直接 <answer>、涉状态变更步先 <think> 七字段再 <answer>"},
    {"role": "user", "desc": "首轮：<goal>\\n{目标}\\n\\n<history>\\n{编号行：action -> [flag] delta"
     "，flag ∈ nav/state-change/update/no-effect |(none)}\\n\\n<observation>\\n{axtree 快照}；"
     "多轮样本的后续 user 轮只带新 <observation>"},
    {"role": "assistant", "desc": "监督目标：常规步 `<answer> 动作`；风险步 <think><observation>.."
     "<reasoning>..<prediction>..<rev_check>..<reversibility>..<undo>..<decision>..</think>"
     "<answer>.."},
]

CARD_SFT_SCHEMA = [
    ("sample_id", "str", "state 名 + '__' + 变体（constraint|request）；与 DPO pair_id、"
     "lineage 的自然连接键"),
    ("messages", "list", "Qwen chat 序列：单步=3 条；多轮=system + 交替 user/assistant"),
    ("messages[0].content", "str", "system：安全 agent 指令（registry agent_system，可编辑）"),
    ("messages[1].content", "str", "user 首轮：<goal> + <history>（编号的 动作+[变化标记]+关键delta "
     "行）+ <observation>"),
    ("messages[-1].content", "str", "assistant 决策轮：结构化 <think> + <answer>"),
    ("meta", "dict", "审计/切分字段，训练时不喂入模型（见 meta 表）"),
]

CARD_META_SCHEMA = [
    ("action_type", "风险动作类；绑定 grounded 可逆性标签的键"),
    ("site", "站点（shopping / reddit / …），cross_site split 依据"),
    ("reversibility", "行为实测可逆性标签（S5 探针，非 LLM/人工意见）"),
    ("decision", "oracle 决策：EXECUTE / VERIFY / CONFIRM / AVOID"),
    ("variant", "constraint（目标禁止该动作）| request（目标请求该动作）"),
    ("constraint_style", "explicit_constraint / implicit_constraint / request"),
    ("goal_template", "目标模板 id（审计：防止决策可由单一表面形式预测）"),
    ("reversibility_grounded", "恒 true：标签来自探针实测而非规则"),
    ("undo_steps", "实测 undo 步数（探针 undo 控制器实录；无实测为 null）"),
    ("prose_source", "template | teacher（蒸馏只升级措辞，结论 pin 死）"),
    ("history_source", "plan（记录的 reach 计划）| canonical（按动作类合成）| none | "
     "trajectory（多轮：真实轨迹）"),
    ("risky_raw_action", "该状态风险控件的可执行动作（GRPO 约束违反奖励用）"),
    ("format", "样本格式版本（iris.v2 = 七字段 think 块 + 三要素历史行）"),
    ("prompts_fp", "生成该样本时生效 prompt 集的指纹（prompt 调优溯源）"),
]

CARD_ASSISTANT_FORMAT = [
    ("<observation>", "当前处境复述（含目标）"),
    ("<reasoning>", "以可逆性结论为条件的决策推理"),
    ("<prediction>", "动作后果预测（前向世界模型字段）"),
    ("<rev_check>", "逆向世界模型 reasoning：在标签之前检查站内 undo 机制是否存在"
     "（事实由探针 undo 实录钉死，teacher 只改措辞）"),
    ("<reversibility>", "REVERSIBLE / PARTIALLY_RECOVERABLE / IRREVERSIBLE / UNKNOWN"),
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
        meta = next((m for m in self.trajectories_meta()
                     if m.get("trajectory_id") == tid), {})
        key_states = [k for k in self.key_states() if k["trajectory_id"] == tid]
        return {"trajectory_id": tid, "meta": meta, "steps": steps,
                "key_states": key_states}

    def key_states(self) -> list[dict]:
        rows = _jsonl(self.root / "raw" / "state_bank" / f"{config.SITE}_key_states.jsonl")
        out = []
        for r in rows:
            out.append({
                "state_id": r.get("state_id"), "task_id": r.get("task_id"),
                "trajectory_id": r.get("trajectory_id"), "step_id": r.get("step_id"),
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
        return load_reversibility(self.root / "grounded" / "reversibility.jsonl")

    def manifest(self) -> list[dict]:
        return _jsonl(self.root / "grounded" / "MANIFEST.jsonl")

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
    def sft(self, distilled: bool = False) -> list[dict]:
        name = "revact_sft_distilled.jsonl" if distilled else "revact_sft.jsonl"
        rows = _jsonl(self.root / "train" / "sft" / name)
        test_ids = {r.get("sample_id")
                    for r in _jsonl(self.root / "train" / "splits" / "sft_test.jsonl")}
        out = []
        for r in rows:
            goal, obs, hist = _goal_obs(r["messages"][1]["content"])
            m = r.get("meta", {})
            asst = r["messages"][-1]["content"]
            out.append({
                "sample_id": r.get("sample_id", ""),
                "action_type": m.get("action_type", ""), "variant": m.get("variant", ""),
                "constraint_style": m.get("constraint_style", ""),
                "goal_template": m.get("goal_template", ""),
                "decision": m.get("decision", ""), "reversibility": m.get("reversibility", ""),
                "prose_source": m.get("prose_source", "template"),
                "history": hist, "history_source": m.get("history_source", ""),
                "split": "test" if r.get("sample_id") in test_ids else "train",
                "goal": goal, "obs": _clip(obs, 4000),
                "assistant": asst, "answer": _answer_of(asst),
                "observation": _field_of(asst, "observation"),
                "reasoning": _field_of(asst, "reasoning"),
                "prediction": _field_of(asst, "prediction"),
            })
        return out

    def dpo(self) -> list[dict]:
        out = []
        for r in _jsonl(self.root / "train" / "dpo" / "revact_dpo.jsonl"):
            goal = _goal_obs(r["prompt"][1]["content"])[0] if r.get("prompt") else ""
            m = r.get("meta", {})
            out.append({
                "pair_id": r.get("pair_id", ""), "pair_type": m.get("pair_type", ""),
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
        d = self.root / "train" / "splits"
        return {name: len(_jsonl(d / f"{name}.jsonl"))
                for name in ("sft_train", "sft_test", "dpo_train")}

    def sample_raw(self, sample_id: str) -> dict | None:
        """Full, unclipped JSONL rows behind one sample_id: the template SFT
        row, the distilled row (if any), and every DPO pair derived from it."""
        def find(name: str) -> dict | None:
            return next((r for r in _jsonl(self.root / "train" / "sft" / name)
                         if r.get("sample_id") == sample_id), None)
        row = find("revact_sft.jsonl")
        if row is None:
            return None
        distilled = find("revact_sft_distilled.jsonl")
        dpo = [r for r in _jsonl(self.root / "train" / "dpo" / "revact_dpo.jsonl")
               if r.get("pair_id", "").startswith(sample_id + "__")]
        test_ids = {r.get("sample_id") for r in
                    _jsonl(self.root / "train" / "splits" / "sft_test.jsonl")}
        return {"sample_id": sample_id,
                "split": "test" if sample_id in test_ids else "train",
                "sft": row, "distilled": distilled, "dpo": dpo,
                "n_chars": {m["role"]: len(m.get("content", ""))
                            for m in row.get("messages", [])}}

    def dataset_card(self) -> dict:
        """HF-dataset-card-style structured description of the training files:
        sample anatomy, field schema, and live counts/length stats."""
        lens: dict[str, list[int]] = {"user": [], "assistant": []}
        for r in _jsonl(self.root / "train" / "sft" / "revact_sft.jsonl"):
            for m in r.get("messages", []):
                if m.get("role") in lens:
                    lens[m["role"]].append(len(m.get("content", "")))
        length_stats = {
            role: ({"n": len(v), "min": min(v), "avg": round(sum(v) / len(v)),
                    "max": max(v)} if v else {"n": 0})
            for role, v in lens.items()}
        return {
            "summary": self.summary(),
            "granularity": CARD_GRANULARITY,
            "system_prompt": prompts.get("agent_system"),
            "prompts_fingerprint": prompts.fingerprint(),
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
        """Candidate/counterfactual actions derivable from pipeline data:
        expert risky action, safe alternative, and the four DPO rejected
        variants (assemble's real builders, evaluated in-process)."""
        raw = self._raw_reached(state_name)
        if raw is None:
            return None
        st = next(s for s in self.reached_states() if s["name"] == state_name)
        at = st["grounded_action_type"]
        labels = self.effective_labels()
        cands = [{
            "kind": "expert_risky", "source": "reach/scale",
            "text": st["risky_text"], "raw_action": st["risky_raw_action"],
            "reversibility": labels.get(at) if at else None, "grounded": bool(at),
        }, {
            "kind": "safe_alternative", "source": "reach/scale",
            "text": "safe alternative (rule)", "raw_action": st["safe_answer"],
            "reversibility": None, "grounded": False,
        }]
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
        return {"state": state_name, "action_type": at,
                "candidates": cands, "counterfactuals": counterfactuals}

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
        sample = next((s for s in self.sft() if s["sample_id"] == sample_id), None)
        if sample is None:
            return None
        state_name = sample_id.rsplit("__", 1)[0]
        state = next((s for s in self.reached_states() if s["name"] == state_name), None)
        at = sample["action_type"]
        grounded = [g for g in self.grounded_runs() if g["action_type"] == at]
        pairs = [p for p in self.dpo() if p["pair_id"].startswith(sample_id + "__")]
        distilled = next((d for d in self.sft(distilled=True)
                          if d["sample_id"] == sample_id), None)
        related_ks = [k for k in self.key_states()
                      if at in k.get("afforded_action_types", [])][:8]
        return {"sample": sample, "state": state, "grounded_runs": grounded,
                "effective_label": self.effective_labels().get(at),
                "dpo_pairs": pairs, "distilled": distilled,
                "related_key_states": related_ks}

    # ------------------------------------------------------------ summary -- #
    def summary(self) -> dict:
        trajs = self.trajectory_index()
        sft = self.sft()
        labels = self.effective_labels()
        return {
            "site": config.SITE, "data_root": str(self.root),
            "n_traj": len(trajs), "n_traj_success": sum(t["success"] for t in trajs),
            "n_key_states": len(self.key_states()),
            "n_reached_states": len(self.reached_states()),
            "n_grounded_runs": len(self.grounded_runs()),
            "n_grounded_classes": len(labels),
            "effective_labels": labels,
            "n_sft": len(sft), "n_dpo": len(self.dpo()),
            "n_distilled": len(self.sft(distilled=True)),
            "splits": self.splits_report(),
        }
