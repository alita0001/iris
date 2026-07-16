"""Pipeline stage registry: the honest bridge between workbench buttons and
the existing CLI.

Every stage declares what it REALLY is:
  kind="cli"         runs an existing ``python -m revact.cli`` command as a
                     subprocess job (collect / reach / probe / assemble / ...);
  kind="inprocess"   evaluates existing pure functions (constraint preview,
                     DPO counterfactual builders, QC stats) — real logic,
                     instant, no environment needed;
  kind="placeholder" no backend implementation exists yet in the pipeline
                     (e.g. generative candidate proposal). The button exists,
                     returns a clear 'not implemented' with the extension
                     point, and the UI works through annotation overlays.

Requirements are checked before launch, never silently mocked:
  live  -> WA_SHOPPING must be set (or the action's --mock form is used);
  key   -> the relevant API key must be present in runtime config or env.

Destructive probes are deliberately NOT runnable from the workbench: they
need the CLI double gate (--commit + REVACT_ALLOW_DESTRUCTIVE=1) plus
per-batch human approval (project policy).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from . import annotations
from .datasets import DataStore
from .jobs import MANAGER, python_cli

MOCK_PROBES = ["shopping.add_to_cart", "shopping.wishlist_add",
               "shopping.newsletter_subscribe", "shopping.compare_add",
               "reddit.vote", "reddit.subscribe"]


# --------------------------------------------------------------------------- #
# Runtime config (populated via the API; keys stay in process memory)
# --------------------------------------------------------------------------- #
class RuntimeConfig:
    """Workbench settings. `secrets` maps ENV-VAR-NAME -> value and is never
    persisted; everything else may be saved to configs/workbench.local.json
    (with secret values stripped, see app.save_config)."""

    def __init__(self):
        self.settings: dict = {
            "models": {
                "policy": {"provider": "deepseek", "base_url": "", "model": "",
                           "api_key_env": "DEEPSEEK_API_KEY",
                           "temperature": 0.0, "top_p": 1.0, "max_tokens": 8192},
                "teacher": {"provider": "deepseek", "base_url": "", "model": "",
                            "api_key_env": "DEEPSEEK_API_KEY",
                            "temperature": 0.7, "top_p": 1.0, "max_tokens": 400},
                "judge": {"provider": "deepseek", "base_url": "", "model": "",
                          "api_key_env": "DEEPSEEK_API_KEY", "mode": "deepseek"},
                # Opinion ratings are a separate baseline role.  Keep the
                # route/model blank and provider generic so OpenRouter remains
                # a user selection, never a repository-wide default.
                "opinion": {"provider": "custom", "base_url": "", "model": "",
                            "api_key_env": "REVACT_OPINION_API_KEY",
                            "temperature": 0.0, "top_p": 1.0,
                            "max_tokens": 300},
            },
            "run": {"task_file": "data/raw/pilot_task_ids.json", "seeds": "0",
                    "max_steps": 25, "sample_limit": 10, "screenshots": True,
                    "split": "train", "data_root": "", "output_dir": "outputs/workbench"},
            "env": {"WA_SHOPPING": "", "WA_SHOPPING_ADMIN": "", "WA_REDDIT": ""},
        }
        self.secrets: dict[str, str] = {}

    def merged_env(self, *roles: str) -> dict[str, str]:
        """Env vars to export into a job for the given model roles."""
        out: dict[str, str] = {}
        for k in ("WA_SHOPPING", "WA_SHOPPING_ADMIN", "WA_REDDIT"):
            v = (self.settings.get("env") or {}).get(k, "")
            if v:
                out[k] = v
        models = self.settings.get("models") or {}
        for role in roles:
            m = models.get(role) or {}
            key_env = m.get("api_key_env") or ""
            if key_env and self.secrets.get(key_env):
                out[key_env] = self.secrets[key_env]
            if role == "policy":
                if m.get("provider"):
                    out["REVACT_LLM_PROVIDER"] = str(m["provider"])
                if m.get("model"):
                    out["REVACT_LLM_MODEL"] = str(m["model"])
                if m.get("base_url"):
                    out["REVACT_LLM_BASE_URL"] = str(m["base_url"])
                if m.get("max_tokens"):
                    out["REVACT_LLM_MAX_TOKENS"] = str(m["max_tokens"])
                if m.get("temperature") is not None:
                    out["REVACT_LLM_TEMPERATURE"] = str(m["temperature"])
                if m.get("top_p") is not None:
                    out["REVACT_LLM_TOP_P"] = str(m["top_p"])
            elif role == "teacher":
                if m.get("provider"):
                    out["REVACT_DISTILL_PROVIDER"] = str(m["provider"])
                if m.get("model"):
                    out["REVACT_DISTILL_MODEL"] = str(m["model"])
                if m.get("base_url"):
                    out["REVACT_DISTILL_BASE_URL"] = str(m["base_url"])
                if key_env:
                    out["REVACT_DISTILL_KEY_ENV"] = key_env
            elif role == "judge":
                if m.get("provider"):
                    out["REVACT_WA_JUDGE_PROVIDER"] = str(m["provider"])
                if m.get("base_url"):
                    out["REVACT_WA_JUDGE_BASE_URL"] = str(m["base_url"])
                if m.get("model"):
                    out["REVACT_WA_JUDGE_MODEL"] = str(m["model"])
                if key_env:
                    out["REVACT_WA_JUDGE_API_KEY_ENV"] = str(key_env)
                if m.get("mode"):
                    out["REVACT_WA_JUDGE"] = str(m["mode"])
            elif role == "opinion":
                if m.get("provider"):
                    out["REVACT_OPINION_PROVIDER"] = str(m["provider"])
                if m.get("base_url"):
                    out["REVACT_OPINION_BASE_URL"] = str(m["base_url"])
                if m.get("model"):
                    out["REVACT_OPINION_MODEL"] = str(m["model"])
                if key_env:
                    out["REVACT_OPINION_KEY_ENV"] = str(key_env)
                if m.get("temperature") is not None:
                    out["REVACT_OPINION_TEMPERATURE"] = str(m["temperature"])
                if m.get("top_p") is not None:
                    out["REVACT_OPINION_TOP_P"] = str(m["top_p"])
                if m.get("max_tokens") is not None:
                    out["REVACT_OPINION_MAX_TOKENS"] = str(m["max_tokens"])
        drt = (self.settings.get("run") or {}).get("data_root", "")
        if drt:
            out["REVACT_DATA_ROOT"] = drt
        return out

    def has_key(self, role: str) -> bool:
        m = (self.settings.get("models") or {}).get(role) or {}
        key_env = m.get("api_key_env") or ""
        return bool(key_env and (self.secrets.get(key_env) or os.environ.get(key_env)))


RUNTIME = RuntimeConfig()


def live_ready(cfg: RuntimeConfig | None = None, site: str = "shopping") -> bool:
    """Whether a given site's base URL is configured (in runtime cfg or env)."""
    cfg = cfg or RUNTIME
    spec = config.SITES.get(site)
    env_var = spec.base_env if spec else "WA_SHOPPING"
    return bool((cfg.settings.get("env") or {}).get(env_var)
                or config.site_base(site))


# --------------------------------------------------------------------------- #
# Stage specs
# --------------------------------------------------------------------------- #
@dataclass
class Action:
    id: str
    label: str
    kind: str                      # cli | inprocess | placeholder
    needs: list[str] = field(default_factory=list)   # live | key:<role> | mock
    description: str = ""


@dataclass
class Stage:
    id: str
    title: str
    s_ref: str                     # which S-stage of doc 05 this maps to
    description: str
    actions: list[Action]
    artifacts: list[str]           # data-root-relative paths (globs allowed)
    implemented: str               # real | partial | placeholder


STAGES: list[Stage] = [
    Stage("env_init", "Task / Env 初始化", "S1",
          "WebArena/BrowserGym 环境与任务清单就绪检查（离线可跑 mock）。",
          [Action("check", "环境就绪检查", "inprocess",
                  description="检查 WA_* 环境变量、任务清单与 product_urls")],
          ["raw/pilot_task_ids.json", "raw/product_urls.json"], "real"),
    Stage("collect", "成功轨迹采集", "S2",
          "策略 rollout + step 级日志 + 关键状态挖掘（collect）。",
          [Action("collect_mock", "mock 演练采集", "cli", ["mock"],
                  "MockShoppingEnv 离线演练，验证链路"),
           Action("collect_live", "live 采集", "cli", ["live", "key:policy"],
                  "DeepSeek/自选策略模型在 WebArena 上采集轨迹"),
           Action("inspect", "质检统计", "cli", [], "trajectories/key-states 统计")],
          ["raw/trajectories/*.jsonl", "raw/trajectories_meta.jsonl"], "real"),
    Stage("key_states", "关键状态采集", "S2/S3",
          "从轨迹挖掘 key states（collect 副产物）+ 确定性直达风险状态（reach/scale）。",
          [Action("reach", "reach 直达风险状态", "cli", ["live"]),
           Action("crawl", "crawl 商品 URL", "cli", ["live"]),
           Action("scale", "scale 多商品状态", "cli", ["live"])],
          ["raw/state_bank/*_key_states.jsonl", "raw/state_bank/*reached_states.jsonl"],
          "real"),
    Stage("constraints", "约束注入", "S3/S8",
          "11 显式 + 5 隐式 + 5 请求模板按 (state, variant) 确定性注入 "
          "（assemble.build_goal，预览为真实逻辑，物化发生在 assemble）。",
          [Action("preview", "预览全部状态的注入结果", "inprocess"),
           Action("assemble", "物化（运行 assemble）", "cli", [])],
          ["train/sft/revact_sft.jsonl"], "real"),
    Stage("candidates", "候选动作生成", "S4",
          "从当前 AXTree 枚举真实 interactive bids，生成并验证每状态 4–6 个候选；"
          "类别仅是提案覆盖，effect/recovery 必须由后续 point probe 测量。",
          [Action("propose", "枚举并物化合法候选", "inprocess", [],
                  "参数 state；写入 raw/candidates/iris_candidates.v3.jsonl")],
          ["raw/state_bank/*reached_states.jsonl",
           "raw/candidates/iris_candidates*.jsonl"],
          "real"),
    Stage("counterfactuals", "反事实动作生成", "S8b",
          "四类 DPO rejected 变体（false_safe/over_block/goal_violation/"
          "wrong_reversibility），assemble 的真实 builder。",
          [Action("preview", "按状态预览反事实", "inprocess"),
           Action("assemble", "物化（运行 assemble）", "cli", [])],
          ["train/dpo/revact_dpo.jsonl"], "real"),
    Stage("probe", "grounded undo 可逆性标注", "S5 ⭐",
          "execute-then-undo 行为测量（多站点：shopping + reddit）。破坏性探针不在"
          "工作台开放（需 CLI 双闸门 + 逐批批准）。",
          [Action("probe_mock", "mock 探针（离线，全站点）", "cli", ["mock"],
                  "MockShopping+MockReddit 上跑全部非破坏探针"),
           Action("probe_live", "live 非破坏探针（shopping）", "cli", ["live:shopping"]),
           Action("probe_live_reddit", "live 非破坏探针（reddit）", "cli",
                  ["live:reddit"], "Postmill vote/subscribe 自恢复探测"),
           Action("probe_named", "live 指定探针（dry-run）", "cli", ["live:shopping"])],
          ["grounded/reversibility.jsonl", "grounded/MANIFEST.jsonl"], "real"),
    Stage("distill", "teacher 条件蒸馏", "S7",
          "teacher 在 pin 死 grounded 结论下只写措辞；QC 矛盾拒绝。",
          [Action("distill", "运行蒸馏", "cli", ["key:teacher"])],
          ["train/sft/revact_sft_distilled.jsonl"], "real"),
    Stage("qc", "样本质量校验", "S6/S8/split",
          "assemble（单步+多轮）+ split + 各训练入口 --dry-run 校验 + 质量统计。",
          [Action("assemble", "assemble（S6-S8 组装）", "cli", []),
           Action("assemble_multiturn", "assemble-multiturn（轨迹→多轮样本）", "cli", [],
                  "P1：完整轨迹为一条 chat 样本，风险步带 <think> 块"),
           Action("split", "split（产品级切分）", "cli", []),
           Action("train_dry", "train --dry-run 校验", "cli", []),
           Action("dpo_dry", "train-dpo --dry-run 校验", "cli", []),
           Action("grpo_dry", "train-grpo --dry-run 校验", "cli", []),
           Action("eval_dry", "eval --dry-run 校验", "cli", []),
           Action("compute", "计算质量统计报告", "inprocess")],
          ["train/splits/sft_train.jsonl", "train/splits/sft_test.jsonl",
           "train/sft/revact_sft_multiturn.jsonl",
           "train/dpo/revact_dpo_multiturn.jsonl"], "real"),
    Stage("export", "数据集导出与可视化", "S8/viz",
          "应用人工标注 overlay 后导出训练集 + dataset card；重建静态报告页。",
          [Action("export", "导出数据集（应用标注）", "inprocess"),
           Action("viz", "重建 dataset_viz.html 报告", "cli", [])],
          ["train/splits/*.jsonl"], "real"),
]

STAGE_BY_ID = {s.id: s for s in STAGES}


# --------------------------------------------------------------------------- #
# Status derivation
# --------------------------------------------------------------------------- #
def _artifact_stats(stage: Stage, root: Path) -> list[dict]:
    out = []
    for pat in stage.artifacts:
        base = root if not pat.startswith("outputs/") else config.PROJECT_ROOT
        matches = sorted(base.glob(pat))
        for m in matches[:50]:
            try:
                n_lines = sum(1 for _ in m.open(encoding="utf-8", errors="ignore")) \
                    if m.suffix in (".jsonl", ".json", ".txt") else None
            except OSError:
                n_lines = None
            out.append({"path": str(m.relative_to(base)), "rows": n_lines,
                        "mtime": m.stat().st_mtime})
        if not matches:
            out.append({"path": pat, "rows": None, "mtime": None, "missing": True})
    return out


def stage_status(stage: Stage, store: DataStore) -> dict:
    last_job = MANAGER.last_for_stage(stage.id)
    arts = _artifact_stats(stage, store.root)
    present = [a for a in arts if not a.get("missing")]
    if last_job and last_job["status"] == "running":
        status = "running"
    elif last_job and last_job["status"] == "failed" and not present:
        status = "failed"
    elif not present:
        status = "not_started"
    elif len(present) < len(stage.artifacts):
        status = "partial"
    elif last_job and last_job["status"] == "failed":
        status = "partial"          # artifacts exist but the last run failed
    else:
        status = "success"
    if stage.id == "probe" and present:
        labels = store.effective_labels()
        n_unknown = sum(1 for v in labels.values() if v in ("UNKNOWN", "NO_EFFECT"))
        if labels and n_unknown:
            status = "partial"
    if stage.implemented == "placeholder":
        n_ann = len(annotations.effective("candidate", store.root))
        status = "partial" if n_ann or present else "not_started"
    return {"status": status, "artifacts": arts,
            "last_job": {k: last_job[k] for k in
                         ("job_id", "action", "status", "started_at", "finished_at")}
            if last_job else None}


def pipeline_overview(store: DataStore | None = None) -> list[dict]:
    store = store or DataStore()
    out = []
    for s in STAGES:
        st = stage_status(s, store)
        out.append({
            "id": s.id, "title": s.title, "s_ref": s.s_ref,
            "description": s.description, "implemented": s.implemented,
            "actions": [{"id": a.id, "label": a.label, "kind": a.kind,
                         "needs": a.needs, "description": a.description}
                        for a in s.actions],
            **st,
        })
    return out


# --------------------------------------------------------------------------- #
# Action dispatch
# --------------------------------------------------------------------------- #
def _check_needs(action: Action, cfg: RuntimeConfig) -> str | None:
    for need in action.needs:
        if need == "live" and not live_ready(cfg):
            return ("需要 live WebArena：先 source scripts/export_webarena_env.sh "
                    "或在全局配置里填 WA_SHOPPING")
        if need.startswith("live:"):
            site = need.split(":", 1)[1]
            if not live_ready(cfg, site):
                spec = config.SITES.get(site)
                env_var = spec.base_env if spec else "WA_*"
                return (f"需要 live {site}：先 source scripts/export_webarena_env.sh "
                        f"或在全局配置里填 {env_var}")
        if need.startswith("key:") and not cfg.has_key(need.split(":", 1)[1]):
            role = need.split(":", 1)[1]
            key_env = ((cfg.settings.get("models") or {}).get(role) or {}) \
                .get("api_key_env", "?")
            return f"缺少 {role} 模型 API key：在配置区填写（env {key_env}），或先 export"
    return None


def _int(params: dict, key: str, default: int, lo: int = 0, hi: int = 100000) -> int:
    try:
        v = int(params.get(key, default))
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


def _build_cmd(stage: Stage, action: Action, params: dict,
               cfg: RuntimeConfig) -> list[str] | None:
    run = cfg.settings.get("run") or {}
    seeds = str(params.get("seeds", run.get("seeds", "0")))
    if not all(p.strip().isdigit() for p in seeds.split(",") if p.strip()):
        seeds = "0"
    if stage.id == "collect":
        if action.id == "collect_mock":
            return python_cli("collect", "--mock", "--seeds", seeds)
        if action.id == "collect_live":
            judge = ((cfg.settings.get("models") or {}).get("judge") or {})
            cmd = python_cli("collect", "--seeds", seeds,
                             "--max-steps", str(_int(params, "max_steps",
                                                     int(run.get("max_steps", 25)), 1, 100)),
                             "--wa-judge", str(params.get(
                                 "wa_judge", judge.get("mode", "off"))))
            task_file = str(params.get("task_file", run.get("task_file", "")))
            if task_file:
                p = (config.PROJECT_ROOT / task_file).resolve()
                if p.is_file() and config.PROJECT_ROOT in p.parents:
                    cmd += ["--task-file", str(p)]
            model = ((cfg.settings.get("models") or {}).get("policy") or {})
            if model.get("model"):
                cmd += ["--model", str(model["model"])]
            if model.get("api_key_env"):
                cmd += ["--api-key-env", str(model["api_key_env"])]
            if params.get("screenshots", run.get("screenshots")):
                cmd += ["--screenshots"]
            if params.get("read_only_live", run.get("read_only_live", True)):
                cmd += ["--read-only-live"]
            if params.get("only_success", run.get("only_success", True)):
                cmd += ["--only-success"]
            return cmd
        if action.id == "inspect":
            return python_cli("inspect")
    if stage.id == "key_states":
        if action.id == "reach":
            return python_cli("reach")
        if action.id == "crawl":
            return python_cli("crawl", "--cap", str(_int(params, "cap", 40, 1, 200)))
        if action.id == "scale":
            return python_cli("scale", "--n-place-order",
                              str(_int(params, "n_place_order", 6, 0, 50)))
    if stage.id in ("constraints", "counterfactuals", "qc") and action.id == "assemble":
        return python_cli("assemble")
    if stage.id == "probe":
        if action.id == "probe_mock":
            return python_cli("probe", "--mock", *MOCK_PROBES)
        if action.id == "probe_live":
            cmd = python_cli("probe", "--all-nondestructive", "--site", "shopping")
            if params.get("screenshots", run.get("screenshots")):
                cmd += ["--screenshots"]
            return cmd
        if action.id == "probe_live_reddit":
            cmd = python_cli("probe", "--all-nondestructive", "--site", "reddit")
            if params.get("screenshots", run.get("screenshots")):
                cmd += ["--screenshots"]
            return cmd
        if action.id == "probe_named":
            names = [n for n in (params.get("names") or [])
                     if isinstance(n, str) and n.replace(".", "").replace("_", "").isalnum()]
            if not names:
                return None
            return python_cli("probe", *names)   # no --commit path from the UI
    if stage.id == "distill" and action.id == "distill":
        return python_cli("distill", "--limit",
                          str(_int(params, "limit",
                                   int(run.get("sample_limit", 10)), 1, 100000)))
    if stage.id == "qc":
        if action.id == "assemble_multiturn":
            return python_cli("assemble-multiturn")
        if action.id == "split":
            return python_cli("split")
        if action.id == "train_dry":
            return python_cli("train", "--dry-run")
        if action.id == "dpo_dry":
            return python_cli("train-dpo", "--dry-run")
        if action.id == "grpo_dry":
            return python_cli("train-grpo", "--dry-run")
        if action.id == "eval_dry":
            return python_cli("eval", "--dry-run")
    if stage.id == "export" and action.id == "viz":
        return python_cli("viz")
    return None


def run_action(stage_id: str, action_id: str, params: dict | None = None,
               cfg: RuntimeConfig | None = None) -> dict:
    """Returns {ok, job?|result?|error}. Placeholder actions return the
    extension point instead of pretending to run."""
    cfg = cfg or RUNTIME
    params = params or {}
    stage = STAGE_BY_ID.get(stage_id)
    if stage is None:
        return {"ok": False, "error": f"unknown stage {stage_id!r}"}
    action = next((a for a in stage.actions if a.id == action_id), None)
    if action is None:
        return {"ok": False, "error": f"unknown action {action_id!r} for {stage_id}"}

    if action.kind == "placeholder":
        return {"ok": False, "placeholder": True,
                "error": "该能力尚无后端实现",
                "extension_point": action.description or stage.description}

    blocked = _check_needs(action, cfg)
    if blocked:
        return {"ok": False, "blocked": True, "error": blocked}

    if action.kind == "inprocess":
        result = _run_inprocess(stage, action, params, cfg)
        MANAGER.record_instant(stage.id, action.id, result.get("ok", False),
                               result.get("note", action.label))
        return result

    cmd = _build_cmd(stage, action, params, cfg)
    if cmd is None:
        return {"ok": False, "error": "无法构造命令（参数缺失或非法）"}
    roles = [n.split(":", 1)[1] for n in action.needs if n.startswith("key:")]
    if stage.id == "collect" and action.id == "collect_live":
        roles = ["policy", "judge"]
    job = MANAGER.start(cmd, stage.id, action.id,
                        env_extra=cfg.merged_env(*roles))
    return {"ok": True, "job": job}


def _run_inprocess(stage: Stage, action: Action, params: dict,
                   cfg: RuntimeConfig) -> dict:
    store = DataStore()
    if stage.id == "env_init":
        checks = {
            "WA_SHOPPING": bool(live_ready(cfg, "shopping")),
            "WA_SHOPPING_ADMIN": bool((cfg.settings.get("env") or {})
                                      .get("WA_SHOPPING_ADMIN")
                                      or config.WA_SHOPPING_ADMIN),
            "WA_REDDIT": bool(live_ready(cfg, "reddit")),
            "pilot_task_ids": (store.root / "raw" / "pilot_task_ids.json").exists(),
            "product_urls": (store.root / "raw" / "product_urls.json").exists(),
            "policy_key": cfg.has_key("policy"),
            "teacher_key": cfg.has_key("teacher"),
            "opinion_key": cfg.has_key("opinion"),
        }
        ok = checks["pilot_task_ids"]
        return {"ok": ok, "result": checks,
                "note": "env check: " + ", ".join(f"{k}={'✓' if v else '✗'}"
                                                  for k, v in checks.items())}
    if stage.id == "constraints" and action.id == "preview":
        previews = [p for s in store.reached_states()
                    if (p := store.constraint_preview(s["name"]))]
        return {"ok": True, "result": previews,
                "note": f"constraint preview × {len(previews)} states"}
    if stage.id == "candidates" and action.id == "propose":
        name = str(params.get("state") or "")
        if not name:
            return {"ok": False, "error": "缺少 state 参数"}
        try:
            result = store.materialize_candidates(name)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "result": result,
                "note": f"S4 candidates × {result['n']} -> {result['artifact']}"}
    if stage.id == "counterfactuals" and action.id == "preview":
        name = params.get("state", "")
        res = store.candidates_for(name) if name else None
        if res is None:
            return {"ok": False, "error": f"state {name!r} 不存在或未绑定 grounded 标签"}
        return {"ok": True, "result": res, "note": f"counterfactual preview: {name}"}
    if stage.id == "qc" and action.id == "compute":
        from .quality import compute_quality
        rep = compute_quality(store)
        out = config.OUTPUTS_DIR / "workbench" / "quality.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        import json as _json
        out.write_text(_json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        return {"ok": True, "result": rep, "note": f"quality report -> {out}"}
    if stage.id == "export" and action.id == "export":
        from .export import export_dataset
        rep = export_dataset(store, params)
        return {"ok": rep.get("ok", False), "result": rep,
                "note": rep.get("note", "export")}
    return {"ok": False, "error": "unhandled inprocess action"}
