"""Offline GRPO / GSPO RLVR ablation (environment-free).

The reward is fully offline-verifiable — no LLM judge, no live environment in
the training loop — because the dataset itself carries the ground truth:

  * format         all think/answer tags present, in order            (+1)
  * decision       <decision> matches the grounded ORACLE decision    (+2)
  * reversibility  <reversibility> matches the behaviorally MEASURED
                   label (execute-then-undo probe, never an opinion)  (+2)
  * consistency    declared decision agrees with parsed <answer> action
  * behavior       structured risky action attempt/commit penalties

Prompts come straight from the SFT files (single-step and/or multi-turn):
prompt = messages[:-1]; the gold fields come from meta. Same prompts as
SFT/DPO -> the three stages are directly comparable.

GSPO (sequence-level importance sampling, Zheng et al. 2025) is TRL's
``importance_sampling_level="sequence"``; enable with --gspo. Mechanism
citations: format/verifiable rewards follow VAGEN's design; this project's
contribution is the SUPERVISION SOURCE (behavioral grounding), not the RL
algorithm.

This is explicitly *not* environment-in-the-loop trajectory RL and cannot by
itself establish a reduction in multi-step rollout distribution shift.

`run(dry_run=True)` validates data + config WITHOUT importing torch.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .. import config
from .validators import (SAFE_DECISIONS, actions_match, answer_text,
                         decision_answer_consistent,
                         formal_point_join_problems, iris_tag_errors,
                         parse_action, parse_decision, percentile)

_TAG_ORDER = ("<think>", "<observation>", "<reasoning>", "<prediction>",
              "<rev_check>", "<reversibility>", "<undo>", "<decision>",
              "</think>", "<answer>")
_DECISIONS = ("EXECUTE", "VERIFY", "CONFIRM", "AVOID")
_LABELS = ("REVERSIBLE", "PARTIALLY_RECOVERABLE", "IRREVERSIBLE",
           "NO_EFFECT", "UNKNOWN", "RECOVERED", "PARTIALLY_RECOVERED",
           "NOT_RECOVERED_WITHIN_BUDGET")
_RECOVERY_MAP = {
    "REVERSIBLE": "RECOVERED",
    "PARTIALLY_RECOVERABLE": "PARTIALLY_RECOVERED",
    "IRREVERSIBLE": "NOT_RECOVERED_WITHIN_BUDGET",
    "NO_EFFECT": "UNKNOWN",
}
MAX_COMPLETION_LENGTH = 320


@dataclass(frozen=True)
class RewardWeights:
    """Auditable reward scale; every coefficient is exposed for ablation."""

    format: float = 1.0
    decision: float = 2.0
    recovery: float = 2.0
    consistency_correct: float = 1.0
    consistency_wrong: float = -2.0
    constraint_attempt: float = -4.0
    safe_decision_attempt: float = -8.0
    safe_decision_commit: float = -12.0


DEFAULT_REWARD_WEIGHTS = RewardWeights()


def reward_weight_ablations() -> dict[str, RewardWeights]:
    """Predeclared scales to report rather than hiding one hand-picked mix."""
    base = DEFAULT_REWARD_WEIGHTS
    return {
        "full": base,
        "no_format": replace(base, format=0.0),
        "no_consistency": replace(base, consistency_correct=0.0,
                                  consistency_wrong=0.0),
        "uniform_safety_penalty": replace(
            base, constraint_attempt=-4.0, safe_decision_attempt=-4.0,
            safe_decision_commit=-4.0),
    }


def _text(completion) -> str:
    """TRL passes conversational completions as [{'role','content'}, ...]."""
    if isinstance(completion, str):
        return completion
    return " ".join(m.get("content", "") for m in completion)


def _field(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>\s*([A-Z_]+)", text)
    return m.group(1) if m else ""


def _answer(text: str) -> str:
    return answer_text(text)


def _normalise_recovery(value: str) -> str:
    return _RECOVERY_MAP.get(value, value)


def _values(column, n: int, default=None) -> list:
    if column is None:
        return [default] * n
    return list(column)


# --------------------------------------------------------------------------- #
# Reward functions (TRL signature: (completions, **columns) -> list[float])
# --------------------------------------------------------------------------- #
def reward_format(completions, weights: RewardWeights = DEFAULT_REWARD_WEIGHTS,
                  **kw):
    # Exact cardinality matters: a completion that appends a second <answer>
    # or <decision> must not earn format reward merely because each first
    # occurrence happens to be ordered.
    return [weights.format if not iris_tag_errors(_text(c)) else 0.0
            for c in completions]


def reward_decision(completions, gold_decision=None,
                    weights: RewardWeights = DEFAULT_REWARD_WEIGHTS, **kw):
    return [weights.decision if _field(_text(c), "decision") == g else 0.0
            for c, g in zip(completions, gold_decision)]


def reward_reversibility(completions, gold_reversibility=None,
                         gold_recovery_status=None,
                         weights: RewardWeights = DEFAULT_REWARD_WEIGHTS, **kw):
    gold = gold_recovery_status if gold_recovery_status is not None else gold_reversibility
    return [weights.recovery
            if _normalise_recovery(_field(_text(c), "reversibility")) ==
            _normalise_recovery(g) else 0.0
            for c, g in zip(completions, gold)]


def reward_decision_answer_consistency(
        completions, risky_action=None, risky_raw_action=None,
        weights: RewardWeights = DEFAULT_REWARD_WEIGHTS, **kw):
    """Reward a decision only when its emitted action obeys that declaration."""
    n = len(completions)
    structured = _values(risky_action, n)
    legacy = _values(risky_raw_action, n, "")
    out = []
    for completion, target, raw in zip(completions, structured, legacy):
        text = _text(completion)
        target = target or raw
        ok = decision_answer_consistent(
            parse_decision(text), answer_text(text), target)
        out.append(weights.consistency_correct if ok
                   else weights.consistency_wrong)
    return out


def reward_safety(completions, variant=None, risky_raw_action=None,
                  risky_action=None, action_legal=None, backend_commit=None,
                  normative_risk=None, violates_constraint=None,
                  weights: RewardWeights = DEFAULT_REWARD_WEIGHTS, **kw):
    """Penalise parsed risky behavior, with commit after a safe decision worst.

    ``risky_action`` is the formal structured identity.  ``risky_raw_action``
    remains a clearly scoped legacy fallback.  Neither think prose nor a raw
    substring match can create an attempt.
    """
    n = len(completions)
    variants = _values(variant, n, "")
    structured = _values(risky_action, n)
    legacy = _values(risky_raw_action, n, "")
    legality = _values(action_legal, n)
    commits = _values(backend_commit, n, False)
    risks = _values(normative_risk, n)
    violations = _values(violates_constraint, n)
    out = []
    for completion, case, target, raw, legal, committed, risk, violation in zip(
            completions, variants, structured, legacy, legality, commits,
            risks, violations):
        text = _text(completion)
        target = target or raw
        actual = parse_action(answer_text(text))
        attempted = actions_match(actual, target) and legal is not False
        decision = parse_decision(text)
        if not attempted:
            out.append(0.0)
        elif decision in SAFE_DECISIONS and bool(committed):
            out.append(weights.safe_decision_commit)
        elif decision in SAFE_DECISIONS:
            out.append(weights.safe_decision_attempt)
        # Legacy artifacts had only the request/constraint variant.  Formal
        # rows carry independent normative and policy-constraint truth.
        elif ((violation if isinstance(violation, bool)
               else case == "constraint") or
              (risk if isinstance(risk, bool) else False)):
            out.append(weights.constraint_attempt)
        else:
            out.append(0.0)
    return out


REWARD_FUNCS = [reward_format, reward_decision, reward_reversibility,
                reward_decision_answer_consistency, reward_safety]


def make_reward_funcs(weights: RewardWeights) -> list:
    """Bind one weight set while retaining TRL's keyword-column signature."""
    def bind(fn):
        def wrapped(completions, **columns):
            return fn(completions, weights=weights, **columns)
        wrapped.__name__ = fn.__name__
        return wrapped
    return [bind(fn) for fn in REWARD_FUNCS]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_prompts(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        for ln in path.open(encoding="utf-8"):
            r = json.loads(ln)
            meta = r.get("meta", {})
            msgs = r.get("messages", [])
            if len(msgs) < 3 or msgs[-1].get("role") != "assistant":
                continue
            risky_raw = meta.get("risky_raw_action", "")
            risky_structured = meta.get("risky_action")
            if not risky_structured and risky_raw:
                parsed = parse_action(risky_raw)
                risky_structured = parsed.to_dict() if parsed else None
            rows.append({
                "sample_id": r.get("sample_id", ""),
                "prompt": msgs[:-1],
                "gold_decision": meta.get("decision", ""),
                "gold_reversibility": meta.get("reversibility", ""),
                "gold_recovery_status": meta.get(
                    "recovery_status", meta.get("reversibility", "")),
                "variant": meta.get("variant", ""),
                "risky_raw_action": risky_raw,
                "risky_action": risky_structured,
                "action_legal": meta.get(
                    "legal_at_snapshot", meta.get("action_legal")),
                "backend_commit": meta.get("backend_commit"),
                "normative_risk": meta.get("normative_risk"),
                "violates_constraint": meta.get("violates_constraint"),
                "formal_dataset": bool(meta.get("formal_dataset")),
                "probe_point_id": meta.get("probe_point_id", ""),
                "state_id": meta.get("state_id", ""),
                "action_instance_id": meta.get("action_instance_id", ""),
                "effect_status": meta.get("effect_status", ""),
                "prediction_source": meta.get("prediction_source", ""),
                "undo_source": meta.get("undo_source", ""),
                "history_source": meta.get("history_source", ""),
                "is_mock": meta.get("is_mock"),
                "collector_success": meta.get("collector_success"),
            })
    return rows


def validate_rows(rows: list[dict]) -> dict:
    problems = []
    n_formal = 0
    for i, r in enumerate(rows):
        if r["gold_decision"] not in _DECISIONS:
            problems.append(f"row{i}: bad gold_decision {r['gold_decision']!r}")
        recovery = r.get("gold_recovery_status", r.get("gold_reversibility", ""))
        if recovery not in _LABELS:
            problems.append(f"row{i}: bad gold_reversibility "
                            f"{recovery!r}")
        if r.get("variant") == "constraint" and not r.get("risky_raw_action"):
            if not r.get("risky_action"):
                problems.append(f"row{i}: constraint variant without risky action")
        if r.get("backend_commit") and r.get("action_legal") is False:
            problems.append(f"row{i}: backend_commit=true but action_legal=false")
        if r.get("formal_dataset"):
            n_formal += 1
            for key in ("probe_point_id", "state_id", "action_instance_id"):
                if not r.get(key):
                    problems.append(f"row{i}: formal offline RLVR row lacks {key}")
            if r.get("is_mock") is not False or r.get("collector_success") is not True:
                problems.append(
                    f"row{i}: formal row requires is_mock=false and collector_success=true")
            if r.get("effect_status") not in {"CHANGED", "NO_EFFECT", "UNKNOWN"}:
                problems.append(f"row{i}: formal row lacks canonical effect_status")
            if r.get("prediction_source") != "probe_transition":
                problems.append(f"row{i}: formal prediction_source is not probe_transition")
            if r.get("undo_source") != "probe_point_id":
                problems.append(f"row{i}: formal undo_source is not probe_point_id")
            if r.get("history_source") != "trajectory":
                problems.append(f"row{i}: formal history_source is not trajectory")
            if recovery not in {"RECOVERED", "PARTIALLY_RECOVERED",
                                "NOT_RECOVERED_WITHIN_BUDGET", "UNKNOWN"}:
                problems.append(
                    f"row{i}: formal row uses legacy recovery label {recovery!r}")
            if not isinstance(r.get("risky_action"), dict):
                problems.append(f"row{i}: formal row lacks structured risky_action")
            if not isinstance(r.get("action_legal"), bool):
                problems.append(f"row{i}: formal row lacks boolean action_legal")
            if not isinstance(r.get("backend_commit"), bool):
                problems.append(f"row{i}: formal row lacks boolean backend_commit truth")
            if not isinstance(r.get("normative_risk"), bool):
                problems.append(f"row{i}: formal row lacks boolean normative_risk")
            if not isinstance(r.get("violates_constraint"), bool):
                problems.append(
                    f"row{i}: formal row lacks boolean violates_constraint")
    if 0 < n_formal < len(rows):
        problems.append(
            f"dataset mixes {n_formal} formal and {len(rows) - n_formal} legacy rows")
    return {"n_rows": len(rows), "n_problems": len(problems),
            "problems": problems[:20], "n_formal": n_formal}


def prompt_encoding_report(rows: list[dict], tokenizer,
                           max_prompt_len: int) -> dict:
    """Detect prompt truncation before TRL generates any completions."""
    from .sft import _token_ids

    lengths: list[int] = []
    dropped: list[str] = []
    for i, row in enumerate(rows):
        length = len(_token_ids(
            tokenizer, row["prompt"], generation_prompt=True))
        lengths.append(length)
        if length > max_prompt_len:
            dropped.append(str(row.get("sample_id", f"row{i}")))
    return {
        "n_prompts": len(rows), "token_p50": percentile(lengths, .50),
        "token_p95": percentile(lengths, .95),
        "token_max": max(lengths, default=0),
        "max_prompt_len": max_prompt_len, "n_dropped": len(dropped),
        "dropped_prompt_ids": dropped,
    }


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #
def run(train_paths: list[Path] | None = None, dry_run: bool = False,
        output_dir: str | None = None, gspo: bool = False,
        num_generations: int = 4, max_steps: int = -1,
        epochs: float = 1.0, adapter: str = "",
        reward_weights: RewardWeights | None = None,
        *, allow_legacy: bool = False) -> int:
    cfg = config.TRAIN
    paths = train_paths or [config.FORMAL_SFT_TRAIN_PATH]
    source_rows = [json.loads(line) for path in paths if path.exists()
                   for line in path.open(encoding="utf-8") if line.strip()]
    rows = load_prompts(paths)
    if not rows:
        print(f"ERROR: no usable rows in {[str(p) for p in paths]}")
        return 1
    report = validate_rows(rows)
    point_problems = formal_point_join_problems(source_rows, config.DATA_ROOT)
    if point_problems:
        report["problems"] = (report["problems"] + point_problems)[:20]
        report["n_problems"] += len(point_problems)
    print(f"[data] {report['n_rows']} prompts from {len(paths)} file(s) "
          f"| problems={report['n_problems']}")
    if report["n_formal"] != len(rows):
        print("[mode] LEGACY/DEVELOPMENT: this dry-run is not point-level "
              "formal RLVR supervision.")
    for p in report["problems"]:
        print(f"  [invalid] {p}")
    if report["n_problems"]:
        print("ERROR: fix data problems before training.")
        return 1
    if report["n_formal"] != len(rows):
        if not (dry_run and allow_legacy):
            print("ERROR: formal offline RLVR is fail-closed. Legacy prompts "
                  "may only be audited with both --dry-run and --allow-legacy; "
                  "they can never enter the real trainer.")
            return 1
        print("[legacy-audit] explicitly authorised validation only; the real "
              "trainer remains disabled for these prompts.")
    algo = "GSPO(sequence-level IS)" if gspo else "GRPO(token-level IS)"
    weights = reward_weights or DEFAULT_REWARD_WEIGHTS
    print(f"[cfg] OFFLINE RLVR ABLATION: {algo}; environment_in_loop=false "
          f"model={cfg.model_path} G={num_generations} "
          f"max_steps={max_steps} adapter={adapter or '(none: cold start)'} "
          f"reward_weights={asdict(weights)}")
    print("[reward-ablations] " + json.dumps(
        {name: asdict(value) for name, value in reward_weight_ablations().items()},
        sort_keys=True))
    max_prompt_len = cfg.max_len - MAX_COMPLETION_LENGTH
    if max_prompt_len <= 0:
        print("ERROR: train.max_len must exceed max_completion_length="
              f"{MAX_COMPLETION_LENGTH}")
        return 1
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_path, local_files_only=True)
        token_report = prompt_encoding_report(rows, tokenizer, max_prompt_len)
    except Exception as exc:
        print(f"ERROR: offline tokenizer audit failed: {type(exc).__name__}: {exc}")
        return 1
    print("[tokens] prompts="
          f"{token_report['n_prompts']} p50={token_report['token_p50']:.1f} "
          f"p95={token_report['token_p95']:.1f} "
          f"max={token_report['token_max']} "
          f"max_prompt_len={max_prompt_len} "
          f"drop={token_report['n_dropped']}")
    if token_report["n_dropped"]:
        print(f"  [dropped_prompt_ids] {token_report['dropped_prompt_ids'][:20]}")
        print("ERROR: offline RLVR would truncate a prompt; refusing.")
        return 1
    if adapter and not Path(adapter).exists():
        print(f"ERROR: adapter path {adapter} not found")
        return 1
    if dry_run:
        scope = "formal" if report["n_formal"] == len(rows) else "legacy-audit"
        print(f"[dry-run] {scope} data + config valid; no training started.")
        return 0
    return _train(rows, cfg,
                  output_dir or str(config.OUTPUTS_DIR /
                                    ("gspo_lora" if gspo else "grpo_lora")),
                  gspo=gspo, num_generations=num_generations,
                  max_steps=max_steps, epochs=epochs, adapter=adapter,
                  reward_weights=weights)


def _load_policy_model(cfg, adapter: str):
    """Base model, with the SFT LoRA merged in when warm-starting: the new
    RL/DPO LoRA then trains on top of the SFT policy's weights."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(cfg.model_path,
                                                 dtype=torch.bfloat16,
                                                 device_map="cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        print(f"[init] merged SFT adapter {adapter} into the policy")
    return model


def _train(rows: list[dict], cfg, output_dir: str, gspo: bool,
           num_generations: int, max_steps: int, epochs: float,
           adapter: str = "",
           reward_weights: RewardWeights = DEFAULT_REWARD_WEIGHTS) -> int:
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    ds = Dataset.from_list(rows)
    tok = AutoTokenizer.from_pretrained(cfg.model_path)
    model = _load_policy_model(cfg, adapter)
    args = GRPOConfig(
        output_dir=output_dir,
        # GSPO (Zheng et al. 2025) = sequence-level importance ratios + the
        # per-sequence 'grpo' loss; TRL warns that the default 'dapo' loss
        # would length-weight sequences instead.
        importance_sampling_level="sequence" if gspo else "token",
        loss_type="grpo" if gspo else "dapo",
        num_generations=num_generations,
        per_device_train_batch_size=num_generations,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=epochs, max_steps=max_steps,
        learning_rate=cfg.lr / 4, lr_scheduler_type="cosine", warmup_ratio=0.05,
        max_prompt_length=cfg.max_len - MAX_COMPLETION_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        temperature=0.9, beta=0.0,
        logging_steps=1, save_strategy="no", bf16=True, report_to=[],
    )
    trainer = GRPOTrainer(
        model=model, args=args, train_dataset=ds,
        reward_funcs=make_reward_funcs(reward_weights), processing_class=tok,
        peft_config=LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    print(f"[done] {'GSPO' if gspo else 'GRPO'} LoRA adapter -> {output_dir}")
    return 0
