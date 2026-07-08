"""GRPO / GSPO with grounded verifiable rewards (RLVR-style, env-free).

The reward is fully offline-verifiable — no LLM judge, no live environment in
the training loop — because the dataset itself carries the ground truth:

  * format         all think/answer tags present, in order            (+1)
  * decision       <decision> matches the grounded ORACLE decision    (+2)
  * reversibility  <reversibility> matches the behaviorally MEASURED
                   label (execute-then-undo probe, never an opinion)  (+2)
  * safety         parsed <answer> executes meta.risky_raw_action even
                   though the goal forbids it (variant=constraint)    (-4)

Prompts come straight from the SFT files (single-step and/or multi-turn):
prompt = messages[:-1]; the gold fields come from meta. Same prompts as
SFT/DPO -> the three stages are directly comparable.

GSPO (sequence-level importance sampling, Zheng et al. 2025) is TRL's
``importance_sampling_level="sequence"``; enable with --gspo. Mechanism
citations: format/verifiable rewards follow VAGEN's design; this project's
contribution is the SUPERVISION SOURCE (behavioral grounding), not the RL
algorithm.

`run(dry_run=True)` validates data + config WITHOUT importing torch.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .. import config

_TAG_ORDER = ("<think>", "<observation>", "<reasoning>", "<prediction>",
              "<rev_check>", "<reversibility>", "<undo>", "<decision>",
              "</think>", "<answer>")
_DECISIONS = ("EXECUTE", "VERIFY", "CONFIRM", "AVOID")
_LABELS = ("REVERSIBLE", "PARTIALLY_RECOVERABLE", "IRREVERSIBLE",
           "NO_EFFECT", "UNKNOWN")


def _text(completion) -> str:
    """TRL passes conversational completions as [{'role','content'}, ...]."""
    if isinstance(completion, str):
        return completion
    return " ".join(m.get("content", "") for m in completion)


def _field(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>\s*([A-Z_]+)", text)
    return m.group(1) if m else ""


def _answer(text: str) -> str:
    m = re.search(r"<answer>\s*(.+)", text, re.DOTALL)
    return m.group(1).strip().splitlines()[0].strip() if m else ""


def _norm_action(a: str) -> str:
    return re.sub(r"\s+", "", (a or "").replace('"', "'"))


# --------------------------------------------------------------------------- #
# Reward functions (TRL signature: (completions, **columns) -> list[float])
# --------------------------------------------------------------------------- #
def reward_format(completions, **kw):
    out = []
    for c in completions:
        t = _text(c)
        pos = [t.find(tag) for tag in _TAG_ORDER]
        out.append(1.0 if all(p >= 0 for p in pos) and pos == sorted(pos) else 0.0)
    return out


def reward_decision(completions, gold_decision=None, **kw):
    return [2.0 if _field(_text(c), "decision") == g else 0.0
            for c, g in zip(completions, gold_decision)]


def reward_reversibility(completions, gold_reversibility=None, **kw):
    return [2.0 if _field(_text(c), "reversibility") == g else 0.0
            for c, g in zip(completions, gold_reversibility)]


def reward_safety(completions, variant=None, risky_raw_action=None, **kw):
    """Constraint violation = the one behavior IRIS exists to prevent."""
    out = []
    for c, v, risky in zip(completions, variant, risky_raw_action):
        ans = _norm_action(_answer(_text(c)))
        violated = (v == "constraint" and risky
                    and _norm_action(risky) in ans)
        out.append(-4.0 if violated else 0.0)
    return out


REWARD_FUNCS = [reward_format, reward_decision, reward_reversibility,
                reward_safety]


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
            rows.append({
                "prompt": msgs[:-1],
                "gold_decision": meta.get("decision", ""),
                "gold_reversibility": meta.get("reversibility", ""),
                "variant": meta.get("variant", ""),
                "risky_raw_action": meta.get("risky_raw_action", ""),
            })
    return rows


def validate_rows(rows: list[dict]) -> dict:
    problems = []
    for i, r in enumerate(rows):
        if r["gold_decision"] not in _DECISIONS:
            problems.append(f"row{i}: bad gold_decision {r['gold_decision']!r}")
        if r["gold_reversibility"] not in _LABELS:
            problems.append(f"row{i}: bad gold_reversibility "
                            f"{r['gold_reversibility']!r}")
        if r["variant"] == "constraint" and not r["risky_raw_action"]:
            problems.append(f"row{i}: constraint variant without risky_raw_action")
    return {"n_rows": len(rows), "n_problems": len(problems),
            "problems": problems[:20]}


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #
def run(train_paths: list[Path] | None = None, dry_run: bool = False,
        output_dir: str | None = None, gspo: bool = False,
        num_generations: int = 4, max_steps: int = -1,
        epochs: float = 1.0, adapter: str = "") -> int:
    cfg = config.TRAIN
    paths = train_paths or [config.SPLITS_DIR / "sft_train.jsonl",
                            config.TRAIN_DIR / "sft" / "revact_sft_multiturn.jsonl"]
    rows = load_prompts(paths)
    if not rows:
        print(f"ERROR: no usable rows in {[str(p) for p in paths]}")
        return 1
    report = validate_rows(rows)
    print(f"[data] {report['n_rows']} prompts from {len(paths)} file(s) "
          f"| problems={report['n_problems']}")
    for p in report["problems"]:
        print(f"  [invalid] {p}")
    if report["n_problems"]:
        print("ERROR: fix data problems before training.")
        return 1
    algo = "GSPO(sequence-level IS)" if gspo else "GRPO(token-level IS)"
    print(f"[cfg] {algo} model={cfg.model_path} G={num_generations} "
          f"max_steps={max_steps} adapter={adapter or '(none: cold start)'} "
          f"rewards=[format+1, decision+2, reversibility+2, safety-4]")
    if adapter and not Path(adapter).exists():
        print(f"ERROR: adapter path {adapter} not found")
        return 1
    if dry_run:
        print("[dry-run] data + config valid; no training started.")
        return 0
    return _train(rows, cfg,
                  output_dir or str(config.OUTPUTS_DIR /
                                    ("gspo_lora" if gspo else "grpo_lora")),
                  gspo=gspo, num_generations=num_generations,
                  max_steps=max_steps, epochs=epochs, adapter=adapter)


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
           adapter: str = "") -> int:
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
        max_completion_length=320,
        temperature=0.9, beta=0.0,
        logging_steps=1, save_strategy="no", bf16=True, report_to=[],
    )
    trainer = GRPOTrainer(
        model=model, args=args, train_dataset=ds,
        reward_funcs=REWARD_FUNCS, processing_class=tok,
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
