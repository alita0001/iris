"""LoRA DPO on RevAct/IRIS preference pairs (TRL DPOTrainer).

Rows ({revact_dpo,revact_dpo_multiturn,splits/dpo_train}.jsonl):
    {"pair_id", "prompt": [messages... ending on a user turn],
     "chosen": str, "rejected": str, "meta": {...}}

Single-step and multi-turn pairs load identically: the prompt is a full
conversation, chosen/rejected are the two candidate final assistant turns
(four counterfactual types, see data/assemble._dpo_pairs_for).

`run(dry_run=True)` validates data + config WITHOUT importing torch; the real
path needs the `qwen-vllm` conda env (torch/trl/peft) and a GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

from .. import config

_ROLES_BODY = ("user", "assistant")


def validate_rows(rows: list[dict]) -> dict:
    problems = []
    pair_types: dict[str, int] = {}
    for i, r in enumerate(rows):
        rid = r.get("pair_id", f"row{i}")
        msgs = r.get("prompt", [])
        roles = [m.get("role") for m in msgs]
        if (len(roles) < 2 or roles[0] != "system" or roles[-1] != "user"
                or any(x != _ROLES_BODY[j % 2]
                       for j, x in enumerate(roles[1:]))):
            problems.append(f"{rid}: bad prompt role sequence {roles}")
            continue
        if not r.get("chosen") or not r.get("rejected"):
            problems.append(f"{rid}: empty chosen/rejected")
            continue
        if r["chosen"] == r["rejected"]:
            problems.append(f"{rid}: chosen == rejected")
        pt = (r.get("meta") or {}).get("pair_type", "?")
        pair_types[pt] = pair_types.get(pt, 0) + 1
    return {"n_rows": len(rows), "n_problems": len(problems),
            "problems": problems[:20], "pair_types": pair_types}


def run(train_path: Path | None = None, dry_run: bool = False,
        output_dir: str | None = None, beta: float = 0.1,
        epochs: float | None = None, max_steps: int = -1,
        adapter: str = "") -> int:
    cfg = config.TRAIN
    train_path = train_path or (config.SPLITS_DIR / "dpo_train.jsonl")
    if not train_path.exists():
        print(f"ERROR: missing {train_path} (run `revact assemble`+`split` first)")
        return 1
    rows = [json.loads(ln) for ln in train_path.open()]
    report = validate_rows(rows)
    print(f"[data] {report['n_rows']} pairs | problems={report['n_problems']} "
          f"| pair_types={report['pair_types']}")
    for p in report["problems"]:
        print(f"  [invalid] {p}")
    if report["n_problems"]:
        print("ERROR: fix data problems before training.")
        return 1
    print(f"[cfg] model={cfg.model_path} lora_r={cfg.lora_r} beta={beta} "
          f"max_len={cfg.max_len} max_steps={max_steps} "
          f"adapter={adapter or '(none: cold start)'}")
    if adapter and not Path(adapter).exists():
        print(f"ERROR: adapter path {adapter} not found")
        return 1
    if dry_run:
        print("[dry-run] data + config valid; no training started.")
        return 0
    return _train(rows, cfg, output_dir or str(config.OUTPUTS_DIR / "dpo_lora"),
                  beta=beta, epochs=epochs if epochs is not None else 1.0,
                  max_steps=max_steps, adapter=adapter)


def _train(rows: list[dict], cfg, output_dir: str, beta: float,
           epochs: float, max_steps: int, adapter: str = "") -> int:
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    from .grpo import _load_policy_model

    ds = Dataset.from_list([{
        "prompt": r["prompt"],
        "chosen": [{"role": "assistant", "content": r["chosen"]}],
        "rejected": [{"role": "assistant", "content": r["rejected"]}],
    } for r in rows])

    tok = AutoTokenizer.from_pretrained(cfg.model_path)
    # warm start: SFT LoRA merged in -> the DPO ref model IS the SFT policy
    model = _load_policy_model(cfg, adapter)
    args = DPOConfig(
        output_dir=output_dir, beta=beta,
        per_device_train_batch_size=max(1, cfg.batch_size // 2),
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=epochs, max_steps=max_steps,
        learning_rate=cfg.lr / 2, lr_scheduler_type="cosine", warmup_ratio=0.05,
        max_length=cfg.max_len,
        logging_steps=5, save_strategy="no", bf16=True, report_to=[],
    )
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=ds, processing_class=tok,
        peft_config=LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    print(f"[done] DPO LoRA adapter -> {output_dir}")
    return 0
