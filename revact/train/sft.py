"""LoRA SFT of Qwen2.5 on RevAct/IRIS sequences (completion-only loss).

`run(dry_run=True)` validates data + config WITHOUT importing torch, so the
CLI works in any environment; the real training path lazily imports the heavy
deps (run inside the `qwen-vllm` conda env with a GPU).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .. import config

_TAGS = ("<observation>", "<reasoning>", "<prediction>", "<rev_check>",
         "<reversibility>", "<undo>", "<decision>", "<answer>")
_DECISIONS = ("EXECUTE", "VERIFY", "CONFIRM", "AVOID")


def _roles_ok(roles: list) -> bool:
    """system, then strictly alternating user/assistant, ending on assistant.
    Covers single-step (3 messages) and multi-turn trajectories alike."""
    if len(roles) < 3 or roles[0] != "system" or roles[-1] != "assistant":
        return False
    body = roles[1:]
    return all(r == ("user" if i % 2 == 0 else "assistant")
               for i, r in enumerate(body))


def validate_rows(rows: list[dict]) -> dict:
    """Schema/format validation shared by --dry-run and the real path.

    The DECISION turn (final assistant message) must carry the full think
    block; earlier assistant turns are routine `<answer> action` steps."""
    problems = []
    decisions, actions = {}, {}
    for i, r in enumerate(rows):
        rid = r.get("sample_id", f"row{i}")
        msgs = r.get("messages", [])
        roles = [m.get("role") for m in msgs]
        if not _roles_ok(roles):
            problems.append(f"{rid}: bad role sequence {roles}")
            continue
        asst = msgs[-1].get("content", "")
        missing = [t for t in _TAGS if t not in asst]
        if missing:
            problems.append(f"{rid}: final assistant missing tags {missing}")
        m = re.search(r"<decision>\s*([A-Z_]+)", asst)
        if not m or m.group(1) not in _DECISIONS:
            problems.append(f"{rid}: unparsable decision")
        for j, msg in enumerate(msgs[2:-1:2], start=1):
            if "<answer>" not in msg.get("content", ""):
                problems.append(f"{rid}: routine assistant turn {j} lacks <answer>")
        meta = r.get("meta", {})
        decisions[meta.get("decision", "?")] = decisions.get(meta.get("decision", "?"), 0) + 1
        actions[meta.get("action_type", "?")] = actions.get(meta.get("action_type", "?"), 0) + 1
    return {"n_rows": len(rows), "n_problems": len(problems),
            "problems": problems[:20], "decision_dist": decisions,
            "action_dist": actions}


def run(train_path: Path | None = None, dry_run: bool = False,
        output_dir: str | None = None, max_len: int | None = None) -> int:
    import dataclasses

    cfg = config.TRAIN
    if max_len:
        cfg = dataclasses.replace(cfg, max_len=max_len)
    train_path = train_path or (config.SPLITS_DIR / "sft_train.jsonl")
    if not train_path.exists():
        print(f"ERROR: missing {train_path} (run `revact split` first)")
        return 1
    rows = [json.loads(ln) for ln in train_path.open()]
    report = validate_rows(rows)
    print(f"[data] {report['n_rows']} rows | problems={report['n_problems']} "
          f"| decisions={report['decision_dist']} | actions={report['action_dist']}")
    for p in report["problems"]:
        print(f"  [invalid] {p}")
    if report["n_problems"]:
        print("ERROR: fix data problems before training.")
        return 1
    print(f"[cfg] model={cfg.model_path} lora_r={cfg.lora_r} epochs={cfg.epochs} "
          f"lr={cfg.lr} bs={cfg.batch_size}x{cfg.grad_accum} max_len={cfg.max_len}")
    if dry_run:
        print("[dry-run] data + config valid; no training started.")
        return 0
    return _train(rows, cfg, output_dir or cfg.output_dir)


def _train(rows: list[dict], cfg, output_dir: str) -> int:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)

    tok = AutoTokenizer.from_pretrained(cfg.model_path)
    pad_id = tok.pad_token_id or tok.eos_token_id

    def encode(msgs, gen_prompt=False):
        return tok.apply_chat_template(msgs, add_generation_prompt=gen_prompt,
                                       tokenize=True, return_dict=True)["input_ids"]

    ex, n_prefix_fail = [], 0
    for r in rows:
        msgs = r["messages"]
        full_ids = encode(msgs)
        if len(full_ids) > cfg.max_len:
            continue
        # Loss on EVERY assistant turn (multi-turn trajectories included):
        # unmask each span between template(msgs[:i]+gen_prompt) and
        # template(msgs[:i+1]). Relies on the chat template's prefix property;
        # rows where it fails are skipped, never silently mistrained.
        labels = [-100] * len(full_ids)
        ok = True
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            pre = encode(msgs[:i], gen_prompt=True)
            end = encode(msgs[:i + 1])
            if full_ids[:len(end)] != end or end[:len(pre)] != pre:
                ok = False
                break
            labels[len(pre):len(end)] = full_ids[len(pre):len(end)]
        if not ok:
            n_prefix_fail += 1
            continue
        ex.append({"input_ids": full_ids, "labels": labels})
    print(f"[data] usable SFT examples: {len(ex)}/{len(rows)} "
          f"(prefix-check failures: {n_prefix_fail})")
    ds = Dataset.from_list(ex)

    def collate(batch):
        m = max(len(b["input_ids"]) for b in batch)
        ids, lbl, att = [], [], []
        for b in batch:
            n = m - len(b["input_ids"])
            ids.append(b["input_ids"] + [pad_id] * n)
            lbl.append(b["labels"] + [-100] * n)
            att.append([1] * len(b["input_ids"]) + [0] * n)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lbl),
                "attention_mask": torch.tensor(att)}

    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    # long multi-turn sequences need activation checkpointing + bs=1 to fit
    bs = 1 if cfg.max_len > 4096 else cfg.batch_size
    args = TrainingArguments(
        output_dir=output_dir, per_device_train_batch_size=bs,
        gradient_accumulation_steps=cfg.grad_accum, num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5, save_strategy="no", bf16=True, report_to=[],
        remove_unused_columns=False)
    Trainer(model=model, args=args, train_dataset=ds, data_collator=collate).train()

    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    print(f"[done] LoRA adapter -> {output_dir}")
    return 0
