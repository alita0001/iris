"""LoRA SFT of Qwen2.5 on RevAct/IRIS sequences (completion-only loss).

`run(dry_run=True)` loads the real tokenizer but never imports torch or starts
training.  The real path lazily imports GPU dependencies (run in ``agentlab``).
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .. import config, prompts
from ..envs.obs_utils import action_bid, bid_is_visible
from ..grounding.schema import EFFECT_STATUSES, RECOVERY_STATUSES
from .validators import (formal_completion_reasons, formal_point_join_problems,
                         iris_tag_errors, percentile)

_TAGS = ("<observation>", "<reasoning>", "<prediction>", "<rev_check>",
         "<reversibility>", "<undo>", "<decision>", "<answer>")
_DECISIONS = ("EXECUTE", "VERIFY", "CONFIRM", "AVOID")


def _roles_ok(roles: list) -> bool:
    """system, then strictly alternating user/assistant, ending on assistant.
    New formal rows are exactly three messages; alternating legacy artifacts
    remain parseable so validators can report their concrete defects."""
    if len(roles) < 3 or roles[0] != "system" or roles[-1] != "assistant":
        return False
    body = roles[1:]
    return all(r == ("user" if i % 2 == 0 else "assistant")
               for i, r in enumerate(body))


def validate_rows(rows: list[dict]) -> dict:
    """Schema/format validation shared by --dry-run and the real path.

    The DECISION turn (final assistant message) must carry the full think
    block. Formal rows use the shared stateless policy topology."""
    problems = []
    decisions, actions = {}, {}
    n_formal = 0
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
        formal = bool(meta.get("formal_dataset"))
        n_formal += int(formal)
        if formal and roles != ["system", "user", "assistant"]:
            problems.append(f"{rid}: formal row must use stateless policy topology")
        if formal and (meta.get("is_mock") is not False or
                       meta.get("collector_success") is not True):
            problems.append(f"{rid}: formal row requires is_mock=false and "
                            "collector_success=true")
        if formal:
            problems.extend(f"{rid}: {reason}"
                            for reason in formal_completion_reasons(r))
            for key in ("probe_point_id", "state_id", "action_instance_id"):
                if not meta.get(key):
                    problems.append(f"{rid}: formal row lacks {key}")
            if meta.get("prediction_source") != "probe_transition":
                problems.append(f"{rid}: formal prediction_source must be "
                                "probe_transition")
            if meta.get("undo_source") != "probe_point_id":
                problems.append(f"{rid}: formal undo_source must be probe_point_id")
            if meta.get("effect_status") not in EFFECT_STATUSES:
                problems.append(f"{rid}: invalid/missing canonical effect_status")
            if meta.get("recovery_status") not in RECOVERY_STATUSES:
                problems.append(f"{rid}: invalid/missing canonical recovery_status")
            if meta.get("history_source") != "trajectory":
                problems.append(f"{rid}: formal history_source must be trajectory")
            turn_types = meta.get("assistant_turn_types")
            if turn_types is None and meta.get("turn_type"):
                turn_types = [meta["turn_type"]]
            if not isinstance(turn_types, list) or len(turn_types) != len(msgs[2::2]):
                problems.append(
                    f"{rid}: formal row must explicitly type every assistant turn")
            else:
                allowed = {"routine", "state_changing", "decision"}
                for turn_no, (turn_type, message) in enumerate(
                        zip(turn_types, msgs[2::2]), start=1):
                    if turn_type not in allowed:
                        problems.append(
                            f"{rid}: assistant turn {turn_no} has bad turn_type {turn_type!r}")
                    if turn_type in {"state_changing", "decision"}:
                        for error in iris_tag_errors(message.get("content", "")):
                            problems.append(
                                f"{rid}: assistant turn {turn_no} {error}")
        current_obs = prompts.parse_observation_message(
            msgs[-2].get("content", ""))
        risky = meta.get("risky_raw_action", "")
        supervised_bids = {action_bid(risky)} if action_bid(risky) else set()
        for message in msgs[2::2]:
            answer = re.search(r"<answer>\s*([^\n]+)",
                               message.get("content", ""))
            if answer and action_bid(answer.group(1)):
                supervised_bids.add(action_bid(answer.group(1)))
        for bid in sorted(supervised_bids):
            if not bid_is_visible(current_obs, bid or ""):
                problems.append(
                    f"{rid}: supervised click bid absent from input: [{bid}]")
        decisions[meta.get("decision", "?")] = decisions.get(meta.get("decision", "?"), 0) + 1
        actions[meta.get("action_type", "?")] = actions.get(meta.get("action_type", "?"), 0) + 1
    if 0 < n_formal < len(rows):
        problems.append(
            f"dataset mixes {n_formal} formal and {len(rows) - n_formal} legacy rows")
    return {"n_rows": len(rows), "n_problems": len(problems),
            "problems": problems[:20], "decision_dist": decisions,
            "action_dist": actions, "n_formal": n_formal}


def _token_ids(tokenizer: Any, messages: list[dict],
               *, generation_prompt: bool = False) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=generation_prompt, tokenize=True,
        return_dict=True)
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    # Some tokenizers return a one-item batch even without tensors.
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(encoded)


def completion_encoding_report(rows: list[dict], tokenizer: Any,
                               max_len: int,
                               *, include_examples: bool = False) -> dict:
    """Tokenise exactly as training does and audit completion-only masks.

    Overlength and chat-template prefix failures are reported by sample id and
    are never silently skipped.  The same routine is used by ``--dry-run`` and
    the real trainer, eliminating dry-run/training drift.
    """
    lengths: list[int] = []
    examples: list[dict] = []
    dropped: list[str] = []
    prefix_failures: list[str] = []
    zero_completion: list[str] = []
    supervised_tokens = 0
    for row_no, row in enumerate(rows):
        rid = str(row.get("sample_id", f"row{row_no}"))
        messages = row["messages"]
        full_ids = _token_ids(tokenizer, messages)
        lengths.append(len(full_ids))
        if len(full_ids) > max_len:
            dropped.append(rid)
            continue
        labels = [-100] * len(full_ids)
        valid_prefix = True
        for index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            prefix = _token_ids(tokenizer, messages[:index],
                                generation_prompt=True)
            end = _token_ids(tokenizer, messages[:index + 1])
            if full_ids[:len(end)] != end or end[:len(prefix)] != prefix:
                valid_prefix = False
                break
            labels[len(prefix):len(end)] = full_ids[len(prefix):len(end)]
        if not valid_prefix:
            prefix_failures.append(rid)
            continue
        n_supervised = sum(label != -100 for label in labels)
        if n_supervised == 0:
            zero_completion.append(rid)
            continue
        supervised_tokens += n_supervised
        if include_examples:
            examples.append({"input_ids": full_ids, "labels": labels})
    report = {
        "n_rows": len(rows),
        "token_p50": percentile(lengths, .50),
        "token_p95": percentile(lengths, .95),
        "token_max": max(lengths, default=0),
        "max_len": max_len,
        "n_dropped": len(dropped),
        "dropped_sample_ids": dropped,
        "n_prefix_failures": len(prefix_failures),
        "prefix_failure_sample_ids": prefix_failures,
        "n_zero_completion": len(zero_completion),
        "zero_completion_sample_ids": zero_completion,
        "n_usable": len(rows) - len(dropped) - len(prefix_failures) - len(zero_completion),
        "supervised_completion_tokens": supervised_tokens,
        # Labels are initialised masked and only assistant spans are unmasked.
        "completion_only": True,
    }
    if include_examples:
        report["examples"] = examples
    return report


def run(train_path: Path | None = None, dry_run: bool = False,
        output_dir: str | None = None, max_len: int | None = None,
        *, allow_legacy: bool = False) -> int:
    import dataclasses

    cfg = config.TRAIN
    if max_len:
        cfg = dataclasses.replace(cfg, max_len=max_len)
    train_path = train_path or config.FORMAL_SFT_TRAIN_PATH
    if not train_path.exists():
        print(f"ERROR: missing {train_path} (run `revact split` first)")
        return 1
    rows = [json.loads(ln) for ln in train_path.open()]
    if not rows:
        print(f"ERROR: {train_path} contains zero rows; refusing a vacuous "
              "training/dry-run success.")
        return 1
    report = validate_rows(rows)
    point_problems = formal_point_join_problems(rows, config.DATA_ROOT)
    if point_problems:
        report["problems"] = (report["problems"] + point_problems)[:20]
        report["n_problems"] += len(point_problems)
    print(f"[data] {report['n_rows']} rows | problems={report['n_problems']} "
          f"| decisions={report['decision_dist']} | actions={report['action_dist']}")
    if report["n_formal"] != len(rows):
        print("[mode] LEGACY/DEVELOPMENT: point-level provenance gates are not "
              "satisfied; this is not a formal training export.")
    for p in report["problems"]:
        print(f"  [invalid] {p}")
    if report["n_problems"]:
        print("ERROR: fix data problems before training.")
        return 1
    if report["n_formal"] != len(rows):
        if not (dry_run and allow_legacy):
            print("ERROR: formal SFT is fail-closed. Legacy rows may only be "
                  "audited with both --dry-run and --allow-legacy; they can "
                  "never enter the real trainer.")
            return 1
        print("[legacy-audit] explicitly authorised validation only; the real "
              "trainer remains disabled for these rows.")
    print(f"[cfg] model={cfg.model_path} lora_r={cfg.lora_r} epochs={cfg.epochs} "
          f"lr={cfg.lr} bs={cfg.batch_size}x{cfg.grad_accum} max_len={cfg.max_len}")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_path, local_files_only=True)
        token_report = completion_encoding_report(rows, tokenizer, cfg.max_len)
    except Exception as exc:
        print(f"ERROR: tokenizer audit failed: {type(exc).__name__}: {exc}")
        return 1
    print("[tokens] "
          f"p50={token_report['token_p50']:.1f} "
          f"p95={token_report['token_p95']:.1f} "
          f"max={token_report['token_max']} max_len={cfg.max_len} "
          f"drop={token_report['n_dropped']} "
          f"prefix_fail={token_report['n_prefix_failures']} "
          f"completion_only={token_report['completion_only']}")
    encoding_failures = (token_report["n_dropped"] +
                         token_report["n_prefix_failures"] +
                         token_report["n_zero_completion"])
    if encoding_failures:
        for key in ("dropped_sample_ids", "prefix_failure_sample_ids",
                    "zero_completion_sample_ids"):
            if token_report[key]:
                print(f"  [{key}] {token_report[key][:20]}")
        print("ERROR: tokenizer drop/prefix/empty-completion count is non-zero; "
              "refusing to train.")
        return 1
    if dry_run:
        scope = "formal" if report["n_formal"] == len(rows) else "legacy-audit"
        print(f"[dry-run] {scope} data + tokenizer + completion mask valid; "
              "no training started.")
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

    encoding = completion_encoding_report(
        rows, tok, cfg.max_len, include_examples=True)
    if (encoding["n_dropped"] or encoding["n_prefix_failures"] or
            encoding["n_zero_completion"]):
        raise RuntimeError("encoding audit changed after dry-run; refusing training")
    ex = encoding.pop("examples")
    print(f"[data] usable completion-only SFT examples: {len(ex)}/{len(rows)}")
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
