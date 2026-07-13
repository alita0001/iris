"""LoRA DPO on RevAct/IRIS preference pairs (TRL DPOTrainer).

Rows ({revact_dpo,revact_dpo_multiturn,splits/dpo_train}.jsonl):
    {"pair_id", "prompt": [messages... ending on a user turn],
     "chosen": str, "rejected": str, "meta": {...}}

Single-step and trajectory-conditioned pairs load identically: new formal
prompts are the same stateless ``system+user`` input used by deployment;
chosen/rejected are the two candidate assistant completions.

`run(dry_run=True)` validates data + config WITHOUT importing torch; the real
path runs in the ``agentlab`` environment with torch/trl/peft and a GPU.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .. import config, prompts
from ..data.candidates import SOURCE_ON_POLICY, SOURCE_SYNTHETIC
from ..envs.obs_utils import action_bid, bid_is_visible
from ..grounding.schema import EFFECT_STATUSES, RECOVERY_STATUSES
from .validators import (answer_text, decision_answer_consistent,
                         formal_completion_reasons,
                         formal_negative_candidate_reasons,
                         formal_point_join_problems,
                         iris_tag_errors,
                         parse_decision, percentile)

_ROLES_BODY = ("user", "assistant")
_NEGATIVE_SOURCES = frozenset({
    SOURCE_SYNTHETIC, "legal_candidate", SOURCE_ON_POLICY,
})
_DEPLOYMENT_NEGATIVE_SOURCES = frozenset({
    "legal_candidate", SOURCE_ON_POLICY,
})


def pair_encoding_report(rows: list[dict], tokenizer, max_len: int) -> dict:
    """Audit both preference branches exactly as conversational DPO sees them."""
    from .sft import _token_ids

    lengths: list[int] = []
    dropped: list[str] = []
    for i, row in enumerate(rows):
        rid = str(row.get("pair_id", f"row{i}"))
        pair_lengths = []
        for side in ("chosen", "rejected"):
            messages = list(row["prompt"]) + [
                {"role": "assistant", "content": row[side]}]
            length = len(_token_ids(tokenizer, messages))
            lengths.append(length)
            pair_lengths.append(length)
        if max(pair_lengths, default=0) > max_len:
            dropped.append(rid)
    return {
        "n_pairs": len(rows), "n_sequences": len(lengths),
        "token_p50": percentile(lengths, .50),
        "token_p95": percentile(lengths, .95),
        "token_max": max(lengths, default=0), "max_len": max_len,
        "n_dropped": len(dropped), "dropped_pair_ids": dropped,
    }


def validate_rows(rows: list[dict], *, formal: bool | None = None,
                  min_deployment_negative_ratio: float = .50) -> dict:
    """Validate DPO pairs and audit where the rejected answer came from.

    Legacy files (no ``formal_dataset`` marker) remain readable.  A formal
    export is fail-closed: every negative has a declared source and at least
    half are legal snapshot candidates or errors sampled from a policy, rather
    than label-flipped template prose.
    """
    problems = []
    pair_types: dict[str, int] = {}
    source_dist: dict[str, int] = {}
    n_formal = 0
    n_deployment = 0
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
        meta = r.get("meta") or {}
        row_formal = (bool(meta.get("formal_dataset")) if formal is None
                      else bool(formal))
        if row_formal:
            n_formal += 1
        if row_formal and roles != ["system", "user"]:
            problems.append(f"{rid}: formal pair must use stateless policy topology")
        if row_formal and (meta.get("is_mock") is not False or
                           meta.get("collector_success") is not True):
            problems.append(f"{rid}: formal pair requires is_mock=false and "
                            "collector_success=true")
        risky = meta.get("risky_raw_action", "")
        supervised_bids = {action_bid(risky)} if action_bid(risky) else set()
        for completion in (r.get("chosen", ""), r.get("rejected", "")):
            answer = re.search(r"<answer>\s*([^\n]+)", completion)
            if answer and action_bid(answer.group(1)):
                supervised_bids.add(action_bid(answer.group(1)))
        current_obs = prompts.parse_observation_message(
            msgs[-1].get("content", ""))
        for bid in sorted(supervised_bids):
            if not bid_is_visible(current_obs, bid or ""):
                problems.append(
                    f"{rid}: supervised click bid absent from input: [{bid}]")
        pt = meta.get("pair_type", "?")
        pair_types[pt] = pair_types.get(pt, 0) + 1
        source = str(meta.get("negative_source") or "legacy_unspecified")
        source_dist[source] = source_dist.get(source, 0) + 1
        if row_formal:
            problems.extend(f"{rid}: {reason}"
                            for reason in formal_completion_reasons(r))
            problems.extend(f"{rid}: {reason}"
                            for reason in formal_negative_candidate_reasons(r))
            for key in ("probe_point_id", "state_id", "action_instance_id"):
                if not meta.get(key):
                    problems.append(f"{rid}: formal pair lacks {key}")
            if meta.get("prediction_source") != "probe_transition":
                problems.append(
                    f"{rid}: formal prediction_source must be probe_transition")
            if meta.get("undo_source") != "probe_point_id":
                problems.append(f"{rid}: formal undo_source must be probe_point_id")
            if meta.get("history_source") != "trajectory":
                problems.append(f"{rid}: formal history_source must be trajectory")
            if not isinstance(meta.get("normative_risk"), bool):
                problems.append(f"{rid}: formal pair lacks boolean normative_risk")
            if meta.get("effect_status") not in EFFECT_STATUSES:
                problems.append(f"{rid}: invalid/missing canonical effect_status")
            if meta.get("recovery_status") not in RECOVERY_STATUSES:
                problems.append(f"{rid}: invalid/missing canonical recovery_status")
            for side in ("chosen", "rejected"):
                for error in iris_tag_errors(r.get(side, "")):
                    problems.append(f"{rid}: formal {side} {error}")
            if parse_decision(r["chosen"]) != meta.get("decision"):
                problems.append(f"{rid}: chosen decision does not match pinned metadata")
            risky_action = meta.get("risky_action") or meta.get("risky_raw_action")
            if not risky_action:
                problems.append(f"{rid}: formal pair lacks structured/risky action identity")
            elif not decision_answer_consistent(
                    str(meta.get("decision") or ""), answer_text(r["chosen"]),
                    risky_action):
                problems.append(f"{rid}: chosen decision/answer is inconsistent")
            if source not in _NEGATIVE_SOURCES:
                problems.append(f"{rid}: invalid/missing formal negative_source {source!r}")
            if source == "legal_candidate":
                if meta.get("legal_at_snapshot") is not True:
                    problems.append(f"{rid}: legal_candidate was not legal_at_snapshot=true")
                if not meta.get("negative_candidate_id"):
                    problems.append(
                        f"{rid}: legal_candidate lacks negative_candidate_id")
                if not meta.get("negative_candidate_snapshot_hash"):
                    problems.append(
                        f"{rid}: legal_candidate lacks "
                        "negative_candidate_snapshot_hash")
            if source == SOURCE_ON_POLICY and not (
                    meta.get("policy_model_version") or
                    meta.get("proposer_model_version")):
                problems.append(f"{rid}: {source} lacks policy model/version provenance")
            if source in _DEPLOYMENT_NEGATIVE_SOURCES:
                n_deployment += 1
    ratio = n_deployment / n_formal if n_formal else None
    effective_formal = formal is True or n_formal > 0
    if formal is None and 0 < n_formal < len(rows):
        problems.append(
            f"dataset mixes {n_formal} formal and {len(rows) - n_formal} legacy pairs")
    if effective_formal and not n_formal:
        problems.append("formal DPO requested but no rows are marked formal")
    if ratio is not None and ratio < min_deployment_negative_ratio:
        problems.append(
            "formal deployment/on-policy negative ratio "
            f"{ratio:.3f} < {min_deployment_negative_ratio:.3f} "
            f"({n_deployment}/{n_formal})")
    return {"n_rows": len(rows), "n_problems": len(problems),
            "problems": problems[:20], "pair_types": pair_types,
            "negative_source_dist": source_dist,
            "n_formal": n_formal,
            "n_deployment_negatives": n_deployment,
            "deployment_negative_ratio": ratio,
            "min_deployment_negative_ratio": min_deployment_negative_ratio}


def run(train_path: Path | None = None, dry_run: bool = False,
        output_dir: str | None = None, beta: float = 0.1,
        epochs: float | None = None, max_steps: int = -1,
        adapter: str = "", *, allow_legacy: bool = False) -> int:
    cfg = config.TRAIN
    train_path = train_path or config.FORMAL_DPO_TRAIN_PATH
    if not train_path.exists():
        print(f"ERROR: missing {train_path} (run `revact assemble`+`split` first)")
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
    print(f"[data] {report['n_rows']} pairs | problems={report['n_problems']} "
          f"| pair_types={report['pair_types']} "
          f"| negative_sources={report['negative_source_dist']}")
    if report["n_formal"] != len(rows):
        print("[mode] LEGACY/DEVELOPMENT: negative provenance/50% gates are not "
              "satisfied; this is not a formal DPO export.")
    if report["n_formal"]:
        print("[formal-gate] deployment/on-policy negatives="
              f"{report['n_deployment_negatives']}/{report['n_formal']} "
              f"({report['deployment_negative_ratio']:.1%}; "
              f"required >= {report['min_deployment_negative_ratio']:.0%})")
    for p in report["problems"]:
        print(f"  [invalid] {p}")
    if report["n_problems"]:
        print("ERROR: fix data problems before training.")
        return 1
    if report["n_formal"] != len(rows):
        if not (dry_run and allow_legacy):
            print("ERROR: formal DPO is fail-closed. Legacy pairs may only be "
                  "audited with both --dry-run and --allow-legacy; they can "
                  "never enter the real trainer.")
            return 1
        print("[legacy-audit] explicitly authorised validation only; the real "
              "trainer remains disabled for these pairs.")
    print(f"[cfg] model={cfg.model_path} lora_r={cfg.lora_r} beta={beta} "
          f"max_len={cfg.max_len} max_steps={max_steps} "
          f"adapter={adapter or '(none: cold start)'}")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_path, local_files_only=True)
        token_report = pair_encoding_report(rows, tokenizer, cfg.max_len)
    except Exception as exc:
        print(f"ERROR: offline tokenizer audit failed: {type(exc).__name__}: {exc}")
        return 1
    print("[tokens] pair-sequences="
          f"{token_report['n_sequences']} p50={token_report['token_p50']:.1f} "
          f"p95={token_report['token_p95']:.1f} "
          f"max={token_report['token_max']} max_len={cfg.max_len} "
          f"drop={token_report['n_dropped']}")
    if token_report["n_dropped"]:
        print(f"  [dropped_pair_ids] {token_report['dropped_pair_ids'][:20]}")
        print("ERROR: DPO would truncate/drop a preference branch; refusing.")
        return 1
    if adapter and not Path(adapter).exists():
        print(f"ERROR: adapter path {adapter} not found")
        return 1
    if dry_run:
        scope = "formal" if report["n_formal"] == len(rows) else "legacy-audit"
        print(f"[dry-run] {scope} data + config valid; no training started.")
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
