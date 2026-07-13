"""Decision-accuracy evaluation on held-out samples (base vs SFT adapter).

HONESTY NOTE (from the 2026-07 audit): with few action classes the decision is
nearly a deterministic function of (goal variant x action type), so a high
score here measures format compliance + goal-variant classification, NOT a
learned concept of reversibility. Treat this as a pipeline check; the real
evidence is the cross-action-class split once multiple grounded classes exist.

`run(dry_run=True)` validates the test file without importing torch.
"""
from __future__ import annotations

import collections
import json
import math
import re
from pathlib import Path

from .. import config
from ..train.validators import formal_point_join_problems

_DEC_RE = re.compile(r"<decision>\s*([A-Z_]+)")


def parse_decision(text: str) -> str | None:
    m = _DEC_RE.search(text or "")
    return m.group(1) if m else None


def gold_completion_budget(rows: list[dict], tokenizer,
                           margin_tokens: int = 16) -> dict:
    """Nearest-rank p99 budget over gold assistant completions."""
    lengths = []
    for row in rows:
        assistant = row["messages"][-1]
        if assistant.get("role") != "assistant":
            raise ValueError(f"{row.get('sample_id')}: final message is not assistant")
        encoded = tokenizer(assistant.get("content", ""), add_special_tokens=False)
        ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        lengths.append(len(ids))
    ordered = sorted(lengths)
    if not ordered:
        return {"n": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0,
                "recommended_max_new_tokens": 0}
    def percentile(q: float) -> int:
        return ordered[max(0, math.ceil(q * len(ordered)) - 1)]
    return {
        "n": len(ordered), "p50": percentile(.50), "p95": percentile(.95),
        "p99": percentile(.99), "max": ordered[-1],
        "recommended_max_new_tokens": percentile(.99) + margin_tokens,
    }


def run(test_path: Path | None = None, adapter: str = "",
        max_new_tokens: int = 0, dry_run: bool = False,
        *, allow_legacy: bool = False) -> int:
    test_path = test_path or ((config.SPLITS_DIR if allow_legacy else
                              config.FORMAL_SPLITS_DIR) / "sft_test.jsonl")
    if not test_path.exists():
        print(f"ERROR: missing {test_path} (run `revact split` first)")
        return 1
    rows = [json.loads(ln) for ln in test_path.open()]
    if not rows:
        print(f"ERROR: {test_path} contains zero rows; refusing a vacuous eval.")
        return 1
    n_formal = sum((row.get("meta") or {}).get("formal_dataset") is True
                   for row in rows)
    if n_formal != len(rows):
        if not allow_legacy:
            print("ERROR: formal decision eval rejects legacy/mixed rows; use the "
                  "explicit --legacy-development audit mode for frozen pilot data.")
            return 1
        print("[mode] LEGACY-DEVELOPMENT: results are pipeline diagnostics, not "
              "formal point-grounded evidence.")
    point_problems = formal_point_join_problems(rows, config.DATA_ROOT)
    if point_problems:
        for problem in point_problems[:20]:
            print(f"  [invalid] {problem}")
        print("ERROR: formal evaluation provenance failed exact point/manifest join.")
        return 1
    cells = collections.Counter(
        (r["meta"]["action_type"], r["meta"]["variant"]) for r in rows)
    print(f"[data] {len(rows)} held-out samples; cells={dict(cells)}")
    thin = [c for c, n in cells.items() if n < 30]
    if thin:
        print(f"[warn] cells with n<30: report counts/intervals only; no effect "
              f"conclusion is permitted: {thin}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.TRAIN.model_path)
    budget = gold_completion_budget(rows, tok)
    required = budget["recommended_max_new_tokens"]
    print(f"[generation-budget] {budget}")
    if max_new_tokens <= 0:
        max_new_tokens = required
    elif max_new_tokens < required:
        print(f"ERROR: max_new_tokens={max_new_tokens} is below gold p99+margin "
              f"requirement {required}; refusing truncation-biased eval")
        return 1
    if dry_run:
        print(f"[dry-run] test data valid; generation budget={max_new_tokens}; "
              "no generation started.")
        return 0
    return _evaluate(rows, adapter, max_new_tokens)


def _evaluate(rows: list[dict], adapter: str, max_new_tokens: int) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = config.TRAIN.model_path
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                                                 device_map="cuda")
    tag = "base"
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        tag = f"sft({adapter})"
    model.eval()

    n_ok = n_fmt = 0
    by_cell = collections.defaultdict(lambda: [0, 0])
    contrast = collections.defaultdict(collections.Counter)
    for r in rows:
        enc = tok.apply_chat_template(r["messages"][:-1], add_generation_prompt=True,
                                      tokenize=True, return_tensors="pt",
                                      return_dict=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = parse_decision(text) or "NONE"
        gold = r["meta"]["decision"]
        a, v = r["meta"]["action_type"], r["meta"]["variant"]
        n_fmt += int(pred != "NONE")
        n_ok += int(pred == gold)
        by_cell[(a, v)][0] += int(pred == gold)
        by_cell[(a, v)][1] += 1
        if v == "request":
            contrast[r["meta"]["reversibility"]][pred] += 1

    n = len(rows)
    print(f"\n=== decision eval [{tag}] on {n} held-out samples ===")
    print(f"format-valid (<decision> present): {n_fmt}/{n}")
    print(f"decision accuracy vs grounded oracle (all rows): {n_ok}/{n} = {n_ok/n:.2f}")
    if n_fmt:
        print(f"decision accuracy on format-valid rows only:   {n_ok}/{n_fmt} = {n_ok/n_fmt:.2f}"
              "   <- fairer for the untrained base model")
    print("per (action_type, variant):")
    for (a, v), (ok, tot) in sorted(by_cell.items()):
        print(f"  {a:22s} {v:11s}: {ok}/{tot}")
    print("reversibility contrast (variant=request -> predicted decision):")
    for rev, cnt in contrast.items():
        print(f"  {rev:22s}: {dict(cnt)}")
    return 0
