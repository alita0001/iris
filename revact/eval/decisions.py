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
import re
from pathlib import Path

from .. import config

_DEC_RE = re.compile(r"<decision>\s*([A-Z_]+)")


def parse_decision(text: str) -> str | None:
    m = _DEC_RE.search(text or "")
    return m.group(1) if m else None


def run(test_path: Path | None = None, adapter: str = "",
        max_new_tokens: int = 200, dry_run: bool = False) -> int:
    test_path = test_path or (config.SPLITS_DIR / "sft_test.jsonl")
    if not test_path.exists():
        print(f"ERROR: missing {test_path} (run `revact split` first)")
        return 1
    rows = [json.loads(ln) for ln in test_path.open()]
    cells = collections.Counter(
        (r["meta"]["action_type"], r["meta"]["variant"]) for r in rows)
    print(f"[data] {len(rows)} held-out samples; cells={dict(cells)}")
    thin = [c for c, n in cells.items() if n < 5]
    if thin:
        print(f"[warn] cells with n<5 (no statistical weight): {thin}")
    if dry_run:
        print("[dry-run] test data valid; no generation started.")
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
