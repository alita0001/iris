"""S7 teacher conditional distillation: prose from a teacher, conclusions pinned.

The teacher (DeepSeek via an OpenAI-compatible endpoint) receives the GROUNDED
facts (real next-state effect, undo-measured reversibility + measured undo
path, oracle decision) as immutable ground truth and only writes natural
<observation>/<reasoning>/<prediction>/<rev_check> phrasing consistent with
them. A QC pass rejects any output that contradicts the pinned labels (e.g.
claims something is undoable when the grounded label is IRREVERSIBLE) and
retries; persistent failures are dropped, never patched by hand.

The prompt template lives in the prompt registry (prompts.get("teacher_
distill")) so the workbench can tune it without code changes.

SECURITY: the API key is read lazily from the env var named by
config.DISTILL_KEY_ENV; it is never a parameter, never logged, never written.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from .. import config, prompts

# QC: phrases that contradict a pinned label (checked case-insensitively).
_CONTRADICTIONS = {
    "IRREVERSIBLE": [r"\bcan be (undone|reversed|cancell?ed)\b", r"\breversible\b",
                     r"\beasily undone\b", r"\bcan (?:always )?undo\b"],
    "REVERSIBLE": [r"\bcannot be (undone|reversed|cancell?ed)\b", r"\birreversible\b",
                   r"\bpermanent\b", r"\bno way (?:to )?undo\b"],
    "PARTIALLY_RECOVERABLE": [r"\bfully (undone|reversible)\b",
                              r"\bcompletely (undone|reversible)\b"],
    "UNKNOWN": [],
}
_REQUIRED_TAGS = ("<observation>", "<reasoning>", "<prediction>", "<rev_check>")


def qc_check(text: str, reversibility: str, decision: str) -> Optional[str]:
    """Return a rejection reason, or None when the prose passes QC."""
    for tag in _REQUIRED_TAGS:
        if tag not in text:
            return f"missing tag {tag}"
    low = text.lower()
    for pat in _CONTRADICTIONS.get(reversibility, []):
        if re.search(pat, low):
            return f"contradicts pinned reversibility={reversibility} (matched {pat!r})"
    if decision in ("AVOID", "CONFIRM") and re.search(
            r"<reasoning>[^\n]*\b(i (?:can|will) proceed|safe to (?:execute|click) it now)\b", low):
        return f"claims safe-to-proceed but decision={decision}"
    return None


class DeepSeekClient:
    """Minimal OpenAI-compatible chat client (stdlib urllib only)."""

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 key_env: str | None = None, timeout: int = 90):
        self.model = model or config.DISTILL_MODEL
        self.base_url = (base_url or config.DISTILL_BASE_URL).rstrip("/")
        self.key_env = key_env or config.DISTILL_KEY_ENV
        self.timeout = timeout

    def _key(self) -> str:
        key = os.environ.get(self.key_env, "").strip()
        if not key:
            raise RuntimeError(f"API key not found: export {self.key_env}=... first")
        return key

    def complete(self, prompt: str, max_tokens: int = 400) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7, "max_tokens": max_tokens,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self._key()}"},
            method="POST")
        last = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                msg = data["choices"][0].get("message", {}) or {}
                return (msg.get("content") or msg.get("reasoning_content") or "").strip()
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"teacher call failed: {last}")


def distill_sample(row: dict, complete: Callable[[str], str],
                   max_retries: int | None = None) -> dict:
    """Distill ONE assembled SFT row. Returns a report dict; on success the
    row's assistant message has teacher prose with the labels untouched."""
    max_retries = max_retries if max_retries is not None else config.DISTILL_MAX_RETRIES
    meta = row["meta"]
    asst = row["messages"][-1]["content"]
    goal = prompts.parse_user(row["messages"][1]["content"])["goal"]
    risky = re.search(r"offers a '([^']+)' control", asst)
    effect = re.search(r"<prediction>\s*([^\n]+)", asst)
    undo = re.search(r"<undo>\s*([^\n]+)", asst)
    prompt = prompts.get("teacher_distill").format(
        goal=goal,
        risky_text=risky.group(1) if risky else meta["action_type"],
        effect=(effect.group(1).strip() if effect else meta["action_type"]),
        undo=(undo.group(1).strip() if undo else "unverified"),
        reversibility=meta["reversibility"], decision=meta["decision"])

    for attempt in range(1, max_retries + 1):
        text = complete(prompt)
        reason = qc_check(text, meta["reversibility"], meta["decision"])
        if reason is None:
            new_asst = _splice(asst, text)
            row["messages"][-1]["content"] = new_asst
            row["meta"]["prose_source"] = "teacher"
            return {"ok": True, "attempts": attempt}
    return {"ok": False, "attempts": max_retries, "last_reason": reason}


def _splice(asst: str, teacher_text: str) -> str:
    """Replace the four prose fields; keep <reversibility>/<undo>/<decision>/
    <answer> (the pinned conclusions) byte-identical."""
    fields = {}
    for tag in ("observation", "reasoning", "prediction", "rev_check"):
        m = re.search(rf"<{tag}>\s*([^\n]+)", teacher_text)
        if m:
            fields[tag] = m.group(1).strip()
    out = asst
    for tag, val in fields.items():
        out = re.sub(rf"(<{tag}>)[^\n]*", lambda mm, v=val: f"{mm.group(1)} {v}", out,
                     count=1)
    return out


def run(in_path: Path | None = None, out_path: Path | None = None,
        limit: int = 10, client: Optional[DeepSeekClient] = None) -> int:
    """Distill up to `limit` samples (smoke-sized by default to control cost)."""
    in_path = in_path or config.SFT_PATH
    out_path = out_path or (config.TRAIN_DIR / "sft" / "revact_sft_distilled.jsonl")
    if not in_path.exists():
        print(f"ERROR: missing {in_path} (run `revact assemble` first)")
        return 1
    client = client or DeepSeekClient()
    rows = [json.loads(ln) for ln in in_path.open()][:limit]
    ok = failed = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            rep = distill_sample(row, client.complete)
            if rep["ok"]:
                ok += 1
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                failed += 1
                print(f"  [drop] {row['sample_id']}: {rep['last_reason']}")
    print(f"[distill] ok={ok} dropped={failed} -> {out_path}")
    return 0 if ok else 1
