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
import hashlib
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from .. import config, prompts
from .validators import (answer_text, decision_answer_consistent,
                         formal_completion_reasons, iris_tag_errors,
                         parse_action, parse_decision)

# QC: phrases that contradict a pinned label (checked case-insensitively).
_CONTRADICTIONS = {
    "IRREVERSIBLE": [r"\bcan be (undone|reversed|cancell?ed)\b", r"\breversible\b",
                     r"\beasily undone\b", r"\bcan (?:always )?undo\b"],
    "REVERSIBLE": [r"\bcannot be (undone|reversed|cancell?ed)\b", r"\birreversible\b",
                   r"\bpermanent\b", r"\bno way (?:to )?undo\b"],
    "PARTIALLY_RECOVERABLE": [r"\bfully (undone|reversible)\b",
                              r"\bcompletely (undone|reversible)\b"],
    "UNKNOWN": [],
    "NOT_RECOVERED_WITHIN_BUDGET": [
        r"\bcan be (undone|reversed|cancell?ed)\b", r"\breversible\b",
        r"\beasily undone\b", r"\bcan (?:always )?undo\b"],
    "RECOVERED": [r"\bcannot be (undone|reversed|cancell?ed)\b",
                  r"\birreversible\b", r"\bpermanent\b",
                  r"\bno way (?:to )?undo\b"],
    "PARTIALLY_RECOVERED": [r"\bfully (undone|reversible)\b",
                            r"\bcompletely (undone|reversible)\b"],
}
_REQUIRED_TAGS = ("<observation>", "<reasoning>", "<prediction>", "<rev_check>")


def evidence_token(kind: str, payload) -> str:
    """Content-address one evidence compartment without exposing full payloads."""
    if isinstance(payload, str):
        encoded = payload.encode("utf-8")
    else:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
    return f"[evidence:{kind}:{hashlib.sha256(encoded).hexdigest()[:12]}]"


def _unsupported_control_claim(text: str, input_observation: str) -> str:
    """Return a positively asserted UI control absent from the current AXTree."""
    patterns = (
        r"\b(?:a|an|the)\s+['\"]?([a-z][a-z0-9 _-]{0,40}?)['\"]?\s+"
        r"(?:button|link|control)\s+(?:exists|appears|is visible|is available)",
        r"\b(?:shows?|exposes?|provides?|contains?)\s+(?:a|an|the)?\s*"
        r"['\"]?([a-z][a-z0-9 _-]{0,40}?)['\"]?\s+(?:button|link|control)",
    )
    lower = text.lower()
    observed = input_observation.lower()
    for pattern in patterns:
        for match in re.finditer(pattern, lower):
            prefix = lower[max(0, match.start() - 16):match.start()]
            if re.search(r"\b(?:no|not|without|lacks?)\s*$", prefix):
                continue
            label = re.sub(r"\s+", " ", match.group(1)).strip(" '\"")
            if label and label not in observed:
                return label
    return ""


def qc_check(text: str, reversibility: str, decision: str, *,
             formal: bool = False, input_observation: str = "",
             post_signal_diff=None, undo_actions=None,
             undo_cost_steps=None, probe_evidence=None) -> Optional[str]:
    """Return a teacher-prose rejection reason, or ``None``.

    Formal prose must cite the exact evidence compartments it describes.  The
    citation is deliberately mechanical: it prevents a fluent teacher from
    inventing a currently visible control or presenting a learned mechanism as
    an observed post-transition fact.
    """
    structural = iris_tag_errors(text, require_complete=False)
    if structural:
        # Preserve the historical error wording consumed by the workbench.
        for tag in _REQUIRED_TAGS:
            if tag not in text:
                return f"missing tag {tag}"
        return structural[0]
    low = text.lower()
    for pat in _CONTRADICTIONS.get(reversibility, []):
        if re.search(pat, low):
            return f"contradicts pinned reversibility={reversibility} (matched {pat!r})"
    if decision in ("AVOID", "CONFIRM") and re.search(
            r"<reasoning>[^\n]*\b(i (?:can|will) proceed|safe to (?:execute|click) it now)\b", low):
        return f"claims safe-to-proceed but decision={decision}"
    if formal:
        if not input_observation:
            return "formal teacher QC lacks input observation evidence"
        if post_signal_diff in (None, "", {}, []):
            return "formal teacher QC lacks measured post_signal_diff"
        sections = {tag: _field(text, tag)
                    for tag in ("observation", "prediction", "rev_check")}
        input_ref = evidence_token("input", input_observation)
        post_ref = evidence_token("post_diff", post_signal_diff)
        undo_payload = {"actions": list(undo_actions or []),
                        "undo_cost_steps": undo_cost_steps}
        undo_ref = evidence_token("undo", undo_payload)
        probe_ref = (evidence_token("probe", probe_evidence)
                     if probe_evidence not in (None, "", {}, []) else "")
        if input_ref not in sections["observation"]:
            return f"observation lacks exact input evidence citation {input_ref}"
        if post_ref not in sections["prediction"]:
            return f"prediction lacks exact post_diff evidence citation {post_ref}"
        required_rev_ref = undo_ref if undo_actions else probe_ref
        if not required_rev_ref:
            return "formal rev_check lacks measured undo or probe evidence"
        if required_rev_ref not in sections["rev_check"]:
            kind = "undo" if undo_actions else "probe"
            return (f"rev_check lacks exact {kind} evidence citation "
                    f"{required_rev_ref}")
        for bid in re.findall(r"\[([^\]\s]+)\]", text):
            if bid.startswith("evidence:"):
                continue
            if f"[{bid}]" not in input_observation:
                return f"teacher cites control bid [{bid}] absent from input"
        unsupported = _unsupported_control_claim(
            sections["observation"] + " " + sections["rev_check"],
            input_observation)
        if unsupported:
            return f"teacher claims {unsupported!r} control absent from input"
        if reversibility in {"REVERSIBLE", "RECOVERED",
                             "PARTIALLY_RECOVERABLE", "PARTIALLY_RECOVERED"} \
                and not undo_actions:
            return "positive recovery prose lacks measured undo_actions"
    return None


def _field(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*([^\n]+)", text or "")
    return match.group(1).strip() if match else ""


def _action_token(action) -> str:
    parsed = parse_action(action)
    if not parsed:
        return ""
    return f"{parsed.name}:{parsed.bid or ''}"


def qc_source_sample(row: dict) -> Optional[str]:
    """Validate immutable point evidence and pinned output before API use."""
    meta = row.get("meta") or {}
    messages = row.get("messages") or []
    if not messages:
        return "sample has no messages"
    text = messages[-1].get("content", "")
    formal = bool(meta.get("formal_dataset"))
    if formal:
        shared = formal_completion_reasons(row)
        if shared:
            return shared[0]
        errors = iris_tag_errors(text)
        if errors:
            return errors[0]
        for key in ("probe_point_id", "state_id", "action_instance_id"):
            if not meta.get(key):
                return f"formal sample lacks {key}"
        if meta.get("prediction_source") != "probe_transition":
            return "formal prediction_source must be probe_transition"
        if meta.get("effect_status") not in {"CHANGED", "NO_EFFECT", "UNKNOWN"}:
            return "formal sample lacks canonical effect_status"
        if meta.get("recovery_status") not in {
                "RECOVERED", "PARTIALLY_RECOVERED",
                "NOT_RECOVERED_WITHIN_BUDGET", "UNKNOWN"}:
            return "formal sample lacks canonical recovery_status"
        post_diff = (meta.get("post_signal_diff") or
                     meta.get("transition_diff") or
                     (meta.get("evidence") or {}).get("post_signal_diff"))
        if post_diff in (None, "", {}, []):
            return "formal sample lacks measured post_signal_diff"
        if meta.get("undo_source") != "probe_point_id":
            return "formal undo_source must be probe_point_id"
        if meta.get("undo_source_probe_point_id") != meta.get("probe_point_id"):
            return "formal undo source does not match probe_point_id"
        user = next((message.get("content", "")
                     for message in reversed(messages[:-1])
                     if message.get("role") == "user"), "")
        current_obs = prompts.parse_observation_message(user)
        risky_parsed = parse_action(
            meta.get("risky_action") or meta.get("risky_raw_action"))
        if risky_parsed and risky_parsed.bid and \
                f"[{risky_parsed.bid}]" not in current_obs:
            return f"formal risky bid [{risky_parsed.bid}] absent from input"
    expected_recovery = str(meta.get("recovery_status") or
                            meta.get("reversibility") or "")
    actual_recovery = _field(text, "reversibility").split()
    if not actual_recovery or actual_recovery[0] != expected_recovery:
        return "output reversibility does not match pinned metadata"
    expected_decision = str(meta.get("decision") or "")
    if parse_decision(text) != expected_decision:
        return "output decision does not match pinned metadata"
    risky = meta.get("risky_action") or meta.get("risky_raw_action")
    if not decision_answer_consistent(
            expected_decision, answer_text(text), risky):
        return "decision and <answer> action are inconsistent"
    undo_actions = meta.get("undo_actions") or []
    undo_text = _field(text, "undo")
    if formal and expected_recovery in {
            "REVERSIBLE", "RECOVERED", "PARTIALLY_RECOVERABLE",
            "PARTIALLY_RECOVERED"}:
        if not undo_actions:
            return "positive recovery sample lacks measured undo_actions"
        missing = [token for token in map(_action_token, undo_actions)
                   if token and token not in {
                       _action_token(candidate)
                       for candidate in re.findall(r"[A-Za-z_]\w*\([^\n;]+?\)", undo_text)
                   }]
        if missing:
            return f"<undo> omits measured undo action(s) {missing}"
    undo_cost = meta.get("undo_cost_steps")
    if formal and undo_cost is not None and not re.search(
            rf"(?:undo_)?cost_steps\s*=\s*{int(undo_cost)}\b", undo_text):
        return f"<undo> omits pinned undo_cost_steps={undo_cost}"
    return None


def qc_full_sample(row: dict) -> Optional[str]:
    """Validate source truth plus teacher prose against exact evidence refs."""
    source_error = qc_source_sample(row)
    if source_error:
        return source_error
    meta = row.get("meta") or {}
    if not bool(meta.get("formal_dataset")):
        return None
    messages = row.get("messages") or []
    user = next((message.get("content", "")
                 for message in reversed(messages[:-1])
                 if message.get("role") == "user"), "")
    post_diff = (meta.get("post_signal_diff") or
                 meta.get("transition_diff") or
                 (meta.get("evidence") or {}).get("post_signal_diff"))
    undo_actions = (meta.get("undo_actions") or
                    (meta.get("evidence") or {}).get("undo_actions") or [])
    text = messages[-1].get("content", "")
    prose = "\n".join(
        f"<{tag}> {_field(text, tag)}"
        for tag in ("observation", "reasoning", "prediction", "rev_check"))
    return qc_check(
        prose, str(meta.get("recovery_status")), str(meta.get("decision")),
        formal=True, input_observation=prompts.parse_observation_message(user),
        post_signal_diff=post_diff, undo_actions=undo_actions,
        undo_cost_steps=meta.get("undo_cost_steps"),
        probe_evidence=meta.get("evidence"))


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
    user_message = next((m for m in reversed(row["messages"][:-1])
                         if m.get("role") == "user"), {})
    parsed_user = prompts.parse_user(user_message.get("content", ""))
    goal = parsed_user["goal"]
    formal = bool(meta.get("formal_dataset"))
    if formal:
        preflight = qc_source_sample(row)
        if preflight:
            meta["prose_source"] = "template_fallback"
            meta["teacher_qc_status"] = "source_rejected"
            meta["teacher_qc_reason"] = preflight
            return {"ok": False, "attempts": 0, "last_reason": preflight,
                    "fallback": "template_fallback"}
    risky = re.search(r"offers a '([^']+)' control", asst)
    effect = re.search(r"<prediction>\s*([^\n]+)", asst)
    undo = re.search(r"<undo>\s*([^\n]+)", asst)
    post_diff = (meta.get("post_signal_diff") or meta.get("transition_diff") or
                 (meta.get("evidence") or {}).get("post_signal_diff"))
    undo_actions = (meta.get("undo_actions") or
                    (meta.get("evidence") or {}).get("undo_actions") or [])
    effect_fact = (json.dumps(post_diff, ensure_ascii=False, sort_keys=True)
                   if formal else
                   (effect.group(1).strip() if effect else meta["action_type"]))
    undo_fact = (json.dumps({"actions": undo_actions,
                             "undo_cost_steps": meta.get("undo_cost_steps")},
                            ensure_ascii=False, sort_keys=True)
                 if formal else
                 (undo.group(1).strip() if undo else "unverified"))
    recovery = str(meta.get("recovery_status") or
                   meta.get("reversibility") or "UNKNOWN")
    prompt = prompts.get("teacher_distill").format(
        goal=goal,
        risky_text=risky.group(1) if risky else meta["action_type"],
        effect=effect_fact,
        undo=undo_fact,
        reversibility=recovery, decision=meta["decision"])
    if formal:
        input_ref = evidence_token("input", parsed_user.get("obs", ""))
        post_ref = evidence_token("post_diff", post_diff)
        if undo_actions:
            rev_ref = evidence_token(
                "undo", {"actions": list(undo_actions),
                         "undo_cost_steps": meta.get("undo_cost_steps")})
        else:
            rev_ref = evidence_token("probe", meta.get("evidence") or {})
        prompt += (
            "\n\nEvidence-citation contract (mandatory): append the exact "
            f"token {input_ref} to <observation>, {post_ref} to <prediction>, "
            f"and {rev_ref} to <rev_check>. These content-addressed tokens "
            "bind the prose to this sample's evidence compartments; do not "
            "invent controls not present in the supplied input.")

    reason = "teacher returned no attempts"
    for attempt in range(1, max_retries + 1):
        text = complete(prompt)
        reason = qc_check(
            text, recovery, meta["decision"], formal=formal,
            input_observation=parsed_user.get("obs", ""),
            post_signal_diff=post_diff, undo_actions=undo_actions,
            undo_cost_steps=meta.get("undo_cost_steps"),
            probe_evidence=meta.get("evidence"))
        if reason is None:
            new_asst = _splice(asst, text)
            row["messages"][-1]["content"] = new_asst
            reason = qc_full_sample(row)
            if reason is None:
                row["meta"]["prose_source"] = "teacher"
                row["meta"]["teacher_qc_status"] = "passed"
                return {"ok": True, "attempts": attempt}
            row["messages"][-1]["content"] = asst
    meta["prose_source"] = "template_fallback"
    meta["teacher_qc_status"] = "failed"
    meta["teacher_qc_reason"] = reason
    return {"ok": False, "attempts": max_retries, "last_reason": reason,
            "fallback": "template_fallback"}


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
        limit: int = 10, client: Optional[DeepSeekClient] = None,
        *, overwrite: bool = False,
        min_formal_teacher_coverage: float = .95,
        provenance_root: Path | None = None) -> int:
    """Distill up to `limit` samples (smoke-sized by default to control cost)."""
    in_path = in_path or config.FORMAL_SFT_PATH
    out_path = out_path or config.FORMAL_DISTILLED_SFT_PATH
    if not in_path.exists():
        print(f"ERROR: missing {in_path} (run `revact assemble` first)")
        return 1
    fallback_path = out_path.with_name(out_path.stem + ".template_fallback.jsonl")
    if out_path.exists() and not overwrite:
        print(f"ERROR: refusing to overwrite existing distilled asset {out_path}")
        return 1
    if fallback_path.exists() and not overwrite:
        print(f"ERROR: refusing to overwrite existing fallback asset {fallback_path}")
        return 1
    rows = [json.loads(ln) for ln in in_path.open()][:limit]
    if not rows:
        print(f"ERROR: {in_path} contains zero rows; no teacher calls made.")
        return 1
    n_formal = sum(bool((row.get("meta") or {}).get("formal_dataset"))
                   for row in rows)
    if 0 < n_formal < len(rows):
        print(f"ERROR: refusing mixed teacher input ({n_formal} formal, "
              f"{len(rows) - n_formal} legacy)")
        return 1
    if n_formal != len(rows):
        print("ERROR: teacher distillation is formal-only; legacy/template "
              "pilot rows are quarantined and no teacher calls were made.")
        return 1
    client = client or DeepSeekClient()
    teacher_model = {
        "provider": "openai-compatible",
        "name": str(getattr(client, "model", "injected-client")),
        "base_url": str(getattr(client, "base_url", "injected")),
    }
    teacher_decode = {
        "temperature": 0.7, "max_tokens": 400,
        "max_retries": config.DISTILL_MAX_RETRIES,
    }
    teacher_provenance = prompts.snapshot_generation(
        root=provenance_root or config.DATA_ROOT, author="teacher-distill",
        producer="revact.train.distill", model=teacher_model,
        decode_config=teacher_decode)
    for row in rows:
        meta = dict(row.get("meta") or {})
        meta["teacher_prompts_fp"] = teacher_provenance["prompts_fp"]
        meta["teacher_prompt_generation_fp"] = teacher_provenance[
            "prompt_generation_fp"]
        meta["teacher_model_config"] = {
            "producer": "revact.train.distill", "model": teacher_model,
            "decode": teacher_decode,
        }
        row["meta"] = meta
    ok = failed = 0
    teacher_rows = []
    fallback_rows = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        rep = distill_sample(row, client.complete)
        if rep["ok"]:
            ok += 1
            teacher_rows.append(row)
        else:
            failed += 1
            fallback_rows.append(row)
            print(f"  [drop] {row['sample_id']}: {rep['last_reason']}")
    _atomic_write_jsonl(out_path, teacher_rows, overwrite=overwrite)
    if fallback_rows:
        _atomic_write_jsonl(fallback_path, fallback_rows, overwrite=overwrite)
    coverage = ok / len(rows) if rows else 0.0
    print(f"[distill] teacher={ok} template_fallback={failed} "
          f"coverage={coverage:.1%} -> {out_path}; fallback={fallback_path}")
    formal = any(bool((row.get("meta") or {}).get("formal_dataset"))
                 for row in rows)
    if formal and coverage < min_formal_teacher_coverage:
        print("ERROR: formal teacher coverage "
              f"{coverage:.1%} < preregistered {min_formal_teacher_coverage:.1%}; "
              "fallback rows remain quarantined from the teacher set.")
        return 1
    return 0 if ok else 1


def _atomic_write_jsonl(path: Path, rows: list[dict], *, overwrite: bool) -> None:
    """Commit a generated artifact atomically; never silently truncate JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.",
                                    suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not overwrite:  # race-safe second check
            raise FileExistsError(f"refusing to overwrite {path}")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
