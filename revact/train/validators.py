"""Fail-closed validators shared by SFT, preference training and offline RLVR.

The helpers in this module intentionally do not import torch/transformers.  In
particular, BrowserGym actions are parsed as a tiny AST rather than matched by
raw substring: prose that merely *mentions* ``click('42')`` must never count as
an attempted action.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


IRIS_TAG_ORDER = (
    "<think>", "<observation>", "<reasoning>", "<prediction>",
    "<rev_check>", "<reversibility>", "<undo>", "<decision>",
    "</think>", "<answer>",
)
SAFE_DECISIONS = frozenset({"AVOID", "CONFIRM", "VERIFY"})
DECISIONS = frozenset({"EXECUTE", *SAFE_DECISIONS})


@dataclass(frozen=True)
class ParsedAction:
    """A deliberately small, serialisable view of one BrowserGym action."""

    name: str
    args: tuple[Any, ...] = ()
    bid: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "args": list(self.args), "bid": self.bid}


def answer_text(completion: Any) -> str:
    """Return only the first action line in ``<answer>``, never think prose."""
    if not isinstance(completion, str):
        completion = " ".join(
            m.get("content", "") for m in (completion or [])
            if isinstance(m, dict)
        )
    match = re.search(r"<answer>\s*(.+)", completion or "", re.DOTALL)
    return match.group(1).strip().splitlines()[0].strip() if match else ""


def parse_action(action: Any) -> ParsedAction | None:
    """Parse a single literal BrowserGym call without executing any code.

    Only a direct function call whose positional/keyword values are literals is
    accepted.  Chained calls, attributes, comprehensions and interpolated code
    are rejected.  The first argument is the bid for DOM-targeted primitives.
    """
    if isinstance(action, dict):
        raw = action.get("raw_action") or action.get("browsergym_action")
        if raw:
            return parse_action(raw)
        name = str(action.get("name") or action.get("action") or "").strip()
        args = tuple(action.get("args") or ())
        bid = action.get("bid")
        if bid is None and name in _BID_ACTIONS and args:
            bid = args[0]
        return ParsedAction(name=name, args=args,
                            bid=None if bid is None else str(bid)) if name else None
    if not isinstance(action, str) or not action.strip():
        return None
    source = answer_text(action) if "<answer>" in action else action.strip().splitlines()[0]
    try:
        node = ast.parse(source, mode="eval").body
    except (SyntaxError, ValueError):
        return None
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return None
    try:
        args = tuple(ast.literal_eval(a) for a in node.args)
        # Parse keywords to reject executable expressions even though the
        # current comparison key only needs positional arguments.
        for kw in node.keywords:
            if kw.arg is None:
                return None
            ast.literal_eval(kw.value)
    except (ValueError, TypeError, SyntaxError):
        return None
    name = node.func.id
    bid = str(args[0]) if name in _BID_ACTIONS and args else None
    return ParsedAction(name=name, args=args, bid=bid)


_BID_ACTIONS = frozenset({
    "click", "dblclick", "hover", "type", "fill", "select_option",
    "check", "uncheck", "press", "focus",
})


def actions_match(actual: ParsedAction | None,
                  target: ParsedAction | dict | str | None) -> bool:
    """Match an executed action to the structured risky action identity."""
    wanted = target if isinstance(target, ParsedAction) else parse_action(target)
    if actual is None or wanted is None or actual.name != wanted.name:
        return False
    if wanted.bid is not None:
        return actual.bid == wanted.bid
    # Non-DOM calls (e.g. go_back) are identified by primitive + literal args.
    return actual.args == wanted.args


def parse_decision(text: str) -> str:
    match = re.search(r"<decision>\s*([A-Z_]+)", text or "")
    return match.group(1) if match else ""


def iris_tag_errors(text: str, *, require_complete: bool = True) -> list[str]:
    """Check exact cardinality and ordering of the machine-readable protocol."""
    required = IRIS_TAG_ORDER if require_complete else IRIS_TAG_ORDER[1:5]
    errors: list[str] = []
    positions: list[int] = []
    for tag in required:
        count = (text or "").count(tag)
        if count != 1:
            errors.append(f"tag {tag} occurs {count} times (expected exactly 1)")
        positions.append((text or "").find(tag))
    if all(pos >= 0 for pos in positions) and positions != sorted(positions):
        errors.append("tags are out of order")
    return errors


def decision_answer_consistent(decision: str, answer: str,
                               risky_action: dict | str | None = None) -> bool:
    """Mechanical consistency check between a decision and its actual action.

    It does not decide whether the policy decision is normatively correct; it
    only prevents a completion from declaring AVOID/CONFIRM/VERIFY and then
    emitting the exact risky action in ``<answer>``.
    """
    decision = (decision or "").strip().upper()
    actual = parse_action(answer)
    risky_attempt = actions_match(actual, risky_action) if risky_action else False
    if decision not in DECISIONS or actual is None:
        return False
    if decision in SAFE_DECISIONS:
        if risky_attempt:
            return False
        if decision == "CONFIRM":
            return actual.name == "send_msg_to_user"
        return True
    # In a point-level decision example EXECUTE must select the pinned action.
    return actions_match(actual, risky_action) if risky_action else True


def _tag_field(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*([^\n]+)", text or "")
    return match.group(1).strip() if match else ""


def _evidence_ref(kind: str, payload: Any) -> str:
    encoded = (payload.encode("utf-8") if isinstance(payload, str) else
               json.dumps(payload, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8"))
    return f"[evidence:{kind}:{hashlib.sha256(encoded).hexdigest()[:12]}]"


def _unsupported_control_claim(text: str, input_observation: str) -> str:
    """Return a positively asserted UI control absent from the policy input."""
    patterns = (
        r"\b(?:a|an|the)\s+['\"]?([a-z][a-z0-9 _-]{0,40}?)['\"]?\s+"
        r"(?:button|link|control)\s+(?:exists|appears|is visible|is available)",
        r"\b(?:shows?|exposes?|provides?|contains?)\s+(?:a|an|the)?\s*"
        r"['\"]?([a-z][a-z0-9 _-]{0,40}?)['\"]?\s+(?:button|link|control)",
    )
    lower = (text or "").lower()
    observed = (input_observation or "").lower()
    for pattern in patterns:
        for match in re.finditer(pattern, lower):
            prefix = lower[max(0, match.start() - 16):match.start()]
            if re.search(r"\b(?:no|not|without|lacks?)\s*$", prefix):
                continue
            label = re.sub(r"\s+", " ", match.group(1)).strip(" '\"")
            if label and label not in observed:
                return label
    return ""


def formal_completion_reasons(row: dict) -> list[str]:
    """Validate the gold/chosen completion against pinned point/truth metadata.

    This check is intentionally independent of a model or tokenizer and is
    shared by SFT, DPO, RLVR source validation and formal export.  A structurally
    valid completion cannot silently disagree with its point-level label.
    """
    meta = row.get("meta") if isinstance(row, dict) else {}
    meta = meta if isinstance(meta, dict) else {}
    if meta.get("formal_dataset") is not True:
        return []
    messages = row.get("messages") or []
    if messages:
        roles = [message.get("role") if isinstance(message, dict) else None
                 for message in messages]
        if roles != ["system", "user", "assistant"]:
            return ["completion_message_topology_mismatch"]
        completion = str((messages[-1] or {}).get("content") or "")
        prompt = messages[:-1]
    else:
        completion = str(row.get("chosen") or "")
        prompt = row.get("prompt") or []
        roles = [message.get("role") if isinstance(message, dict) else None
                 for message in prompt]
        if roles != ["system", "user"]:
            return ["completion_message_topology_mismatch"]
    reasons = ["completion_" + error for error in iris_tag_errors(completion)]

    expected_recovery = str(meta.get("recovery_status") or "")
    actual_recovery = _tag_field(completion, "reversibility").split()
    if not actual_recovery or actual_recovery[0] != expected_recovery:
        reasons.append("completion_recovery_status_mismatch")
    expected_decision = str(meta.get("decision") or "")
    if parse_decision(completion) != expected_decision:
        reasons.append("completion_decision_mismatch")
    risky = meta.get("risky_action") or meta.get("risky_raw_action")
    if not decision_answer_consistent(
            expected_decision, answer_text(completion), risky):
        reasons.append("completion_decision_answer_inconsistent")

    user = next((str(message.get("content") or "")
                 for message in reversed(prompt)
                 if isinstance(message, dict) and message.get("role") == "user"), "")
    from .. import prompts
    from ..envs.obs_utils import bid_is_visible
    input_observation = prompts.parse_observation_message(user)
    observed_hash = hashlib.sha256(input_observation.encode("utf-8")).hexdigest()
    if str(meta.get("policy_input_observation_hash") or "") != observed_hash:
        reasons.append("completion_policy_input_observation_hash_mismatch")
    risky_parsed = parse_action(risky)
    if risky_parsed and risky_parsed.bid and \
            not bid_is_visible(input_observation, risky_parsed.bid):
        reasons.append("completion_risky_bid_absent_from_input")

    rev_check = _tag_field(completion, "rev_check")
    undo_payload = {"actions": list(meta.get("undo_actions") or []),
                    "undo_cost_steps": meta.get("undo_cost_steps")}
    probe_id = str(meta.get("probe_point_id") or "")
    evidence = meta.get("evidence") or {}
    if probe_id not in rev_check and \
            _evidence_ref("undo", undo_payload) not in rev_check and \
            _evidence_ref("probe", evidence) not in rev_check:
        reasons.append("completion_rev_check_not_tied_to_probe_evidence")
    for cited_bid in re.findall(r"\[([^\]\s]+)\]", rev_check):
        if cited_bid.startswith("evidence:"):
            continue
        if not bid_is_visible(input_observation, cited_bid):
            reasons.append("completion_rev_check_cites_absent_bid")
            break
    visible_claim = str(evidence.get("pre_visible_undo_evidence") or "").strip()
    if visible_claim:
        if visible_claim not in rev_check or visible_claim.lower() not in \
                input_observation.lower():
            reasons.append("completion_rev_check_visible_evidence_mismatch")
    elif re.search(r"current pre-action input shows", rev_check, re.I):
        reasons.append("completion_rev_check_unpinned_visible_claim")
    if _unsupported_control_claim(rev_check, input_observation):
        reasons.append("completion_rev_check_claims_absent_control")

    post_diff = (meta.get("post_signal_diff") or
                 (meta.get("evidence") or {}).get("post_signal_diff"))
    prediction = _tag_field(completion, "prediction")
    if post_diff in (None, "", {}, []):
        reasons.append("completion_missing_post_signal_diff")
    else:
        pre = post_diff.get("pre_signal") if isinstance(post_diff, dict) else None
        post = post_diff.get("post_signal") if isinstance(post_diff, dict) else None
        exact_values = (
            pre is not None and post is not None and
            json.dumps(pre, ensure_ascii=False, sort_keys=True) in prediction and
            json.dumps(post, ensure_ascii=False, sort_keys=True) in prediction)
        if not exact_values and _evidence_ref("post_diff", post_diff) not in prediction:
            reasons.append("completion_prediction_not_tied_to_post_diff")

    undo_actions = list(meta.get("undo_actions") or [])
    undo_text = _tag_field(completion, "undo")
    if expected_recovery in {"RECOVERED", "PARTIALLY_RECOVERED"}:
        if not undo_actions:
            reasons.append("completion_positive_recovery_without_undo_actions")
        for action in undo_actions:
            if str(action).strip() and str(action).strip() not in undo_text:
                reasons.append("completion_undo_action_mismatch")
                break
    undo_cost = meta.get("undo_cost_steps")
    if undo_cost is not None and not re.search(
            rf"(?:undo_)?cost_steps\s*=\s*{int(undo_cost)}\b", undo_text):
        reasons.append("completion_undo_cost_mismatch")
    return reasons


def formal_negative_candidate_reasons(
        row: dict, candidates: dict[str, Any] | None = None) -> list[str]:
    """Validate a deployment-shaped DPO error against its exact S4 candidate."""
    from .. import prompts as prompt_registry
    from ..envs.obs_utils import bid_is_visible
    meta = row.get("meta") if isinstance(row, dict) else {}
    meta = meta if isinstance(meta, dict) else {}
    if meta.get("formal_dataset") is not True or \
            meta.get("negative_source") not in {"legal_candidate", "on_policy"}:
        return []
    reasons: list[str] = []
    source = str(meta.get("negative_source") or "")
    candidate_id = str(meta.get("negative_candidate_id") or "")
    snapshot_hash = str(meta.get("negative_candidate_snapshot_hash") or "")
    if source == "legal_candidate":
        if not candidate_id:
            reasons.append("negative_candidate_id_missing")
        if not snapshot_hash:
            reasons.append("negative_candidate_snapshot_hash_missing")
        if meta.get("legal_at_snapshot") is not True:
            reasons.append("negative_candidate_not_legal_at_snapshot")
    else:
        required_trace = {
            "on_policy_trace_id", "on_policy_input_messages_sha256",
            "on_policy_raw_completion_sha256", "on_policy_error_types",
            "on_policy_action_kind", "on_policy_action_legal",
            "on_policy_rollout_run_id",
        }
        missing = sorted(name for name in required_trace
                         if meta.get(name) in (None, "", []))
        if meta.get("on_policy_trace_verified") is not True:
            reasons.append("on_policy_trace_not_verified")
        if missing:
            reasons.append("on_policy_trace_fields_missing:" + ",".join(missing))
        action_legal = meta.get("on_policy_action_legal")
        if not isinstance(action_legal, bool):
            reasons.append("on_policy_action_legal_not_boolean")
        if action_legal is False and (candidate_id or snapshot_hash):
            reasons.append("illegal_on_policy_action_claims_candidate")
        rejected_action = parse_action(answer_text(row.get("rejected") or ""))
        if (action_legal is True and rejected_action is not None and
                rejected_action.bid and not candidate_id):
            reasons.append("legal_dom_on_policy_action_lacks_candidate")
    candidate = candidates.get(candidate_id) if candidates is not None else None
    if candidates is not None and candidate_id:
        if candidate is None:
            reasons.append("unknown_negative_candidate_id")
        else:
            mismatched = []
            expected = {
                "state_id": str(meta.get("state_id") or ""),
                "snapshot_hash": snapshot_hash,
                "legal_at_snapshot": True,
            }
            for key, value in expected.items():
                if getattr(candidate, key) != value:
                    mismatched.append(key)
            rejected_action = parse_action(answer_text(row.get("rejected") or ""))
            primitive = candidate.canonical_action.split(":", 1)[0]
            if (rejected_action is None or rejected_action.bid != candidate.bid or
                    rejected_action.name != primitive):
                mismatched.append("rejected_bid")
            if mismatched:
                reasons.append(
                    "negative_candidate_mismatch:" + ",".join(mismatched))
    prompt = row.get("prompt") or []
    user = next((str(message.get("content") or "")
                 for message in reversed(prompt)
                 if isinstance(message, dict) and message.get("role") == "user"), "")
    rejected = parse_action(answer_text(row.get("rejected") or ""))
    should_be_visible = (source == "legal_candidate" or
                         meta.get("on_policy_action_legal") is True)
    if should_be_visible and rejected and rejected.bid and not bid_is_visible(
            prompt_registry.parse_observation_message(user), rejected.bid):
        reasons.append("negative_candidate_bid_absent_from_input")
    if source == "on_policy" and not (
            meta.get("policy_model_version") or
            meta.get("proposer_model_version")):
        reasons.append("on_policy_model_version_missing")
    return reasons


def percentile(values: Iterable[int], q: float) -> float:
    """Linear percentile matching NumPy's default for small audit reports."""
    vals = sorted(int(v) for v in values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return float(vals[0])
    rank = (len(vals) - 1) * q
    lo, hi = int(rank), min(int(rank) + 1, len(vals) - 1)
    frac = rank - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def formal_point_join_problems(rows: Iterable[dict], data_root: Path) -> list[str]:
    """Verify every formal trainer row against the canonical point+manifest.

    Syntax-only validators are useful for fixtures, but the production training
    entry points call this gate so a decorative ``probe_point_id`` cannot bypass
    the exact body/manifest and immutable prompt-bundle join.
    """
    from ..data.governance import (formal_prompt_content_reasons,
                                   formal_release_context,
                                   formal_release_reasons)

    context = formal_release_context(Path(data_root))
    problems: list[str] = []
    for index, row in enumerate(rows):
        meta = row.get("meta") or {}
        if meta.get("formal_dataset") is not True:
            continue
        ident = str(row.get("sample_id") or row.get("pair_id") or f"row{index}")
        reasons = formal_release_reasons(meta, context)
        reasons.extend(formal_prompt_content_reasons(row, context))
        for reason in reasons:
            problems.append(f"{ident}: formal provenance {reason}")
    return problems
