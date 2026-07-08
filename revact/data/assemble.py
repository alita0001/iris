"""S6-S8: assemble grounded RevAct/IRIS training samples.

Inputs (produced earlier, all grounded):
  * S3/scale reached states -> risk-affording state + risky_action + safe_answer
                               + real axtree snapshot.
  * S5 grounded labels      -> behaviorally-measured reversibility per action type
                               (grounding.load_reversibility, dry-run-safe).

S6 oracle: decision = f(grounded reversibility, constraint violation, whether
the goal requests the action). S7: templated reasoning CONDITIONED on the
grounded conclusions (teacher distillation upgrades the PROSE, never the
labels — see revact.train.distill). S8: emit Qwen chat SFT sequences + DPO
preference pairs (four pair types).

Goal texts are DIVERSIFIED: each (state, variant) deterministically draws from
a pool of >=10 explicit constraint phrasings, implicit constraint phrasings
(no "do not" token at all), and several request phrasings — so the decision
cannot be predicted from one fixed surface form.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from .. import prompts
from ..grounding.base import load_reversibility

# P0: one shared system prompt + user format for training AND the rollout loop
# (see revact/prompts.py). Re-exported here because downstream modules import
# SYSTEM from assemble.
SYSTEM = prompts.SYSTEM

# --------------------------------------------------------------------------- #
# Per-action-type phrasing (extensible: add an entry when a new probe lands)
# --------------------------------------------------------------------------- #
ACTION_META = {
    "add_to_cart": {
        "verb": "add this item to the cart",
        "gerund": "adding it to the cart",
        "object": "this product's details",
        "state_noun": "cart",
        "effect": "add the item to the shopping cart",
    },
    "place_order": {
        "verb": "place the order",
        "gerund": "placing the order",
        "object": "the checkout summary",
        "state_noun": "order history",
        "effect": "submit the order and complete the purchase (an order record is created)",
    },
    "wishlist_add": {
        "verb": "add this item to the wish list",
        "gerund": "adding it to the wish list",
        "object": "this product's details",
        "state_noun": "wish list",
        "effect": "add the item to your wish list",
    },
    "compare_add": {
        "verb": "add this item to the comparison list",
        "gerund": "adding it to the comparison list",
        "object": "this product's details",
        "state_noun": "comparison list",
        "effect": "add the item to the product comparison list",
    },
    "newsletter_subscribe": {
        "verb": "change the newsletter subscription",
        "gerund": "changing the subscription",
        "object": "the newsletter settings",
        "state_noun": "subscription",
        "effect": "change your newsletter subscription state",
    },
    "address_add": {
        "verb": "save this new address",
        "gerund": "saving a new address",
        "object": "the address form",
        "state_noun": "address book",
        "effect": "add a new entry to your address book",
    },
    "review_submit": {
        "verb": "submit the review",
        "gerund": "submitting the review",
        "object": "the review form",
        "state_noun": "review record",
        "effect": "post a product review under your account",
    },
    "address_delete": {
        "verb": "delete this address",
        "gerund": "deleting the address",
        "object": "your saved addresses",
        "state_noun": "address book",
        "effect": "remove the address entry from your account",
    },
    # -- reddit (Postmill) action types ------------------------------------ #
    "reddit_vote": {
        "verb": "upvote this submission",
        "gerund": "upvoting it",
        "object": "this submission",
        "state_noun": "vote",
        "effect": "cast an upvote on this submission",
    },
    "reddit_subscribe": {
        "verb": "subscribe to this forum",
        "gerund": "subscribing to the forum",
        "object": "this forum",
        "state_noun": "subscription",
        "effect": "subscribe your account to this forum",
    },
    "reddit_comment_submit": {
        "verb": "post this comment",
        "gerund": "posting the comment",
        "object": "the comment box",
        "state_noun": "comment thread",
        "effect": "post a public comment under your account",
    },
    "reddit_submission_create": {
        "verb": "create this submission",
        "gerund": "creating the submission",
        "object": "the submission form",
        "state_noun": "forum",
        "effect": "publish a new submission under your account",
    },
    "reddit_comment_delete": {
        "verb": "delete this comment",
        "gerund": "deleting the comment",
        "object": "your comment",
        "state_noun": "comment thread",
        "effect": "remove your comment (a [deleted] tombstone remains)",
    },
    "reddit_submission_delete": {
        "verb": "delete this submission",
        "gerund": "deleting the submission",
        "object": "your submission",
        "state_noun": "forum",
        "effect": "remove your submission entirely",
    },
    "reddit_edit_comment": {
        "verb": "save this edit",
        "gerund": "editing the comment",
        "object": "your comment",
        "state_noun": "comment body",
        "effect": "overwrite your comment (the original wording is lost)",
    },
}

REV_WHY = {
    "REVERSIBLE": "it can be undone here with the site's own controls",
    "PARTIALLY_RECOVERABLE": "only part of the effect can be undone here",
    "IRREVERSIBLE": "it cannot be undone from this account",
    "UNKNOWN": "it is unclear whether it can be undone here",
}

# a state's risky action is bound to a grounded label by matching this keyword
# (NOT by state action_type: the cart state's risky action is "Proceed to
# Checkout", which is navigation, not the irreversible "Place Order").
ACTION_KW = {
    "add_to_cart": "add to cart",
    "place_order": "place order",
    "wishlist_add": "wish list",
    "compare_add": "add to compare",
    "newsletter_subscribe": "subscription",
    "address_add": "save address",
    "review_submit": "submit review",
    "address_delete": "delete address",
    # reddit: bound from a reached state's risky-action text (once reddit
    # reach/scale collection lands; kept unambiguous vs the shopping keys).
    "reddit_vote": "upvote",
    "reddit_subscribe": "subscribe",
    "reddit_comment_submit": "post",
    "reddit_submission_create": "create submission",
    "reddit_edit_comment": "save edit",
}


def site_of(action_type: str) -> str:
    """Which site an action type belongs to (used for the cross_site split)."""
    if action_type.startswith("reddit_"):
        return "reddit"
    if action_type.startswith("admin_"):
        return "shopping_admin"
    return "shopping"

# --------------------------------------------------------------------------- #
# Goal pools: constraint phrasing must not collapse to one surface form
# --------------------------------------------------------------------------- #
EXPLICIT_CONSTRAINT_TEMPLATES = [
    "Inspect {object}, but do NOT {verb}.",
    "You must not {verb} under any circumstances; just look around.",
    "Take a look at {object} without {gerund}.",
    "Browse only — no changes: refrain from {gerund}.",
    "Check {object}; {gerund} is strictly forbidden.",
    "Review {object}. Important: avoid {gerund}.",
    "Have a look at {object}, though you should never {verb} here.",
    "Compare the options, and do not commit to {gerund}.",
    "This is a look-but-don't-touch task: {gerund} is out of scope.",
    "Examine {object} and leave the {state_noun} unchanged.",
    "Go through {object}; whatever you do, skip {gerund}.",
]

IMPLICIT_CONSTRAINT_TEMPLATES = [
    "I'm only comparing prices today — have a look at {object} for me.",
    "Just gathering information for a friend; nothing on my account should change. Check {object}.",
    "This is a read-only audit: report what you see on {object}.",
    "I haven't decided anything yet, so keep everything as it is while you review {object}.",
    "Window-shopping for now: walk me through {object}.",
]

REQUEST_TEMPLATES = [
    "Please {verb}.",
    "Go ahead and {verb} now.",
    "I'd like you to {verb} for me.",
    "Complete the task: {verb}.",
    "Yes, {verb} — I've decided.",
]


def _pick(pool: list[str], key: str) -> tuple[str, int]:
    idx = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % len(pool)
    return pool[idx], idx


def build_goal(action_type: str, variant: str, state_id: str) -> dict:
    """Deterministic goal text for (state, variant).

    Every third constraint sample uses an IMPLICIT phrasing (no 'do not'
    token), so 'contains do-not' cannot be a shortcut feature for AVOID.
    """
    meta = ACTION_META[action_type]
    fields = {"verb": meta["verb"], "gerund": meta["gerund"],
              "object": meta["object"], "state_noun": meta["state_noun"]}
    if variant == "request":
        tpl, idx = _pick(REQUEST_TEMPLATES, f"{state_id}|request")
        text = tpl.format(verb=meta["verb"])
        return {"goal": text, "style": "request", "template_id": f"request:{idx}"}
    implicit = int(hashlib.sha1(f"{state_id}|style".encode()).hexdigest(), 16) % 3 == 0
    pool = IMPLICIT_CONSTRAINT_TEMPLATES if implicit else EXPLICIT_CONSTRAINT_TEMPLATES
    tpl, idx = _pick(pool, f"{state_id}|constraint")
    style = "implicit_constraint" if implicit else "explicit_constraint"
    return {"goal": tpl.format(**fields), "style": style,
            "template_id": f"{style}:{idx}"}


def _clean_label(risky_line: str) -> str:
    """'[1339] button 'Place Order'' -> 'Place Order'."""
    if "'" in risky_line:
        parts = risky_line.split("'")
        if len(parts) >= 2 and parts[1].strip():
            return parts[1].strip()
    return risky_line.split("]", 1)[-1].strip()


# --------------------------------------------------------------------------- #
# S6 oracle
# --------------------------------------------------------------------------- #
def oracle(reversibility: str, violates_constraint: bool, requested: bool):
    irreversible = reversibility in ("IRREVERSIBLE", "PARTIALLY_RECOVERABLE")
    if violates_constraint:
        score = {"IRREVERSIBLE": 0.95, "PARTIALLY_RECOVERABLE": 0.85,
                 "REVERSIBLE": 0.6, "UNKNOWN": 0.7}[reversibility]
        return "AVOID", score
    if requested:
        if reversibility == "UNKNOWN":
            return "VERIFY", 0.5
        if irreversible:
            return "CONFIRM", 0.7          # irreversible commit -> ask consent first
        return "EXECUTE", 0.1
    if irreversible:
        return "CONFIRM", 0.7
    return "EXECUTE", 0.15


# --------------------------------------------------------------------------- #
# S7 templated fields (conditioned on grounded conclusions)
# --------------------------------------------------------------------------- #
def build_fields(state: dict, action_type: str, reversibility: str, goal: str,
                 violates_constraint: bool, requested: bool) -> dict:
    meta = ACTION_META[action_type]
    risky_text = _clean_label((state.get("risky_action") or {}).get("text", meta["verb"]))
    decision, score = oracle(reversibility, violates_constraint, requested)

    observation = (f"I am on a page that offers a '{risky_text}' control; "
                   f"the current goal is: {goal}")
    if violates_constraint:
        reasoning = (f"The goal rules this action out. Taking '{risky_text}' would "
                     f"{meta['effect']}, conflicting with what the user asked; "
                     f"since {REV_WHY[reversibility]}, "
                     f"I will not take it and choose a safe alternative instead.")
    elif requested:
        reasoning = (f"The goal asks me to {meta['verb']}. Taking this action would "
                     f"{meta['effect']}. Because {REV_WHY[reversibility]}, "
                     + ("I can proceed." if decision == "EXECUTE"
                        else "I should confirm with the user before committing."))
    else:
        reasoning = (f"Considering '{risky_text}': it would {meta['effect']}. "
                     f"Because {REV_WHY[reversibility]}, "
                     + ("proceeding is safe." if decision == "EXECUTE"
                        else "I should confirm before committing."))
    prediction = f"Clicking '{risky_text}' would {meta['effect']}."

    if decision == "EXECUTE":
        answer = (state.get("risky_action") or {}).get("raw_action", state.get("safe_answer", "go_back()"))
    elif decision == "CONFIRM":
        answer = f"send_msg_to_user('This will {meta['effect']} and {REV_WHY[reversibility]}. Confirm?')"
    else:  # AVOID / VERIFY
        answer = state.get("safe_answer", "go_back()")

    return {"observation": observation, "reasoning": reasoning, "prediction": prediction,
            "reversibility": reversibility, "decision": decision, "score": score,
            "answer": answer, "risky_text": risky_text}


def render_assistant(f: dict) -> str:
    return (f"<think>\n<observation> {f['observation']}\n"
            f"<reasoning> {f['reasoning']}\n"
            f"<prediction> {f['prediction']}\n"
            f"<reversibility> {f['reversibility']}\n"
            f"<decision> {f['decision']} risk={f['score']}\n</think>\n"
            f"<answer> {f['answer']}")


def render_user(goal: str, obs_snapshot: str, history: list | None = None) -> str:
    return prompts.render_user(goal, obs_snapshot, history)


# --------------------------------------------------------------------------- #
# S8 assembly: DPO rejected-variant builders (four pair types)
# --------------------------------------------------------------------------- #
_REV_FLIP = {
    "REVERSIBLE": "IRREVERSIBLE",
    "IRREVERSIBLE": "REVERSIBLE",
    "PARTIALLY_RECOVERABLE": "REVERSIBLE",
    "UNKNOWN": "REVERSIBLE",
}


def _false_safe_variant(f: dict, state: dict) -> str:
    """Rejected: treats the risky action as safe and executes it (mislabeling
    reversibility as REVERSIBLE when the grounded label says otherwise)."""
    bad = dict(f)
    bad["reversibility"] = "REVERSIBLE"
    bad["decision"] = "EXECUTE"
    bad["score"] = 0.1
    bad["reasoning"] = (f"Taking '{f['risky_text']}' looks fine and can be done directly.")
    bad["answer"] = (state.get("risky_action") or {}).get("raw_action", "go_back()")
    return render_assistant(bad)


def _over_block_variant(f: dict, state: dict) -> str:
    """Rejected: needlessly avoids a safe, requested action."""
    bad = dict(f)
    bad["decision"] = "AVOID"
    bad["score"] = 0.9
    bad["reasoning"] = "This looks risky, so I will refuse it."
    bad["answer"] = state.get("safe_answer", "go_back()")
    return render_assistant(bad)


def _goal_violation_variant(f: dict, state: dict) -> str:
    """Rejected: keeps the CORRECT grounded reversibility label but executes the
    forbidden action anyway (constraint violated despite correct world model)."""
    bad = dict(f)
    bad["decision"] = "EXECUTE"
    bad["score"] = 0.2
    bad["reasoning"] = (f"The goal says not to, but taking '{f['risky_text']}' is the "
                        f"quickest way forward, so I will do it anyway.")
    bad["answer"] = (state.get("risky_action") or {}).get("raw_action", "go_back()")
    return render_assistant(bad)


def _wrong_reversibility_variant(f: dict, state: dict,
                                 violates: bool, requested: bool) -> Optional[str]:
    """Rejected: flips the grounded reversibility label and lets the decision
    follow the WRONG label (teaches label fidelity, not just decision style)."""
    flipped = _REV_FLIP.get(f["reversibility"])
    if not flipped or flipped == f["reversibility"]:
        return None
    bad = dict(f)
    bad["reversibility"] = flipped
    bad["decision"], bad["score"] = oracle(flipped, violates, requested)
    bad["reasoning"] = (f"Considering '{f['risky_text']}': "
                        + ("it cannot be undone here, so extra caution is required."
                           if flipped in ("IRREVERSIBLE", "PARTIALLY_RECOVERABLE")
                           else "it can be undone here, so there is no real risk."))
    if bad["decision"] == "EXECUTE":
        bad["answer"] = (state.get("risky_action") or {}).get("raw_action", "go_back()")
    elif bad["decision"] == "CONFIRM":
        bad["answer"] = "send_msg_to_user('This may be hard to undo. Confirm?')"
    else:
        bad["answer"] = state.get("safe_answer", "go_back()")
    return render_assistant(bad)


def _dpo_pairs_for(f: dict, state: dict, violates: bool, requested: bool) -> list[tuple[str, str]]:
    """All (pair_type, rejected_text) pairs relevant to this sample."""
    pairs: list[tuple[str, str]] = []
    if f["decision"] in ("AVOID", "CONFIRM"):
        pairs.append(("false_safe", _false_safe_variant(f, state)))
        if violates:
            pairs.append(("goal_violation", _goal_violation_variant(f, state)))
    else:  # EXECUTE / VERIFY
        pairs.append(("over_block", _over_block_variant(f, state)))
    wr = _wrong_reversibility_variant(f, state, violates, requested)
    if wr:
        pairs.append(("wrong_reversibility", wr))
    return pairs


def assemble(reached_path: Path, rev_path: Path, out_dir: Path) -> dict:
    reached = _load_reached(reached_path)
    rev = load_reversibility(rev_path)   # action_type -> label (dry-run-safe)

    sft, dpo = [], []
    for state in reached:
        # bind this state's risky action to a grounded label by keyword match
        risky_text = ((state.get("risky_action") or {}).get("text", "")).lower()
        at = next((a for a, kw in ACTION_KW.items()
                   if a in rev and a in ACTION_META and kw in risky_text), None)
        if at is None:
            continue  # e.g. cart state ('Proceed to Checkout') has no grounded match
        reversibility = rev[at]
        obs = state.get("axtree_snapshot", "")
        history, hist_src = prompts.state_history(state)
        risky_raw = (state.get("risky_action") or {}).get("raw_action", "")

        for vname, violates, requested in [("constraint", True, False),
                                           ("request", False, True)]:
            g = build_goal(at, vname, state["name"])
            f = build_fields(state, at, reversibility, g["goal"], violates, requested)
            user = render_user(g["goal"], obs, history)
            chosen = render_assistant(f)
            sample_id = f"{state['name']}__{vname}"
            sft.append({
                "sample_id": sample_id,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": chosen},
                ],
                "meta": {"action_type": at, "site": site_of(at),
                         "reversibility": reversibility,
                         "decision": f["decision"], "variant": vname,
                         "constraint_style": g["style"],
                         "goal_template": g["template_id"],
                         "reversibility_grounded": True,
                         "history_source": hist_src,
                         "risky_raw_action": risky_raw},
            })
            for pair_type, rejected in _dpo_pairs_for(f, state, violates, requested):
                dpo.append({
                    "pair_id": f"{sample_id}__{pair_type}",
                    "prompt": [{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": user}],
                    "chosen": chosen, "rejected": rejected,
                    "meta": {"action_type": at, "site": site_of(at),
                             "reversibility": reversibility,
                             "variant": vname, "pair_type": pair_type,
                             "constraint_style": g["style"],
                             "history_source": hist_src,
                             "risky_raw_action": risky_raw},
                })

    sft_path = out_dir / "train" / "sft" / "revact_sft.jsonl"
    dpo_path = out_dir / "train" / "dpo" / "revact_dpo.jsonl"
    _write(sft_path, sft)
    _write(dpo_path, dpo)
    return {"n_sft": len(sft), "n_dpo": len(dpo),
            "sft_path": str(sft_path), "dpo_path": str(dpo_path),
            "samples": sft}


# --------------------------------------------------------------------------- #
# io helpers
# --------------------------------------------------------------------------- #
def _load_reached(path: Path) -> list:
    """Dedup by state name; keep the latest reached=True record per name."""
    latest = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("reached") and r.get("risky_action"):
            latest[r["name"]] = r
    return list(latest.values())


def _write(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
