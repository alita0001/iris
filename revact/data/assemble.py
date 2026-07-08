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

P2 output format: the inverse world model now REASONS before it labels —
<rev_check> (mechanism-level check for an in-site undo path, constructed from
the measured undo evidence) precedes <reversibility>, and <undo> (the plan
the undo controller actually executed, summarized + cost) follows it. The
label alone was a bare classification target; the rationale ties it to
evidence visible on the page, which is what transfers to unseen action
classes (rationale-augmented distillation > label-only, Hsieh et al. 2023).

Goal texts are DIVERSIFIED: each (state, variant) deterministically draws from
a pool of >=10 explicit constraint phrasings, implicit constraint phrasings
(no "do not" token at all), and several request phrasings — so the decision
cannot be predicted from one fixed surface form. The pools live in the prompt
registry (revact/prompts.py) and are workbench-editable.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from .. import prompts
from ..grounding.base import load_reversibility_details


def __getattr__(name: str):
    """Import-compat: SYSTEM and the template pools now resolve through the
    prompt registry at access time (workbench overrides stay live)."""
    if name == "SYSTEM":
        return prompts.get("agent_system")
    if name == "EXPLICIT_CONSTRAINT_TEMPLATES":
        return prompts.get_list("explicit_constraint_templates")
    if name == "IMPLICIT_CONSTRAINT_TEMPLATES":
        return prompts.get_list("implicit_constraint_templates")
    if name == "REQUEST_TEMPLATES":
        return prompts.get_list("request_templates")
    raise AttributeError(name)

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

# What the measured undo path looks like per action type, phrased as the plan
# the agent itself could execute (used for <undo> and the <rev_check>
# rationale; the STEP COUNT comes from the grounded record, never invented).
UNDO_HINTS = {
    "add_to_cart": "remove the item from the cart",
    "wishlist_add": "remove the item from the wish list",
    "compare_add": "remove the item from the comparison list",
    "newsletter_subscribe": "restore the previous subscription setting",
    "address_add": "delete the newly created address entry",
    "address_delete": "re-create the address entry (original content is lost)",
    "reddit_vote": "retract the vote",
    "reddit_subscribe": "unsubscribe from the forum",
    "reddit_comment_submit": "delete the comment (a [deleted] tombstone remains)",
    "reddit_submission_create": "delete the submission",
    "reddit_comment_delete": "re-post the comment text (thread position is lost)",
    "reddit_edit_comment": "edit the comment again (original wording is lost)",
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
# Goal pools: constraint phrasing must not collapse to one surface form.
# The pools live in the prompt registry (workbench-editable); module-level
# names still resolve via __getattr__ for import compat.
# --------------------------------------------------------------------------- #
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
        tpl, idx = _pick(prompts.get_list("request_templates"), f"{state_id}|request")
        text = tpl.format(verb=meta["verb"])
        return {"goal": text, "style": "request", "template_id": f"request:{idx}"}
    implicit = int(hashlib.sha1(f"{state_id}|style".encode()).hexdigest(), 16) % 3 == 0
    pool = prompts.get_list("implicit_constraint_templates" if implicit
                            else "explicit_constraint_templates")
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
def _as_rev_info(reversibility) -> dict:
    """Accept either a bare label (legacy callers) or a grounded-details dict
    from load_reversibility_details."""
    if isinstance(reversibility, dict):
        return reversibility
    return {"label": reversibility, "undo_steps": None,
            "undo_actions": [], "residual_diff": None}


def _residual_text(info: dict) -> str:
    rd = info.get("residual_diff")
    if isinstance(rd, dict) and rd:
        return ", ".join(f"{k}={v}" for k, v in rd.items())
    return "a residual record remains"


def build_rev_fields(action_type: str, info: dict) -> tuple[str, str]:
    """(<rev_check>, <undo>) from the grounded undo evidence.

    <rev_check> reasons about the undo MECHANISM (which in-site control exists
    or is missing) BEFORE the label is emitted; <undo> states the plan the
    undo controller actually executed, with the measured step cost. Facts come
    from the probe record; the teacher may later rephrase but never alter them.
    """
    meta = ACTION_META[action_type]
    label = info["label"]
    hint = UNDO_HINTS.get(action_type)
    k = info.get("undo_steps")
    steps_txt = (f" ({k} step{'s' if k != 1 else ''})"
                 if isinstance(k, int) and k > 0 else "")
    if label == "REVERSIBLE":
        plan = hint or "undo it with the site's own controls"
        rev_check = (f"Checking for an in-site undo path: after {meta['gerund']}, "
                     f"this account can {plan}{steps_txt} and the measured undo "
                     f"left no residual difference, so the effect stays fully "
                     f"undoable within my own action space.")
        return rev_check, f"{plan}{steps_txt}"
    if label == "PARTIALLY_RECOVERABLE":
        plan = hint or "only a compensating action exists"
        rev_check = (f"Checking for an in-site undo path: the best available "
                     f"path is to {plan}, but it does not fully restore the "
                     f"prior state ({_residual_text(info)}), so the effect is "
                     f"only partly recoverable.")
        return rev_check, f"partial: {plan}{steps_txt}"
    if label == "IRREVERSIBLE":
        rev_check = (f"Checking for an in-site undo path: I looked for a "
                     f"control this account could use to reverse "
                     f"{meta['gerund']} and none exists — once committed there "
                     f"is no route back within my own action space.")
        return rev_check, "none available"
    rev_check = (f"Checking for an in-site undo path: I cannot verify any undo "
                 f"control for {meta['gerund']} from here, so I must treat "
                 f"reversibility as unverified.")
    return rev_check, "unverified"


def build_fields(state: dict, action_type: str, reversibility, goal: str,
                 violates_constraint: bool, requested: bool) -> dict:
    meta = ACTION_META[action_type]
    info = _as_rev_info(reversibility)
    reversibility = info["label"]
    risky_text = _clean_label((state.get("risky_action") or {}).get("text", meta["verb"]))
    decision, score = oracle(reversibility, violates_constraint, requested)
    rev_check, undo = build_rev_fields(action_type, info)

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
            "rev_check": rev_check, "undo": undo,
            "undo_steps": info.get("undo_steps"),
            "reversibility": reversibility, "decision": decision, "score": score,
            "answer": answer, "risky_text": risky_text}


def render_assistant(f: dict) -> str:
    return (f"<think>\n<observation> {f['observation']}\n"
            f"<reasoning> {f['reasoning']}\n"
            f"<prediction> {f['prediction']}\n"
            f"<rev_check> {f['rev_check']}\n"
            f"<reversibility> {f['reversibility']}\n"
            f"<undo> {f['undo']}\n"
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
    bad["rev_check"] = ("Checking for an undo path: this kind of action can "
                        "always be undone later, so nothing here is really at risk.")
    bad["undo"] = "undo later via the site's controls"
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
    if flipped in ("IRREVERSIBLE", "PARTIALLY_RECOVERABLE"):
        bad["rev_check"] = ("Checking for an undo path: I see no way to reverse "
                            "this here, so extra caution is required.")
        bad["undo"] = "none available"
        bad["reasoning"] = (f"Considering '{f['risky_text']}': it cannot be "
                            f"undone here, so extra caution is required.")
    else:
        bad["rev_check"] = ("Checking for an undo path: the site surely offers "
                            "a control to undo this, so there is no real risk.")
        bad["undo"] = "undo via the site's controls"
        bad["reasoning"] = (f"Considering '{f['risky_text']}': it can be undone "
                            f"here, so there is no real risk.")
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
    rev = load_reversibility_details(rev_path)  # action_type -> grounded record
    system = prompts.get("agent_system")
    prompts_fp = prompts.fingerprint()

    sft, dpo = [], []
    for state in reached:
        # bind this state's risky action to a grounded label by keyword match
        risky_text = ((state.get("risky_action") or {}).get("text", "")).lower()
        at = next((a for a, kw in ACTION_KW.items()
                   if a in rev and a in ACTION_META and kw in risky_text), None)
        if at is None:
            continue  # e.g. cart state ('Proceed to Checkout') has no grounded match
        info = rev[at]
        obs = state.get("axtree_snapshot", "")
        history, hist_src = prompts.state_history(state)
        risky_raw = (state.get("risky_action") or {}).get("raw_action", "")

        for vname, violates, requested in [("constraint", True, False),
                                           ("request", False, True)]:
            g = build_goal(at, vname, state["name"])
            f = build_fields(state, at, info, g["goal"], violates, requested)
            user = render_user(g["goal"], obs, history)
            chosen = render_assistant(f)
            sample_id = f"{state['name']}__{vname}"
            sft.append({
                "sample_id": sample_id,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": chosen},
                ],
                "meta": {"action_type": at, "site": site_of(at),
                         "reversibility": info["label"],
                         "undo_steps": info.get("undo_steps"),
                         "decision": f["decision"], "variant": vname,
                         "constraint_style": g["style"],
                         "goal_template": g["template_id"],
                         "reversibility_grounded": True,
                         "history_source": hist_src,
                         "risky_raw_action": risky_raw,
                         "format": "iris.v2", "prompts_fp": prompts_fp},
            })
            for pair_type, rejected in _dpo_pairs_for(f, state, violates, requested):
                dpo.append({
                    "pair_id": f"{sample_id}__{pair_type}",
                    "prompt": [{"role": "system", "content": system},
                               {"role": "user", "content": user}],
                    "chosen": chosen, "rejected": rejected,
                    "meta": {"action_type": at, "site": site_of(at),
                             "reversibility": info["label"],
                             "variant": vname, "pair_type": pair_type,
                             "constraint_style": g["style"],
                             "history_source": hist_src,
                             "risky_raw_action": risky_raw,
                             "format": "iris.v2", "prompts_fp": prompts_fp},
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
