"""S6-S8: assemble grounded RevAct/IRIS training samples.

Formal inputs:
  * S3/scale reached states -> point-addressed state + action instance + real
                               accessibility snapshot and trajectory history;
  * S5 grounding            -> one canonical ``GroundingPoint`` for that exact
                               state/action transition;
  * evaluation truth        -> independent point×goal policy truth.

The old action-class loader remains only behind ``formal=False`` for frozen
pilot inspection and never produces a formal artifact.

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

from .. import config, prompts
from ..eval.truth import (assert_truth_manifest_integrity,
                          load_truth_records, truth_by_point_variant)
from ..grounding.base import load_reversibility_details
from ..grounding.schema import (GroundingPoint, GroundingValidationError,
                                load_probe_points)
from ..grounding.transitions import (ProbeTransition,
                                     assert_point_transition_integrity,
                                     load_probe_transitions,
                                     transition_manifest_row)
from ..train.validators import parse_action


class FormalAssemblyError(ValueError):
    """A row cannot be admitted to the formal point-grounded dataset."""


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
    "RECOVERED": "the recorded solver restored the measured state within budget",
    "PARTIALLY_RECOVERABLE": "only part of the effect can be undone here",
    "PARTIALLY_RECOVERED": "the recorded solver restored only part of the measured state",
    "IRREVERSIBLE": "the legacy record claims no undo path",
    "NOT_RECOVERED_WITHIN_BUDGET": (
        "the recorded solver set did not recover the measured state within budget"),
    "NO_EFFECT": "the measured action produced no effect",
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
def oracle(reversibility: str, violates_constraint: bool, requested: bool,
           normative_risk: Optional[bool] = None):
    """Decision oracle with an explicit normative-risk axis.

    ``normative_risk=None`` preserves legacy fixture behavior only.  Formal
    assembly requires a boolean policy truth and never derives safety from
    recoverability alone.
    """
    difficult_recovery = reversibility in (
        "IRREVERSIBLE", "PARTIALLY_RECOVERABLE", "PARTIALLY_RECOVERED",
        "NOT_RECOVERED_WITHIN_BUDGET")
    if violates_constraint:
        score = 0.95 if normative_risk is True else 0.8
        return "AVOID", score
    if normative_risk is True:
        return "CONFIRM", 0.9
    if normative_risk is False:
        return "EXECUTE", 0.1
    # Explicitly legacy-only compatibility below.
    if requested:
        if reversibility == "UNKNOWN":
            return "VERIFY", 0.5
        if difficult_recovery:
            return "CONFIRM", 0.7
        return "EXECUTE", 0.1
    if difficult_recovery:
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
    if info.get("formal_point"):
        point_id = info["probe_point_id"]
        k_budget = info.get("budget_k")
        solvers = ", ".join(info.get("solver_set") or []) or "unrecorded solver"
        raw_undo = list(info.get("undo_actions") or [])
        semantic = list(info.get("undo_semantic_actions") or [])
        k = info.get("undo_steps")
        visible = (info.get("evidence") or {}).get("pre_visible_undo_evidence")
        visible_text = (
            f"The current pre-action input shows: {visible}."
            if visible else
            "The current pre-action input does not itself establish a post-action undo control."
        )
        if semantic:
            plan = " -> ".join(semantic)
        elif raw_undo:
            plan = " -> ".join(raw_undo)
        else:
            plan = "no successful undo sequence was recorded"
        if label == "RECOVERED":
            rev_check = (
                f"{visible_text} Post-action probe {point_id} executed the recorded "
                f"undo sequence with {solvers}; the final measured signal matched "
                f"the pre-action signal within budget k={k_budget}. This is "
                f"constructive, probe-relative recovery evidence, not a safety claim.")
            raw_plan = " -> ".join(raw_undo)
            return rev_check, (
                f"semantic={plan}; browsergym_actions={raw_plan}; "
                f"undo_cost_steps={k}")
        if label == "PARTIALLY_RECOVERED":
            rev_check = (
                f"{visible_text} Post-action probe {point_id} tried {plan} with "
                f"{solvers}, but residual measured state remained "
                f"({_residual_text(info)}). Recovery is partial under this signal.")
            raw_plan = " -> ".join(raw_undo)
            return rev_check, (
                f"partial semantic={plan}; browsergym_actions={raw_plan}; "
                f"undo_cost_steps={k}")
        if label == "NOT_RECOVERED_WITHIN_BUDGET":
            rev_check = (
                f"{visible_text} Probe {point_id} used solver set [{solvers}] up to "
                f"budget k={k_budget} and did not restore the measured signal. This "
                f"is a bounded search failure; it does not prove that no undo path exists.")
            return rev_check, "not found by the recorded solver set within budget"
        if info.get("effect_status") == "NO_EFFECT":
            return (
                f"Probe {point_id} measured no state change after the action; recovery "
                f"is therefore not applicable, rather than a reversibility class.",
                "not applicable (no measured effect)",
            )
        return (
            f"{visible_text} Probe {point_id} did not establish a recovery result for "
            f"the measured transition, so recovery remains unknown.",
            "unverified",
        )

    # Legacy compatibility only.  These hints are never formal gold.
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
    if label in ("IRREVERSIBLE", "NOT_RECOVERED_WITHIN_BUDGET"):
        rev_check = (f"Checking for an in-site undo path: I looked for a "
                     f"control this account could use to reverse {meta['gerund']} "
                     f"within the recorded budget and did not find one; this is "
                     f"bounded failed recovery, not proof of impossibility.")
        return rev_check, "not found within the recorded budget"
    rev_check = (f"Checking for an in-site undo path: I cannot verify any undo "
                 f"control for {meta['gerund']} from here, so I must treat "
                 f"reversibility as unverified.")
    return rev_check, "unverified"


def _transition_prediction(info: dict) -> str:
    """Render only facts measured in the immediate post-action observation."""
    transition = info.get("transition")
    if not isinstance(transition, ProbeTransition):
        pre = json.dumps(info.get("pre_signal"), ensure_ascii=False, sort_keys=True)
        post = json.dumps(info.get("post_signal"), ensure_ascii=False, sort_keys=True)
        effect = info.get("effect_status")
        measured = (
            f"left the measured signal unchanged at {pre} (post={post})"
            if effect == "NO_EFFECT" else
            f"changed the measured signal from {pre} to {post}"
            if effect == "CHANGED" else
            f"recorded measured signal {pre} -> {post}; effect status is UNKNOWN, "
            "so no directional next-state change is asserted")
        return (
            f"Probe point {info['probe_point_id']} {measured}, "
            "but its exact post-action observation body was not retained; this is "
            "hash-only effect evidence, not a next-state target.")

    from .candidates import interactive_elements
    before = {
        (row["role"], row["name"]) for row in interactive_elements(
            transition.pre_observation.policy_axtree)
    }
    after = {
        (row["role"], row["name"]) for row in interactive_elements(
            transition.post_observation.policy_axtree)
    }
    added = [f"{role}:{name}" for role, name in sorted(after - before)][:5]
    removed = [f"{role}:{name}" for role, name in sorted(before - after)][:5]
    facts = [
        f"immediate post URL={transition.post_observation.url}",
        "measured signal " + json.dumps(
            transition.pre_signal, ensure_ascii=False, sort_keys=True) +
        " -> " + json.dumps(
            transition.post_signal, ensure_ascii=False, sort_keys=True),
    ]
    if added:
        facts.append("interactive controls added=" + json.dumps(added, ensure_ascii=False))
    if removed:
        facts.append("interactive controls removed=" + json.dumps(
            removed, ensure_ascii=False))
    return (
        f"Probe transition {transition.transition_id} actually executed the pinned "
        f"action; " + "; ".join(facts) + ".")


def build_fields(state: dict, action_type: str, reversibility, goal: str,
                 violates_constraint: bool, requested: bool,
                 normative_risk: Optional[bool] = None) -> dict:
    meta = ACTION_META[action_type]
    info = _as_rev_info(reversibility)
    reversibility = info["label"]
    risky_text = _clean_label((state.get("risky_action") or {}).get("text", meta["verb"]))
    decision, score = oracle(reversibility, violates_constraint, requested,
                             normative_risk=normative_risk)
    rev_check, undo = build_rev_fields(action_type, info)

    observation = (f"I am on a page that offers a '{risky_text}' control; "
                   f"the current goal is: {goal}")
    if info.get("formal_point") and violates_constraint:
        reasoning = (
            f"The action violates the explicit policy constraint, so I must avoid "
            f"it. Its independently measured recovery status is {reversibility}; "
            f"recoverability does not determine the normative decision.")
    elif info.get("formal_point") and normative_risk is True:
        reasoning = (
            f"The independent policy truth marks this action as risky, so I must "
            f"obtain confirmation before attempting it. The probe-relative recovery "
            f"status is {reversibility}, which is reported separately.")
    elif info.get("formal_point") and normative_risk is False:
        reasoning = (
            f"The requested action is not marked risky by the independent policy "
            f"truth, so I can execute it. Its recovery status {reversibility} is a "
            f"separate operational property, not the safety label.")
    elif violates_constraint:
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
    if info.get("formal_point"):
        prediction = _transition_prediction(info)
        prediction_source = (
            "probe_transition" if info.get("transition_body_verified")
            else "probe_signal_diff_hash_only")
    else:
        prediction = f"Clicking '{risky_text}' would {meta['effect']}."
        prediction_source = "action_meta_template_legacy"

    if decision == "EXECUTE":
        answer = (state.get("risky_action") or {}).get("raw_action", state.get("safe_answer", "go_back()"))
    elif decision == "CONFIRM":
        if info.get("formal_point"):
            answer = ("send_msg_to_user('Policy marks this action as risky; "
                      "confirm before I attempt it?')")
        else:
            answer = f"send_msg_to_user('This will {meta['effect']} and {REV_WHY[reversibility]}. Confirm?')"
    else:  # AVOID / VERIFY
        answer = state.get("safe_answer", "go_back()")

    return {"observation": observation, "reasoning": reasoning, "prediction": prediction,
            "rev_check": rev_check, "undo": undo,
            "undo_steps": info.get("undo_steps"),
            "reversibility": reversibility, "decision": decision, "score": score,
            "answer": answer, "risky_text": risky_text,
            "normative_risk": normative_risk,
            "prediction_source": prediction_source,
            "undo_source": ("probe_point_id" if info.get("formal_point")
                            else "legacy_hint"),
            "undo_source_probe_point_id": (info.get("probe_point_id")
                                             if info.get("formal_point") else "")}


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
    "RECOVERED": "NOT_RECOVERED_WITHIN_BUDGET",
    "IRREVERSIBLE": "REVERSIBLE",
    "NOT_RECOVERED_WITHIN_BUDGET": "RECOVERED",
    "PARTIALLY_RECOVERABLE": "REVERSIBLE",
    "PARTIALLY_RECOVERED": "RECOVERED",
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
    bad["decision"], bad["score"] = oracle(
        flipped, violates, requested, normative_risk=f.get("normative_risk"))
    if flipped in ("IRREVERSIBLE", "PARTIALLY_RECOVERABLE",
                   "PARTIALLY_RECOVERED", "NOT_RECOVERED_WITHIN_BUDGET"):
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


def _point_info(point: GroundingPoint,
                transition: ProbeTransition | None = None) -> dict:
    return {
        "formal_point": True,
        "label": point.recovery_status,
        "effect_status": point.effect_status,
        "undo_steps": point.undo_cost_steps,
        "undo_actions": list(point.undo_actions),
        "undo_semantic_actions": list(point.undo_semantic_actions),
        "residual_diff": point.residual_diff,
        "probe_point_id": point.probe_point_id,
        "budget_k": point.budget_k,
        "solver_set": list(point.solver_set),
        "pre_signal": point.pre_signal,
        "post_signal": point.post_signal,
        "evidence": point.evidence,
        "transition": transition,
        "transition_body_verified": transition is not None,
    }


def _state_normative_risk(state: dict) -> Optional[bool]:
    value = state.get("normative_risk")
    if value is None and isinstance(state.get("policy_constraint_truth"), dict):
        value = state["policy_constraint_truth"].get("normative_risk")
    return value if isinstance(value, bool) else None


def _resolve_formal_point(state: dict,
                          points: dict[str, GroundingPoint],
                          transition: ProbeTransition | None = None
                          ) -> GroundingPoint:
    risky = state.get("risky_action") or {}
    point_id = state.get("probe_point_id") or risky.get("probe_point_id")
    if not point_id:
        raise FormalAssemblyError("missing unique probe_point_id")
    point = points.get(str(point_id))
    if point is None:
        raise FormalAssemblyError(f"unknown probe_point_id {point_id!r}")
    point.validate(formal=True)
    if transition is None:
        raise FormalAssemblyError(
            "exact probe transition body is absent; hash-only grounding cannot "
            "enter the formal forward-supervised main set")
    state_id = state.get("state_id")
    if not state_id or str(state_id) != point.state_id:
        raise FormalAssemblyError(
            f"state_id mismatch: sample={state_id!r}, point={point.state_id!r}")
    action_instance_id = (state.get("action_instance_id") or
                          risky.get("action_instance_id"))
    if not action_instance_id or str(action_instance_id) != point.action_instance_id:
        raise FormalAssemblyError(
            "action_instance_id is missing or differs from the point")
    raw_action = risky.get("raw_action", "")
    if raw_action != point.raw_action:
        raise FormalAssemblyError(
            f"raw action differs from point: {raw_action!r} != {point.raw_action!r}")
    canonical_action = (risky.get("canonical_action") or
                        state.get("canonical_action"))
    if canonical_action != point.canonical_action:
        raise FormalAssemblyError("canonical action differs from the point")
    state_hash = (state.get("pre_observation_hash") or
                  (state.get("pre_fingerprint") or {}).get("axtree_hash"))
    if not state_hash:
        raise FormalAssemblyError("pre observation hash is missing")
    if str(state_hash) != point.pre_observation_hash:
        dynamic_verified = (
            transition is not None and
            transition.pre_observation.policy_axtree_sha256 ==
            point.pre_observation_hash and
            transition.replay_verification == "dynamic_page_target_contract")
        if not dynamic_verified:
            raise FormalAssemblyError(
                "pre observation hash differs without a verified dynamic target "
                "transition body")
    if point.is_mock:
        raise FormalAssemblyError("mock point is forbidden in the formal main set")
    if state.get("collector_success") is not True:
        raise FormalAssemblyError("formal expert sample requires collector_success=true")
    if point.action_type not in ACTION_META:
        raise FormalAssemblyError(f"unsupported action_type {point.action_type!r}")
    return point


def _legacy_binding(state: dict, rev: dict) -> tuple[str, dict] | None:
    """Explicit development-only class binding retained for old fixtures."""
    risky_text = ((state.get("risky_action") or {}).get("text", "")).lower()
    at = next((a for a, kw in ACTION_KW.items()
               if a in rev and a in ACTION_META and kw in risky_text), None)
    return (at, rev[at]) if at else None


def assemble(reached_path: Path, rev_path: Path, out_dir: Path, *,
             formal: bool = True, truth_path: Path | None = None,
             truth_manifest_path: Path | None = None) -> dict:
    """Materialize single-step samples with a fail-closed formal join.

    Formal mode joins only ``state.probe_point_id`` to the canonical point
    body.  ``formal=False`` is a visibly quarantined legacy compatibility mode;
    it may inspect class-smoke labels but never writes the historical main-set
    filenames and never marks its rows as grounded formal supervision.
    """
    reached = _load_reached(reached_path, formal=formal)
    try:
        points = load_probe_points(rev_path, validate=True) if formal else {}
    except GroundingValidationError as exc:
        raise FormalAssemblyError(str(exc)) from exc
    rev = {} if formal else load_reversibility_details(rev_path)
    transitions_by_point: dict[str, ProbeTransition] = {}
    if formal:
        transition_path = (rev_path.parent / "transitions" /
                           "probe_transitions.v1.jsonl")
        transitions = load_probe_transitions(transition_path, validate=True)
        assert_point_transition_integrity(
            points, transitions, require_all=False)
        transitions_by_point = {
            transition.probe_point_id: transition
            for transition in transitions.values()
        }
    truth = {}
    if formal:
        truth_path = truth_path or out_dir / "eval" / "truth.jsonl"
        truth_manifest_path = (truth_manifest_path or
                               out_dir / "eval" / "TRUTH_MANIFEST.jsonl")
        if truth_path.exists() and truth_manifest_path.exists():
            assert_truth_manifest_integrity(
                truth_path, truth_manifest_path, points)
            truth = truth_by_point_variant(
                load_truth_records(truth_path, points=points).values())
    prompt_provenance = prompts.snapshot_generation(
        root=out_dir, author="assemble-single",
        producer="revact.data.assemble",
        model={"provider": "local", "name": "deterministic-template",
               "revision": "iris-fields-v1"},
        decode_config={
            "strategy": "deterministic", "sampling": False,
            "format": ("iris.v3" if formal else "iris.v2-legacy"),
            "message_topology": "stateless",
            "policy_history_steps": config.POLICY_HISTORY_STEPS,
            "snapshot_max_chars": config.MAX_AXTREE_CHARS_SNAPSHOT,
        })
    prompts_fp = prompt_provenance["prompts_fp"]
    prompt_generation_fp = prompt_provenance["prompt_generation_fp"]

    sft: list[dict] = []
    dpo: list[dict] = []
    blocked: list[dict] = []
    for state in reached:
        state_name = str(state.get("name") or state.get("state_id") or "<unnamed>")
        try:
            if formal:
                point_id = str(state.get("probe_point_id") or
                               (state.get("risky_action") or {}).get(
                                   "probe_point_id") or "")
                transition = transitions_by_point.get(point_id)
                point = _resolve_formal_point(state, points, transition)
                at, info = point.action_type, _point_info(point, transition)
            else:
                binding = _legacy_binding(state, rev)
                if binding is None:
                    raise FormalAssemblyError("no legacy keyword/class binding")
                at, info = binding
                point = None
            obs = (transition.pre_observation.policy_axtree
                   if formal and transition is not None
                   else state.get("axtree_snapshot", ""))
            history, hist_src = prompts.state_history(state)
            if formal and hist_src != "trajectory":
                raise FormalAssemblyError(
                    f"formal history_source must be trajectory, got {hist_src!r}")
            risky_raw = (state.get("risky_action") or {}).get("raw_action", "")

            for vname, violates, requested in [
                    ("constraint", True, False), ("request", False, True)]:
                truth_record = truth.get((point.probe_point_id, vname)) \
                    if point else None
                if formal and truth_record is None:
                    raise FormalAssemblyError(
                        f"missing evaluation truth for {point.probe_point_id}/{vname}")
                normative_risk = (truth_record.normative_risk
                                  if truth_record else None)
                if formal and (
                        truth_record.violates_constraint != violates or
                        truth_record.action_required_for_goal != requested):
                    raise FormalAssemblyError(
                        f"evaluation truth variant semantics mismatch for {vname}")
                g = build_goal(at, vname, state_name)
                f = build_fields(
                    state, at, info, g["goal"], violates, requested,
                    normative_risk=normative_risk,
                )
                input_messages = prompts.build_policy_messages(
                    g["goal"], obs, history,
                    system_prompt=prompts.get("agent_system"),
                    required_actions=[risky_raw, f["answer"]],
                )
                policy_observation = prompts.parse_observation_message(
                    input_messages[-1]["content"])
                policy_input_observation_hash = hashlib.sha256(
                    policy_observation.encode("utf-8")).hexdigest()
                chosen = render_assistant(f)
                if formal and f["decision"] != truth_record.expected_decision:
                    raise FormalAssemblyError(
                        "evaluation truth expected_decision disagrees with policy oracle")
                sample_id = f"{state_name}__{vname}"
                fmt = "iris.v3" if formal else "iris.v2-legacy"
                meta = {
                    "action_type": at,
                    "site": point.site if point else site_of(at),
                    "effect_status": point.effect_status if point else None,
                    "recovery_status": point.recovery_status if point else None,
                    "reversibility": info["label"],
                    "undo_cost_steps": info.get("undo_steps"),
                    "decision": f["decision"],
                    "variant": vname,
                    "normative_risk": normative_risk,
                    "policy_constraint_truth": (
                        truth_record.policy_constraint_truth if truth_record else None),
                    "evaluation_case_id": (truth_record.evaluation_case_id
                                           if truth_record else ""),
                    "normative_truth_source": (truth_record.truth_source
                                               if truth_record else ""),
                    "normative_policy_id": (truth_record.policy_id
                                            if truth_record else ""),
                    "normative_policy_version": (truth_record.policy_version
                                                 if truth_record else ""),
                    "constraint_style": g["style"],
                    "goal_template": g["template_id"],
                    "formal_dataset": formal,
                    "dataset_tier": "formal_point" if formal else "legacy_quarantine",
                    "message_topology": "stateless",
                    "turn_type": "decision",
                    "assistant_turn_types": ["decision"],
                    "reversibility_grounded": formal,
                    "history_source": hist_src,
                    "history_snapshot_alignment": state.get(
                        "history_snapshot_alignment", "exact_raw_snapshot"),
                    "risky_raw_action": risky_raw,
                    "risky_action": ((parsed.to_dict()
                                      if (parsed := parse_action(risky_raw)) else None)
                                     if formal else None),
                    "canonical_action": point.canonical_action if point else "",
                    "legal_at_snapshot": True if formal else None,
                    "action_legal": True if formal else None,
                    "backend_commit": ((point.evidence or {}).get("backend_commit")
                                       if point else None),
                    "violates_constraint": violates,
                    "action_required_for_goal": requested,
                    "prediction_source": f["prediction_source"],
                    "undo_source": f["undo_source"],
                    "undo_source_probe_point_id": f["undo_source_probe_point_id"],
                    "state_id": point.state_id if point else state.get("state_id", ""),
                    "candidate_id": point.candidate_id if point else "",
                    "action_instance_id": (point.action_instance_id if point else ""),
                    "probe_point_id": point.probe_point_id if point else "",
                    "probe_run_id": point.probe_run_id if point else "",
                    "environment_origin": (point.environment_origin if point else
                                           state.get("environment_origin", "legacy")),
                    "environment_family": (point.environment_family if point else ""),
                    "environment_instance": (point.environment_instance if point else ""),
                    "is_mock": point.is_mock if point else bool(state.get("is_mock", False)),
                    "collector_success": state.get("collector_success"),
                    "task_id": point.task_id if point else state.get("task_id", ""),
                    "trajectory_id": (point.trajectory_id if point else
                                      state.get("trajectory_id", "")),
                    "run_id": point.run_id if point else state.get("run_id", ""),
                    "seed": point.seed if point else state.get("seed"),
                    "account": point.account if point else state.get("account", ""),
                    "privilege": (point.privilege if point else
                                  state.get("privilege", "")),
                    "url": point.url if point else state.get("url", ""),
                    "canonical_entity_id": state.get("canonical_entity_id", ""),
                    "page_template_id": state.get("page_template_id", ""),
                    "pre_observation_hash": (point.pre_observation_hash if point else ""),
                    "post_observation_hash": (point.post_observation_hash if point else ""),
                    "post_signal_diff": ({
                        "pre_signal": point.pre_signal,
                        "post_signal": point.post_signal,
                    } if point else None),
                    "transition_id": (
                        transition.transition_id if formal and transition else ""),
                    "transition_record_sha256": (
                        transition_manifest_row(transition)["record_sha256"]
                        if formal and transition else ""),
                    "transition_body_verified": bool(
                        formal and transition is not None),
                    "undo_actions": list(point.undo_actions) if point else [],
                    "undo_semantic_actions": (
                        list(point.undo_semantic_actions) if point else []),
                    "undo_observation_hashes": (
                        list(point.undo_observation_hashes) if point else []),
                    "residual_diff": point.residual_diff if point else None,
                    "budget_k": point.budget_k if point else None,
                    "solver_set": list(point.solver_set) if point else [],
                    "candidate_snapshot_hash": (
                        str((point.evidence or {}).get("candidate_snapshot_hash") or "")
                        if point else ""),
                    "policy_input_observation_hash": policy_input_observation_hash,
                    "evidence": dict(point.evidence) if point else {},
                    "format": fmt,
                    "prompts_fp": prompts_fp,
                    "prompt_generation_fp": prompt_generation_fp,
                }
                sft.append({
                    "sample_id": sample_id,
                    "messages": input_messages + [
                        {"role": "assistant", "content": chosen}],
                    "meta": meta,
                })
                for pair_type, rejected in _dpo_pairs_for(
                        f, state, violates, requested):
                    dpo_meta = dict(meta)
                    dpo_meta.update({
                        "pair_type": pair_type,
                        "negative_source": "synthetic_flip",
                        # Synthetic flips alone cannot enter the formal DPO main set.
                        "formal_dataset": False,
                        "dataset_tier": "synthetic_ablation",
                    })
                    dpo.append({
                        "pair_id": f"{sample_id}__{pair_type}",
                        "prompt": input_messages,
                        "chosen": chosen,
                        "rejected": rejected,
                        "meta": dpo_meta,
                    })
        except (FormalAssemblyError, ValueError) as exc:
            blocked.append({"state": state_name, "reason": str(exc)})

    if formal:
        sft_path = (out_dir / "train" / "formal" /
                    config.FORMAL_SFT_PATH.name)
        dpo_path = out_dir / "train" / "ablation" / "iris_dpo_synthetic_point_v1.jsonl"
        report_path = out_dir / "train" / "formal" / "assembly_report.json"
    else:
        sft_path = out_dir / "train" / "quarantine" / "legacy_sft.jsonl"
        dpo_path = out_dir / "train" / "quarantine" / "legacy_dpo.jsonl"
        report_path = out_dir / "train" / "quarantine" / "assembly_report.json"
    _write(sft_path, sft)
    _write(dpo_path, dpo)
    report = {
        "formal": formal,
        "n_reached": len(reached),
        "n_sft": len(sft),
        "n_dpo": len(dpo),
        "n_dpo_synthetic": len(dpo),
        "formal_dpo_eligible": False,
        "n_blocked_states": len(blocked),
        "blocked": blocked,
        "sft_path": str(sft_path),
        "dpo_path": str(dpo_path),
        "prompts_fp": prompts_fp,
        "prompt_generation_fp": prompt_generation_fp,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return {**report, "report_path": str(report_path), "samples": sft}


# --------------------------------------------------------------------------- #
# io helpers
# --------------------------------------------------------------------------- #
def _load_reached(path: Path, *, formal: bool = False) -> list:
    """Load reached states without silently collapsing formal point identities.

    Frozen scale files historically appended the same human-readable ``name``
    and the legacy reader kept the last row.  A formal state/action point may
    never be selected by append order, so its composite identity must be unique
    or assembly fails closed.
    """
    rows = [json.loads(line) for line in path.open(encoding="utf-8")
            if line.strip()]
    rows = [row for row in rows if row.get("reached") and row.get("risky_action")]
    if not formal:
        latest = {row["name"]: row for row in rows}
        return list(latest.values())

    indexed: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        risky = row.get("risky_action") or {}
        key = (
            str(row.get("state_id") or ""),
            str(row.get("action_instance_id") or
                risky.get("action_instance_id") or ""),
            str(row.get("probe_point_id") or risky.get("probe_point_id") or ""),
        )
        if not key[0]:
            raise FormalAssemblyError("formal reached state is missing state_id")
        if key in indexed:
            raise FormalAssemblyError(
                "duplicate formal reached state/action/point identity: " +
                "/".join(part or "<missing>" for part in key))
        indexed[key] = row
    return list(indexed.values())


def _write(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
