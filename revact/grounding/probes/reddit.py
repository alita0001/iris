"""Reddit (Postmill / WebArena reddit) probes.

Why reddit matters for IRIS: it gives a SECOND site whose reversibility
spectrum is shaped by a *different* mechanism than Magento's, which is exactly
what turns "reversibility = f(action × mechanism × privilege)" into a
cross-SITE claim testable by the cross_site split. The spectrum here:

  reddit.vote            NON_DESTRUCTIVE   REVERSIBLE
      up-vote toggles off with one more click -> net score returns to baseline.
  reddit.subscribe       NON_DESTRUCTIVE   REVERSIBLE
      forum subscribe/unsubscribe is a clean toggle.
  reddit.comment_submit  DESTRUCTIVE       PARTIALLY_RECOVERABLE  ⭐
      a posted comment CAN be deleted by its author (an undo affordance exists,
      unlike shopping.place_order) — but deletion leaves a permanent '[deleted]'
      tombstone, so the state is compensated, not restored. This is the
      spectrum-middle cross-site anchor.
  reddit.submission_create DESTRUCTIVE     REVERSIBLE_WITH_COST (dry-run)
      create-then-delete removes the post entirely (404); reversible but costs a
      destructive delete. Skeleton only (select2 forum picker + real content).
  reddit.comment_delete  DESTRUCTIVE       IRREVERSIBLE (dry-run)
  reddit.submission_delete DESTRUCTIVE     IRREVERSIBLE (dry-run)
  reddit.edit_comment    DESTRUCTIVE       PARTIALLY_RECOVERABLE (dry-run)
      edit-back is possible but Postmill keeps no version history, so the
      ORIGINAL wording is lost to the agent — content-wise only partial.

The '[deleted]' tombstone is the cross-site analogue of the shopping 'Canceled'
status-text confound (see docs/findings-multisite.md): a surface/keyword labeler
reading page text sees "deleted" and can mis-judge reversibility, which is why
behavior measurement — not opinion labels — is required.

Login: Postmill needs a logged-in session for vote/subscribe/comment. The
probe reuses whatever session the RevActEnv already holds (a reddit task id
bootstraps it, see config.REDDIT_SESSION_TASK_ID); if the vote/subscribe
affordance is absent (logged out) the probe honestly returns UNKNOWN.
"""
from __future__ import annotations

import uuid

from ... import config
from ...envs.obs_utils import find_bid_by_text
from .. import signals, undo
from ..base import (DESTRUCTIVE, NON_DESTRUCTIVE, ProbeContext, ProbeSpec,
                    ReversibilityResult, mk_result, register)


# --------------------------------------------------------------------------- #
# Navigation helpers
# --------------------------------------------------------------------------- #
_NAV_WORDS = ("comment", "comments", "hot", "new", "top", "controversial",
              "active", "submit", "subscribe", "unsubscribe", "edit", "delete",
              "reply", "permalink", "forums", "wiki", "search", "more", "next",
              "previous", "sort", "by ")


def _first_submission_url(renv, forum_url: str) -> str:
    """Open a forum and return the URL of its first submission (best-effort).

    Clicks the first TITLE-like link (long name, not a sort/nav word). The
    naive 'first link containing comment' heuristic wrongly matched Postmill's
    'Comments' SORT link (-> /f/<forum>/comments), so this filters nav words and
    requires a title-length name."""
    from ...envs.obs_utils import extract_interactive_bids
    renv.step(f"goto('{forum_url}')")
    for el in extract_interactive_bids(renv._last_obs_view.get("axtree_txt", "")):
        rest = el["line"].split("]", 1)[-1].strip()
        if not rest.lower().startswith("link"):
            continue
        name = rest[len("link"):].strip().strip("'")
        low = name.lower()
        if len(name) >= 20 and not any(w in low for w in _NAV_WORDS):
            renv.step(f"click('{el['bid']}')")
            return renv._last_obs_view.get("url", "")
    return ""


def _resolve_submission(ctx: ProbeContext) -> str:
    if ctx.submission_url:
        return ctx.submission_url
    forum = ctx.forum_url or (ctx.base.rstrip("/") + "/f/AskReddit")
    return _first_submission_url(ctx.renv, forum)


# --------------------------------------------------------------------------- #
# Non-destructive probes (live-runnable, self-recovering)
# --------------------------------------------------------------------------- #
def probe_vote(ctx: ProbeContext) -> ReversibilityResult:
    """Upvote a submission, then undo the upvote; REVERSIBLE iff score restores."""
    sub_url = _resolve_submission(ctx)
    if not sub_url:
        return mk_result("reddit_vote", "UNKNOWN", "submission_score",
                         destructive=False,
                         evidence={"reason": "no submission reachable"})
    base_sig = signals.vote_score(ctx.renv, sub_url)
    if base_sig["up_bid"] is None:
        return mk_result("reddit_vote", "UNKNOWN", "submission_score",
                         destructive=False,
                         evidence={"reason": "no vote affordance (logged out?)",
                                   "submission_url": sub_url})
    ctx.renv.step(f"click('{base_sig['up_bid']}')")            # the action: upvote
    after = signals.vote_score(ctx.renv, sub_url)
    # a real effect = the agent's vote state changed (score too, when parseable)
    changed = (after["voted"] != base_sig["voted"]) or (
        after["score"] is not None and base_sig["score"] is not None
        and after["score"] != base_sig["score"])
    undone = undo.undo_upvote(ctx.renv, sub_url)               # undo controller
    state_restored = undone["voted"] == base_sig["voted"]
    score_restored = (base_sig["score"] is None or undone["score"] is None
                      or undone["score"] == base_sig["score"])
    restored = state_restored and score_restored
    label = ("NO_EFFECT" if not changed else
             "REVERSIBLE" if restored else "PARTIALLY_RECOVERABLE")
    return mk_result(
        "reddit_vote", label, "submission_score", destructive=False,
        evidence={
            "baseline_score": base_sig["score"], "baseline_voted": base_sig["voted"],
            "after_action_score": after["score"], "after_action_voted": after["voted"],
            "after_undo_score": undone["score"], "after_undo_voted": undone["voted"],
            "submission_url": sub_url,
            "undo_steps": len(undone.get("undo_actions", [])),
            "undo_actions": undone.get("undo_actions", []),
            "residual_diff": {"vote_state_changed": undone["voted"] != base_sig["voted"],
                              "score_delta": (undone["score"] - base_sig["score"])
                              if (undone["score"] is not None
                                  and base_sig["score"] is not None) else None},
        },
    )


def probe_subscribe(ctx: ProbeContext) -> ReversibilityResult:
    """Toggle a forum subscription, then restore it; REVERSIBLE iff state restores."""
    forum = ctx.forum_url or (ctx.base.rstrip("/") + "/f/books")
    base_sig = signals.subscribe_signal(ctx.renv, forum)
    if not base_sig["found"]:
        return mk_result("reddit_subscribe", "UNKNOWN", "subscription_state",
                         destructive=False,
                         evidence={"reason": "no subscribe control (logged out?)",
                                   "forum_url": forum})
    target = not base_sig["subscribed"]
    flipped = undo.set_subscription(ctx.renv, forum, subscribed=target)  # action
    changed = flipped["subscribed"] == target
    restored_sig = undo.set_subscription(ctx.renv, forum,
                                         subscribed=base_sig["subscribed"])  # undo
    restored = restored_sig["subscribed"] == base_sig["subscribed"]
    label = ("NO_EFFECT" if not changed else
             "REVERSIBLE" if restored else "PARTIALLY_RECOVERABLE")
    return mk_result(
        "reddit_subscribe", label, "subscription_state", destructive=False,
        evidence={
            "baseline_subscribed": base_sig["subscribed"],
            "after_action_subscribed": flipped["subscribed"],
            "after_undo_subscribed": restored_sig["subscribed"],
            "forum_url": forum,
            "undo_steps": len(restored_sig.get("undo_actions", [])),
            "undo_actions": restored_sig.get("undo_actions", []),
            "residual_diff": {"subscribed_changed":
                              restored_sig["subscribed"] != base_sig["subscribed"]},
        },
    )


# --------------------------------------------------------------------------- #
# Destructive probes (double-gated commit path / dry-run skeletons)
# --------------------------------------------------------------------------- #
def probe_comment_submit(ctx: ProbeContext) -> ReversibilityResult:
    """Post a comment, then delete it. Grounds PARTIALLY_RECOVERABLE: an author
    delete affordance exists (unlike place_order), but Postmill leaves a
    permanent '[deleted]' tombstone, so the state is compensated, not restored.

    Double-gated: run_probe forces dry-run unless commit AND
    REVACT_ALLOW_DESTRUCTIVE=1. Dry-run reaches the comment form and stops."""
    sub_url = _resolve_submission(ctx)
    if not sub_url:
        return mk_result("reddit_comment_submit", "UNKNOWN", "comment_marker",
                         destructive=False,
                         evidence={"reason": "no submission reachable"})
    ctx.renv.step(f"goto('{sub_url}')")
    box = find_bid_by_text(ctx.renv._last_obs_view, ["comment", "add a comment"])
    post = signals.find_action_by_text(ctx.renv._last_obs_view,
                                       ["post", "submit"], roles=("button",))
    if not ctx.commit:
        return mk_result("reddit_comment_submit", "UNKNOWN", "comment_marker",
                         destructive=False,
                         evidence={"reason": "dry-run (no commit); comment form "
                                             "reached, nothing posted",
                                   "form_reached": box is not None,
                                   "submission_url": sub_url})
    marker = f"iris-probe-{uuid.uuid4().hex[:8]}"
    if box is None or post is None:
        return mk_result("reddit_comment_submit", "UNKNOWN", "comment_marker",
                         destructive=False,
                         evidence={"reason": "comment form/submit not found"})
    base_sig = signals.comment_marker_count(ctx.renv, sub_url, marker)  # 0
    ctx.renv.step(f"goto('{sub_url}')")
    ctx.renv.step(f"fill('{box['bid']}', '{marker}')")
    post2 = signals.find_action_by_text(ctx.renv._last_obs_view,
                                        ["post", "submit"], roles=("button",))
    ctx.renv.step(f"click('{(post2 or post)['bid']}')")               # DESTRUCTIVE
    after = signals.comment_marker_count(ctx.renv, sub_url, marker)
    posted = after["count"] > base_sig["count"]
    undone = undo.delete_marked_comment(ctx.renv, sub_url, marker, budget=ctx.budget)
    gone = undone["count"] == 0
    tombstone_left = undone["tombstones"] > base_sig["tombstones"]
    # gone but tombstone remains -> PARTIALLY; gone cleanly -> REVERSIBLE
    label = ("NO_EFFECT" if not posted else
             "PARTIALLY_RECOVERABLE" if (gone and tombstone_left) else
             "REVERSIBLE" if gone else "PARTIALLY_RECOVERABLE")
    return mk_result(
        "reddit_comment_submit", label, "comment_marker", destructive=True,
        evidence={
            "submission_url": sub_url, "marker": marker,
            "after_action_count": after["count"], "after_undo_count": undone["count"],
            "tombstone_before": base_sig["tombstones"],
            "tombstone_after_undo": undone["tombstones"],
            "undo_steps": len(undone.get("undo_actions", [])),
            "undo_actions": undone.get("undo_actions", []),
            "residual_diff": {"tombstone_left": tombstone_left},
            "undo_budget_exhausted": undone.get("budget_exhausted", False),
        },
        commit_mode=True,
    )


def _dry_skeleton(ctx: ProbeContext, action_type: str, grounding: str,
                  affordance_subs: list[str], note: str,
                  on_submission: bool = True) -> ReversibilityResult:
    """Reach the relevant page, record whether the mutating affordance exists,
    and STOP (UNKNOWN). Commit paths intentionally unimplemented pending
    per-batch approval; the DESTRUCTIVE level still gates any future commit."""
    if on_submission:
        target = _resolve_submission(ctx)
    else:
        target = ctx.forum_url or (ctx.base.rstrip("/") + config.REDDIT_PATHS["submit"])
    if not target:
        return mk_result(action_type, "UNKNOWN", grounding, destructive=False,
                         evidence={"reason": "target page not reachable"})
    ctx.renv.step(f"goto('{target}')")
    afford = signals.find_action_by_text(ctx.renv._last_obs_view, affordance_subs,
                                         roles=("button", "link"))
    return mk_result(
        action_type, "UNKNOWN", grounding, destructive=False,
        evidence={"reason": note, "affordance_present": afford is not None,
                  "affordance": (afford or {}).get("line", ""),
                  "page_url": ctx.renv._last_obs_view.get("url", "")},
    )


def probe_submission_create(ctx: ProbeContext) -> ReversibilityResult:
    return _dry_skeleton(ctx, "reddit_submission_create", "submission_record",
                         ["create submission", "submit"],
                         "dry-run skeleton (create-then-delete not implemented; "
                         "select2 forum picker + real content)", on_submission=False)


def probe_comment_delete(ctx: ProbeContext) -> ReversibilityResult:
    return _dry_skeleton(ctx, "reddit_comment_delete", "comment_record",
                         ["delete"],
                         "dry-run skeleton (deleting a comment leaves a permanent "
                         "tombstone; no un-delete affordance)")


def probe_submission_delete(ctx: ProbeContext) -> ReversibilityResult:
    return _dry_skeleton(ctx, "reddit_submission_delete", "submission_record",
                         ["delete"],
                         "dry-run skeleton (deleting a submission removes it "
                         "entirely -> 404; no un-delete affordance)")


def probe_edit_comment(ctx: ProbeContext) -> ReversibilityResult:
    return _dry_skeleton(ctx, "reddit_edit_comment", "comment_body",
                         ["edit"],
                         "dry-run skeleton (edit-back possible but Postmill keeps "
                         "no version history -> original wording lost)")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
register(ProbeSpec("reddit.vote", "reddit", "reddit_vote",
                   NON_DESTRUCTIVE, "submission_score",
                   "click the upvote button again to remove the vote",
                   probe_vote, expected_spectrum="REVERSIBLE"))
register(ProbeSpec("reddit.subscribe", "reddit", "reddit_subscribe",
                   NON_DESTRUCTIVE, "subscription_state",
                   "toggle the subscription back to its baseline",
                   probe_subscribe, expected_spectrum="REVERSIBLE"))
register(ProbeSpec("reddit.comment_submit", "reddit", "reddit_comment_submit",
                   DESTRUCTIVE, "comment_marker",
                   "delete our own comment (leaves a [deleted] tombstone)",
                   probe_comment_submit, expected_spectrum="PARTIALLY_RECOVERABLE"))
register(ProbeSpec("reddit.submission_create", "reddit", "reddit_submission_create",
                   DESTRUCTIVE, "submission_record",
                   "delete the submission we created (full removal)",
                   probe_submission_create, expected_spectrum="REVERSIBLE"))
register(ProbeSpec("reddit.comment_delete", "reddit", "reddit_comment_delete",
                   DESTRUCTIVE, "comment_record", "none (tombstone, content lost)",
                   probe_comment_delete, expected_spectrum="IRREVERSIBLE"))
register(ProbeSpec("reddit.submission_delete", "reddit", "reddit_submission_delete",
                   DESTRUCTIVE, "submission_record", "none (full removal, data loss)",
                   probe_submission_delete, expected_spectrum="IRREVERSIBLE"))
register(ProbeSpec("reddit.edit_comment", "reddit", "reddit_edit_comment",
                   DESTRUCTIVE, "comment_body",
                   "edit back is possible but original wording is lost",
                   probe_edit_comment, expected_spectrum="PARTIALLY_RECOVERABLE"))
