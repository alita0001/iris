"""Undo controllers: agent-action-space routines that try to restore a signal.

Each controller returns the final signal dict augmented with:
  * ``undo_actions``      exact env actions executed while undoing
  * ``budget_exhausted``  True if items remained when the step budget ran out

Controllers only use actions available to the agent itself (click on the
per-row remove/delete links) — reversibility is agent-relative by definition.
"""
from __future__ import annotations

from . import signals


def _drain(renv, signal_fn, url: str, budget: int) -> dict:
    """Click the first remove-link until the signal is empty or budget is out.

    Some Magento removals (compare list) pop a confirm modal; when an OK
    button appears right after the remove click, confirm it."""
    sig = signal_fn(renv, url)
    undo_actions: list[str] = []
    it = 0
    while sig["remove_bids"] and it < budget:
        action = f"click('{sig['remove_bids'][0]}')"
        renv.step(action)
        undo_actions.append(action)
        ok = signals.find_action_by_text(renv._last_obs_view, ["ok"],
                                         roles=("button",))
        if ok:
            undo_actions.append(f"click('{ok['bid']}')")
            renv.step(undo_actions[-1])
        sig = signal_fn(renv, url)
        it += 1
    sig["undo_actions"] = undo_actions
    sig["budget_exhausted"] = bool(sig["remove_bids"]) and it >= budget
    return sig


def empty_cart(renv, cart_url: str, max_iter: int = 12) -> dict:
    return _drain(renv, signals.cart_signal, cart_url, max_iter)


def empty_wishlist(renv, wishlist_url: str, max_iter: int = 12) -> dict:
    return _drain(renv, signals.wishlist_signal, wishlist_url, max_iter)


def empty_compare(renv, compare_url: str, max_iter: int = 12) -> dict:
    return _drain(renv, signals.compare_signal, compare_url, max_iter)


def set_newsletter(renv, manage_url: str, subscribed: bool) -> dict:
    """Set the newsletter checkbox to `subscribed` and save. Returns the final
    signal plus the undo actions taken."""
    sig = signals.newsletter_signal(renv, manage_url)
    undo_actions: list[str] = []
    if sig["found"] and sig["checked"] != subscribed:
        undo_actions.append(f"click('{sig['bid']}')")
        renv.step(undo_actions[-1])
        save = signals.find_action_by_text(renv._last_obs_view, ["save"])
        if save:
            undo_actions.append(f"click('{save['bid']}')")
            renv.step(undo_actions[-1])
        sig = signals.newsletter_signal(renv, manage_url)
    sig["undo_actions"] = undo_actions
    return sig


def delete_last_address(renv, address_url: str) -> dict:
    """Delete the LAST additional address entry (the one this probe created),
    confirming the modal if one appears. Self-recovery only — never used on
    pre-existing addresses."""
    sig = signals.address_signal(renv, address_url)
    undo_actions: list[str] = []
    if sig["delete_bids"]:
        undo_actions.append(f"click('{sig['delete_bids'][-1]}')")
        renv.step(undo_actions[-1])
        # Magento asks "Are you sure...?" in a modal with an OK button.
        ok = signals.find_action_by_text(renv._last_obs_view, ["ok"])
        if ok:
            undo_actions.append(f"click('{ok['bid']}')")
            renv.step(undo_actions[-1])
        sig = signals.address_signal(renv, address_url)
    sig["undo_actions"] = undo_actions
    return sig


# --------------------------------------------------------------------------- #
# Reddit (Postmill) undo controllers
# --------------------------------------------------------------------------- #
def undo_upvote(renv, submission_url: str) -> dict:
    """Invert a single upvote by clicking the (toggling) 'Upvote' button once.

    Postmill vote buttons toggle: clicking 'Upvote' on an already-upvoted
    submission removes the vote, returning the score to baseline. The
    reddit.vote probe applies exactly one upvote, so its undo is exactly one
    more click on the same control. Agent-relative: only the agent's own vote
    button."""
    sig = signals.vote_score(renv, submission_url)
    undo_actions: list[str] = []
    bid = sig.get("up_bid")
    if bid is not None:
        undo_actions.append(f"click('{bid}')")
        renv.step(undo_actions[-1])
        sig = signals.vote_score(renv, submission_url)
    sig["undo_actions"] = undo_actions
    return sig


def set_subscription(renv, forum_url: str, subscribed: bool) -> dict:
    """Set a forum subscription to `subscribed` (clicks the toggle if needed)."""
    sig = signals.subscribe_signal(renv, forum_url)
    undo_actions: list[str] = []
    if sig["found"] and sig["subscribed"] != subscribed:
        undo_actions.append(f"click('{sig['bid']}')")
        renv.step(undo_actions[-1])
        sig = signals.subscribe_signal(renv, forum_url)
    sig["undo_actions"] = undo_actions
    return sig


def delete_marked_comment(renv, submission_url: str, marker: str,
                          budget: int = 6) -> dict:
    """Delete comment(s) whose text carries `marker` (self-recovery for the
    comment-submit probe). Confirms a modal/dialog if one is surfaced as an OK
    button. Leaves Postmill's '[deleted]' tombstone behind by design — that
    residue is why comment_submit grounds as PARTIALLY_RECOVERABLE, not
    REVERSIBLE."""
    sig = signals.comment_marker_count(renv, submission_url, marker)
    undo_actions: list[str] = []
    it = 0
    while sig["count"] and sig["delete_bids"] and it < budget:
        action = f"click('{sig['delete_bids'][0]}')"
        renv.step(action)
        undo_actions.append(action)
        ok = signals.find_action_by_text(renv._last_obs_view, ["ok", "yes", "confirm"],
                                         roles=("button",))
        if ok:
            undo_actions.append(f"click('{ok['bid']}')")
            renv.step(undo_actions[-1])
        sig = signals.comment_marker_count(renv, submission_url, marker)
        it += 1
    sig["undo_actions"] = undo_actions
    sig["budget_exhausted"] = bool(sig["count"]) and it >= budget
    return sig
