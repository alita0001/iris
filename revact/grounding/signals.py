"""Backend signals for the grounded probes (shopping/Magento + reddit/Postmill).

A *signal* is the deterministic quantity a probe compares before/after the
action and after undo. On the real site there is no DB access, so "backend"
means the most structural UI readout available (counts of per-row action
links, order-number cells, checkbox states, a submission's net score,
subscription state) — NOT free-text keyword matching.

All functions operate on a RevActEnv-like object exposing
``step(action) -> (obs, reward, term, trunc, info, view)`` and
``_last_obs_view``.

Layout: shopping (Magento) signals first, then a reddit (Postmill) section.
"""
from __future__ import annotations

import re
from typing import Optional

from ..envs.obs_utils import extract_interactive_bids

# Order ids are exactly-9-digit tokens, but ONLY when they appear as the quoted
# name of a table/link element (gridcell '000000193'); a bare \b\d{9}\b over the
# whole page would also match phone numbers / SKUs.
_ORDER_LINE_RE = re.compile(r"\b(?:gridcell|cell|link|row|columnheader)\b[^']*'(\d{9})'")


def role_of(line: str) -> str:
    return line.split("]", 1)[-1].strip().split(" ", 1)[0]


def radios(view) -> list:
    return [el["bid"] for el in extract_interactive_bids(view.get("axtree_txt", ""))
            if role_of(el["line"]) == "radio"]


def find_action_by_text(view, subs, roles=("link", "button")) -> Optional[dict]:
    """First ACTIONABLE control (link/button) whose name contains one of `subs`
    as a whole word; None otherwise.

    Word-boundary matching prevents e.g. "cancel" matching "Cancellation
    Policy"; role filtering excludes status text such as a gridcell 'Canceled'
    (which is not a cancel action). Returns the element so callers can record
    WHAT matched as evidence.
    """
    pats = [re.compile(r"\b" + re.escape(s.lower()) + r"\b") for s in subs]
    for el in extract_interactive_bids(view.get("axtree_txt", "")):
        rest = el["line"].split("]", 1)[-1].strip()
        if role_of(el["line"]) in roles and any(p.search(rest.lower()) for p in pats):
            return el
    return None


_LINK_NAME_RE = re.compile(r"\blink '([^']*)'")


def _count_links(view, name_subs) -> list:
    """bids of links whose QUOTED NAME contains any of name_subs.

    Matches inside the quoted name (not a prefix match): Magento prepends
    icon glyphs to control names (``link '\\ue616 Remove Product'``), so
    ``link 'remove product`` -style prefix matching silently fails on live.
    """
    if isinstance(name_subs, str):
        name_subs = [name_subs]
    subs = [s.lower() for s in name_subs]
    out = []
    for el in extract_interactive_bids(view.get("axtree_txt", "")):
        m = _LINK_NAME_RE.search(el["line"].lower())
        if m and any(s in m.group(1) for s in subs):
            out.append(el["bid"])
    return out


# --------------------------------------------------------------------------- #
# Cart
# --------------------------------------------------------------------------- #
def cart_signal(renv, cart_url: str) -> dict:
    _o, _r, _t, _tr, _i, view = renv.step(f"goto('{cart_url}')")
    low = (view.get("axtree_txt", "") or "").lower()
    empty_txt = ("no items in your shopping cart" in low
                 or "shopping cart is empty" in low)
    remove_bids = _count_links(view, ["remove item"])
    # Row count is authoritative; empty text is advisory only (Magento renders
    # an empty-list message in sidebar widgets even when the main list has rows).
    count = len(remove_bids)
    return {"empty": count == 0, "empty_txt": empty_txt, "count": count,
            "remove_bids": remove_bids}


# --------------------------------------------------------------------------- #
# Orders
# --------------------------------------------------------------------------- #
def order_ids(renv, hist_url: str) -> set:
    """Order numbers on the order-history page (quoted 9-digit grid/link names)."""
    _o, _r, _t, _tr, _i, view = renv.step(f"goto('{hist_url}')")
    ids = set()
    for ln in (view.get("axtree_txt", "") or "").splitlines():
        for m in _ORDER_LINE_RE.finditer(ln):
            ids.add(m.group(1))
    return ids


# --------------------------------------------------------------------------- #
# Wishlist
# --------------------------------------------------------------------------- #
def wishlist_signal(renv, wishlist_url: str) -> dict:
    _o, _r, _t, _tr, _i, view = renv.step(f"goto('{wishlist_url}')")
    # live Magento: 'Remove This Item' (icon-prefixed); mock: 'Remove item'
    remove_bids = _count_links(view, ["remove this item", "remove item"])
    count = len(remove_bids)
    return {"empty": count == 0, "count": count, "remove_bids": remove_bids}


# --------------------------------------------------------------------------- #
# Product compare list
# --------------------------------------------------------------------------- #
def compare_signal(renv, compare_url: str) -> dict:
    _o, _r, _t, _tr, _i, view = renv.step(f"goto('{compare_url}')")
    remove_bids = _count_links(view, ["remove product", "remove item"])
    count = len(remove_bids)
    return {"empty": count == 0, "count": count, "remove_bids": remove_bids}


def _checkbox_state(line_low: str) -> bool:
    """Parse a checkbox line's checked state across serializations:
    live BrowserGym renders ``checked='true'`` / ``checked='false'``;
    simpler serializers render a bare ``, checked`` token."""
    m = re.search(r"checked='?(true|false)'?", line_low)
    if m:
        return m.group(1) == "true"
    return bool(re.search(r",\s*checked\b", line_low))


# --------------------------------------------------------------------------- #
# Newsletter subscription (checkbox state)
# --------------------------------------------------------------------------- #
def newsletter_signal(renv, manage_url: str) -> dict:
    """Checkbox state of 'General Subscription' on /newsletter/manage/."""
    _o, _r, _t, _tr, _i, view = renv.step(f"goto('{manage_url}')")
    for el in extract_interactive_bids(view.get("axtree_txt", "")):
        low = el["line"].lower()
        if "checkbox" in low and "subscription" in low:
            return {"found": True, "checked": _checkbox_state(low),
                    "bid": el["bid"], "line": el["line"]}
    return {"found": False, "checked": None, "bid": None, "line": ""}


# --------------------------------------------------------------------------- #
# Address book
# --------------------------------------------------------------------------- #
def address_signal(renv, address_url: str) -> dict:
    """Count of additional address entries = count of 'Delete Address' links.

    The default billing/shipping address has no delete affordance in Magento,
    so this counts exactly the entries a customer (or this probe) added.
    """
    _o, _r, _t, _tr, _i, view = renv.step(f"goto('{address_url}')")
    delete_bids = _count_links(view, "delete address")
    return {"count": len(delete_bids), "delete_bids": delete_bids}


# --------------------------------------------------------------------------- #
# Reddit (Postmill) signals
#
# Postmill renders (via BrowserGym's a11y flattening / the mock):
#   * a submission vote widget with buttons named 'Upvote' / 'Downvote' and a
#     net score rendered as text 'Score: <n>' (or a11y 'N points');
#   * a per-forum 'Subscribe' / 'Unsubscribe' toggle button;
#   * comments as blocks, each with a 'Delete' control ON OUR OWN comments, and
#     a permanent '[deleted]' tombstone left where a parent-with-replies was
#     removed (see docs/findings — the cross-site analogue of the shopping
#     'Canceled' status-text confound).
# The score integer and the subscribe-button name are the structural signals.
# --------------------------------------------------------------------------- #
_INT_RE = re.compile(r"(-?\d+)")


def _submission_vote(lines: list[str]) -> tuple:
    """Parse the submission's own vote widget -> (score, voted, up_line_index).

    The submission widget is the FIRST vote control on the page (comment widgets
    follow). Postmill renders the up-control as ``button 'Upvote'`` when the user
    has NOT voted and ``button 'Retract upvote'`` when they HAVE — both end in
    ``upvote'``, so we match that suffix (matching only ``'upvote'`` would skip
    the active widget and grab the first comment's score, the active-state
    analogue of the tombstone/'Canceled' confound). The net score is the integer
    StaticText immediately after it, before 'Downvote'."""
    for i, ln in enumerate(lines):
        low = ln.lower()
        if "button" in low and "upvote'" in low:
            voted = "retract" in low
            score = None
            for j in range(i + 1, min(i + 4, len(lines))):
                nxt = lines[j].lower()
                if "downvote" in nxt:
                    break
                if "statictext" in nxt:
                    m = _INT_RE.search(lines[j])
                    if m:
                        score = int(m.group(1))
                        break
            return score, voted, i
    return None, False, -1


def vote_score(renv, submission_url: str) -> dict:
    """Submission vote state: net score + whether the agent's vote is active.

    ``score=None`` when unparseable (honest UNKNOWN upstream). ``up_bid`` is the
    first up-control (its label is 'Upvote' or 'Retract upvote'); clicking it
    toggles the agent's own vote."""
    _o, _r, _t, _tr, _i, view = renv.step(f"goto('{submission_url}')")
    lines = (view.get("axtree_txt", "") or "").splitlines()
    score, voted, idx = _submission_vote(lines)
    up_bid = None
    if idx >= 0:
        bm = re.search(r"\[(\d+)\]", lines[idx])
        up_bid = bm.group(1) if bm else None
    return {"score": score, "voted": voted, "up_bid": up_bid,
            "up_line": lines[idx] if idx >= 0 else ""}


def subscribe_signal(renv, forum_url: str) -> dict:
    """Subscription state of a forum: subscribed iff an 'Unsubscribe' control is
    shown (Postmill swaps the button label with state)."""
    _o, _r, _t, _tr, _i, view = renv.step(f"goto('{forum_url}')")
    unsub = find_action_by_text(view, ["unsubscribe"], roles=("button", "link"))
    sub = find_action_by_text(view, ["subscribe"], roles=("button", "link"))
    subscribed = unsub is not None
    ctrl = unsub or sub
    return {"found": ctrl is not None, "subscribed": subscribed,
            "bid": (ctrl or {}).get("bid"), "line": (ctrl or {}).get("line", "")}


def comment_marker_count(renv, submission_url: str, marker: str) -> dict:
    """Count comment blocks whose visible text contains `marker` (used to verify
    a self-recovering comment probe removed exactly its own comment)."""
    _o, _r, _t, _tr, _i, view = renv.step(f"goto('{submission_url}')")
    txt = (view.get("axtree_txt", "") or "")
    count = txt.count(marker)
    del_bids = [el["bid"] for el in extract_interactive_bids(txt)
                if role_of(el["line"]) in ("link", "button")
                and "delete" in el["line"].lower()]
    tombstones = len(re.findall(r"\[deleted\]", txt, re.IGNORECASE))
    return {"count": count, "delete_bids": del_bids, "tombstones": tombstones}
