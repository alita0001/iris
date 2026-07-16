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
def _view_at(renv, url: str, *, navigate: bool) -> dict:
    """Return the current view, optionally navigating to ``url`` first.

    Live point probes use ``navigate=False`` after explicitly recording the
    navigation as an undo action.  Legacy probes keep the default behavior, so
    instrumentation changes do not silently alter their traces.
    """
    if navigate:
        _o, _r, _t, _tr, _i, view = renv.step(f"goto('{url}')")
        return view
    return renv._last_obs_view


def cart_signal(renv, cart_url: str, *, navigate: bool = True) -> dict:
    view = _view_at(renv, cart_url, navigate=navigate)
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
def order_ids(renv, hist_url: str, *, navigate: bool = True) -> set:
    """Order numbers on the order-history page (quoted 9-digit grid/link names)."""
    view = _view_at(renv, hist_url, navigate=navigate)
    ids = set()
    for ln in (view.get("axtree_txt", "") or "").splitlines():
        for m in _ORDER_LINE_RE.finditer(ln):
            ids.add(m.group(1))
    return ids


# --------------------------------------------------------------------------- #
# Wishlist
# --------------------------------------------------------------------------- #
def wishlist_signal(renv, wishlist_url: str, *, navigate: bool = True) -> dict:
    view = _view_at(renv, wishlist_url, navigate=navigate)
    # live Magento: 'Remove This Item' (icon-prefixed); mock: 'Remove item'
    remove_bids = _count_links(view, ["remove this item", "remove item"])
    count = len(remove_bids)
    return {"empty": count == 0, "count": count, "remove_bids": remove_bids}


# --------------------------------------------------------------------------- #
# Product compare list
# --------------------------------------------------------------------------- #
def compare_signal(renv, compare_url: str, *, navigate: bool = True) -> dict:
    view = _view_at(renv, compare_url, navigate=navigate)
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
def newsletter_signal(renv, manage_url: str, *, navigate: bool = True) -> dict:
    """Checkbox state of 'General Subscription' on /newsletter/manage/."""
    view = _view_at(renv, manage_url, navigate=navigate)
    for el in extract_interactive_bids(view.get("axtree_txt", "")):
        low = el["line"].lower()
        if "checkbox" in low and "subscription" in low:
            return {"found": True, "checked": _checkbox_state(low),
                    "bid": el["bid"], "line": el["line"]}
    return {"found": False, "checked": None, "bid": None, "line": ""}


# --------------------------------------------------------------------------- #
# Address book
# --------------------------------------------------------------------------- #
def address_signal(renv, address_url: str, *, navigate: bool = True) -> dict:
    """Count of additional address entries = count of 'Delete Address' links.

    The default billing/shipping address has no delete affordance in Magento,
    so this counts exactly the entries a customer (or this probe) added.
    """
    view = _view_at(renv, address_url, navigate=navigate)
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
    """Parse the submission's own widget -> (score, direction, up_idx, down_idx).

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
            direction = 1 if "retract" in low else 0
            score = None
            down_index = -1
            for j in range(i + 1, min(i + 6, len(lines))):
                nxt = lines[j].lower()
                if "downvote" in nxt:
                    down_index = j
                    if "retract" in nxt:
                        direction = -1
                    break
                if score is None and "statictext" in nxt:
                    m = _INT_RE.search(lines[j])
                    if m:
                        score = int(m.group(1))
            return score, direction, i, down_index
    return None, 0, -1, -1


def _url_entity(url: str, kind: str) -> str:
    if kind == "submission":
        match = re.search(r"/f/[^/]+/(\d+)(?:/|$)", str(url))
    else:
        match = re.search(r"/f/([^/?#]+)(?:/|$)", str(url))
    return match.group(1) if match else ""


def vote_score(renv, submission_url: str, *, navigate: bool = True) -> dict:
    """Submission vote state: net score + whether the agent's vote is active.

    ``score=None`` when unparseable (honest UNKNOWN upstream). ``up_bid`` is the
    first up-control (its label is 'Upvote' or 'Retract upvote'); clicking it
    toggles the agent's own vote."""
    if navigate:
        _o, _r, _t, _tr, _i, view = renv.step(f"goto('{submission_url}')")
    else:
        view = renv._last_obs_view
    lines = (view.get("axtree_txt", "") or "").splitlines()
    score, direction, idx, down_idx = _submission_vote(lines)
    up_bid = None
    if idx >= 0:
        bm = re.search(r"\[(\d+)\]", lines[idx])
        up_bid = bm.group(1) if bm else None
    down_bid = None
    if down_idx >= 0:
        bm = re.search(r"\[(\d+)\]", lines[down_idx])
        down_bid = bm.group(1) if bm else None
    return {"score": score, "voted": direction == 1,
            "vote_direction": direction,
            "submission_id": _url_entity(submission_url, "submission"),
            "up_bid": up_bid, "down_bid": down_bid,
            "up_line": lines[idx] if idx >= 0 else "",
            "down_line": lines[down_idx] if down_idx >= 0 else ""}


def reddit_vote_canonical(renv, submission_url: str, *, navigate: bool = True
                          ) -> dict:
    """Agent-owned vote state; public score is telemetry, not equality truth."""
    raw = vote_score(renv, submission_url, navigate=navigate)
    return {
        "signal": "reddit_own_vote_state",
        "submission_id": raw["submission_id"],
        "vote_direction": raw["vote_direction"],
    }


def subscribe_signal(renv, forum_url: str, *, navigate: bool = True) -> dict:
    """Subscription state of a forum: subscribed iff an 'Unsubscribe' control is
    shown (Postmill swaps the button label with state)."""
    if navigate:
        _o, _r, _t, _tr, _i, view = renv.step(f"goto('{forum_url}')")
    else:
        view = renv._last_obs_view
    controls = []
    for element in extract_interactive_bids(view.get("axtree_txt", "")):
        low = element["line"].split("]", 1)[-1].strip().lower()
        name = re.match(r"button\s+'([^']*)'", low)
        # Live Postmill exposes the same entity-bound toggle as e.g.
        # ``Subscribe No subscribers`` / ``Unsubscribe 1 subscriber``.
        # Accept only that exact state+count grammar; controls such as
        # ``Subscribe via RSS`` remain excluded.
        if name and re.fullmatch(
                r"(?:subscribe|unsubscribe)(?:\s+(?:no|[\d,]+)\s+subscribers?)?",
                name.group(1).strip()):
            controls.append(element)
    ctrl = controls[0] if len(controls) == 1 else None
    line = (ctrl or {}).get("line", "")
    subscribed = "unsubscribe" in line.lower()
    return {"found": ctrl is not None, "ambiguous": len(controls) > 1,
            "subscribed": subscribed,
            "forum": _url_entity(forum_url, "forum"),
            "bid": (ctrl or {}).get("bid"), "line": line,
            "matching_control_count": len(controls)}


def reddit_subscribe_canonical(renv, forum_url: str, *, navigate: bool = True
                               ) -> dict:
    raw = subscribe_signal(renv, forum_url, navigate=navigate)
    return {
        "signal": "reddit_forum_subscription",
        "forum": raw["forum"],
        "found": raw["found"],
        "subscribed": raw["subscribed"],
    }


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
