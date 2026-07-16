"""S1/S8-prep: normalize BrowserGym / mock observations to a common "obs view".

`to_obs_view` accepts either a raw BrowserGym observation (which carries an
``axtree_object``) or a mock observation (which already carries ``axtree_txt``)
and returns a uniform dict:

    {url, title, axtree_txt, backend_state?, form_values?}

Only `to_obs_view`'s axtree branch touches browsergym, and it imports it lazily,
so this module stays importable under plain stdlib.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from ..config import MAX_AXTREE_CHARS_SNAPSHOT

_BID_LINE_RE = re.compile(r"\[([^\]\s]+)\]\s*(.*)")
_CLICK_BID_RE = re.compile(r"^\s*click\(\s*(['\"])([^'\"]+)\1\s*\)\s*$")
_TRUNCATION_MARKER = "... (axtree truncated)"


def _axtree_from_browsergym(obs: dict) -> str:
    """Flatten a BrowserGym AXTree object to the ``[bid] role "name"`` text."""
    ax = obs.get("axtree_object")
    if ax is None:
        return obs.get("axtree_txt", "") or ""
    try:
        from browsergym.utils.obs import flatten_axtree_to_str  # lazy

        # filter_with_bid_only=True dropped StaticText (review bodies, product
        # descriptions) that read/QA tasks need. Keep static text so the policy
        # can actually read page content.
        return flatten_axtree_to_str(
            ax,
            extra_properties=obs.get("extra_element_properties"),
            filter_with_bid_only=False,
        )
    except Exception:
        # Never let serialization crash collection; degrade gracefully.
        return obs.get("axtree_txt", "") or ""


def to_obs_view(obs: dict) -> dict:
    """Uniform view over real/mock observations."""
    if not isinstance(obs, dict):
        return {"url": "", "title": "", "axtree_txt": "", "backend_state": None}

    url = obs.get("url")
    if not url:
        pages = obs.get("open_pages_urls") or []
        idx = obs.get("active_page_index", 0) or 0
        try:
            url = pages[int(idx)]
        except (IndexError, ValueError, TypeError):
            url = pages[0] if pages else ""

    if obs.get("axtree_txt") is not None:
        axtree_txt = obs.get("axtree_txt") or ""
    else:
        axtree_txt = _axtree_from_browsergym(obs)

    return {
        "url": url or "",
        "title": obs.get("title", "") or "",
        "axtree_txt": axtree_txt,
        "backend_state": obs.get("backend_state"),  # only mock provides this
        "form_values": obs.get("form_values", {}) or {},
    }


def _indent(line: str) -> int:
    """Indentation depth in the flattened AXTree (tabs count as four spaces)."""
    prefix = line[:len(line) - len(line.lstrip(" \t"))]
    return sum(4 if c == "\t" else 1 for c in prefix)


def _render_selected(lines: list[str], selected: set[int]) -> str:
    """Render selected source lines in order, marking every omitted interval."""
    if not selected:
        return _TRUNCATION_MARKER
    out: list[str] = []
    previous = -1
    for i in sorted(selected):
        if i > previous + 1:
            out.append(_TRUNCATION_MARKER)
        out.append(lines[i])
        previous = i
    if previous < len(lines) - 1:
        out.append(_TRUNCATION_MARKER)
    return "\n".join(out)


def _anchor_line_indices(lines: list[str], anchor_bids: set[str],
                         anchor_terms: tuple[str, ...]) -> list[int]:
    out: list[int] = []
    for i, line in enumerate(lines):
        m = _BID_LINE_RE.search(line)
        bid_hit = bool(m and m.group(1) in anchor_bids)
        term_hit = bool(anchor_terms and any(t in line.lower() for t in anchor_terms))
        if bid_hit or term_hit:
            out.append(i)
    return out


def prune_axtree_txt(
    text: str,
    max_chars: int = MAX_AXTREE_CHARS_SNAPSHOT,
    *,
    anchor_bids: Iterable[str] | None = None,
    anchor_terms: Iterable[str] | None = None,
) -> str:
    """Cap an AXTree while retaining supervised controls and their ancestry.

    With no anchors this preserves the historical prefix-truncation behaviour.
    When ``anchor_bids`` (preferred) or ``anchor_terms`` are supplied, matching
    source lines are *mandatory*: the function keeps them, their accessibility
    ancestors, a small amount of child context, and fills the remaining budget
    from the original prefix.  All retained lines remain in source order and
    every gap is explicit.  This prevents dataset assembly from supervising a
    ``click('bid')`` whose ``[bid]`` was silently removed by character clipping.

    If an anchor itself is absent from ``text`` this function cannot invent it;
    callers supervising an action must additionally call
    :func:`require_action_bid_visible` as a hard gate.
    """
    if max_chars <= 0:
        return ""
    if not text or len(text) <= max_chars:
        return text or ""
    bids = {str(x) for x in (anchor_bids or []) if str(x)}
    terms = tuple(str(x).lower() for x in (anchor_terms or []) if str(x))
    if bids or terms:
        lines = text.splitlines()
        anchors = _anchor_line_indices(lines, bids, terms)
        if anchors:
            selected: set[int] = set(anchors)

            # Add the ancestor chain nearest-first.  Ancestors materially help
            # disambiguate repeated buttons (e.g. which product an Add button
            # belongs to), but never displace the actual supervised control.
            context: list[int] = []
            for anchor in anchors:
                want_indent = _indent(lines[anchor])
                for j in range(anchor - 1, -1, -1):
                    level = _indent(lines[j])
                    if level < want_indent:
                        context.append(j)
                        want_indent = level
                        if level == 0:
                            break
                # Keep up to two direct/descendant lines following the control.
                for j in range(anchor + 1, min(len(lines), anchor + 3)):
                    if _indent(lines[j]) > _indent(lines[anchor]):
                        context.append(j)
                    else:
                        break

            for i in context:
                trial = set(selected)
                trial.add(i)
                if len(_render_selected(lines, trial)) <= max_chars:
                    selected = trial

            # Use the remaining budget for reading-order context.  Skipping a
            # very long optional line is preferable to dropping the anchor.
            for i in range(len(lines)):
                if i in selected:
                    continue
                trial = set(selected)
                trial.add(i)
                if len(_render_selected(lines, trial)) <= max_chars:
                    selected = trial
            rendered = _render_selected(lines, selected)
            if len(rendered) <= max_chars:
                return rendered

            # Pathological case: a single source line exceeds the whole budget.
            # Preserve its [bid] prefix so the action remains grounded.
            anchor = lines[anchors[0]][:max(0, max_chars - len("\n" + _TRUNCATION_MARKER))]
            return (anchor + "\n" + _TRUNCATION_MARKER)[:max_chars]

    # No supplied anchor was found: retain backward-compatible reading-order
    # truncation.  A supervising caller will reject the sample via the hard gate.
    kept, size = [], 0
    for ln in text.splitlines():
        if size + len(ln) + 1 > max_chars:
            break
        kept.append(ln)
        size += len(ln) + 1
    kept.append(_TRUNCATION_MARKER)
    rendered = "\n".join(kept)
    return rendered[:max_chars]


def action_bid(action: str) -> str | None:
    """Return a BrowserGym bid for an exact click action, otherwise ``None``."""
    m = _CLICK_BID_RE.match(action or "")
    return m.group(2) if m else None


def bid_is_visible(axtree_txt: str, bid: str) -> bool:
    """Exact interactive-control membership, not a prose/static ``[bid]`` hit."""
    # Lazy import avoids making the environment helper depend on candidate
    # generation at module import time while sharing the same AX role parser.
    from ..data.candidates import interactive_bids
    return str(bid) in interactive_bids(axtree_txt)


def require_action_bid_visible(action: str, axtree_txt: str, *,
                               sample_id: str = "") -> None:
    """Hard dataset gate for supervised click actions.

    Non-click actions have no accessibility bid and therefore pass.  A click on
    an absent bid raises ``ValueError``; assembly must skip/quarantine the row,
    never train the model to hallucinate an inaccessible element identifier.
    """
    bid = action_bid(action)
    if bid is not None and not bid_is_visible(axtree_txt, bid):
        where = f" for {sample_id}" if sample_id else ""
        raise ValueError(f"supervised click bid [{bid}] is absent from observation{where}")


def extract_interactive_bids(axtree_txt: str) -> list[dict]:
    """Return [{bid, line}] for every element carrying a [bid] (for S4 anchors)."""
    out = []
    for ln in (axtree_txt or "").splitlines():
        m = _BID_LINE_RE.search(ln)
        if m:
            out.append({"bid": m.group(1), "line": ln.strip()})
    return out


def find_bid_by_text(obs_view: dict, substrings: list[str]) -> dict | None:
    """First interactive element whose line contains any substring (any role).

    Substring (not word-boundary) matching by design: this is a NAVIGATION
    helper for reaching states. Grounding decisions use the stricter
    role-filtered, word-boundary `grounding.signals.find_action_by_text`.
    """
    subs = [s.lower() for s in substrings]
    for el in extract_interactive_bids(obs_view.get("axtree_txt", "")):
        line = el["line"].lower()
        if any(s in line for s in subs):
            return el
    return None


def one_line_summary(obs_view: dict, max_len: int = 120) -> str:
    """Compact one-line description of a state for history compaction (S8)."""
    url = obs_view.get("url", "")
    head = (obs_view.get("axtree_txt", "") or "").strip().splitlines()
    head_txt = head[0].strip() if head else ""
    s = f"{url} | {head_txt}"
    return s[:max_len]


def compact_history(history: list[dict], k: int = 5) -> list[dict]:
    """Keep the last k (action, summary) pairs for prompt assembly."""
    return history[-k:]


# --------------------------------------------------------------------------- #
# P2: observed-delta history entries (action + key delta + change flag)
# --------------------------------------------------------------------------- #
_ROOT_RE = re.compile(r"RootWebArea '([^']+)'")


def _page_label(view: dict) -> str:
    title = (view.get("title") or "").strip()
    if title:
        return title
    m = _ROOT_RE.search(view.get("axtree_txt", "") or "")
    return m.group(1) if m else ""


def _short_url(url: str, max_len: int = 60) -> str:
    from urllib.parse import urlparse

    p = urlparse(url or "")
    tail = (p.path or "/") + (f"?{p.query}" if p.query else "")
    return tail if len(tail) <= max_len else "…" + tail[-max_len:]


def _entity_signals(d: dict) -> list[str]:
    """Human-readable backend-signal deltas from a state_distance dict."""
    added = {e.split(":", 1)[0]: e.split(":", 1)[1] for e in d["entities_added"]}
    removed = {e.split(":", 1)[0]: e.split(":", 1)[1] for e in d["entities_removed"]}
    out: list[str] = []
    for key, label in (("cart_size", "cart items"), ("n_addresses", "addresses")):
        if key in added or key in removed:
            out.append(f"{label} {removed.get(key, '?')} -> {added.get(key, '?')}")
    new_orders = sorted(e.split(":", 1)[1] for e in d["entities_added"]
                        if e.startswith("order:"))
    if new_orders:
        out.append("new order #" + ", #".join(new_orders))
    gone_orders = sorted(e.split(":", 1)[1] for e in d["entities_removed"]
                         if e.startswith("order:"))
    if gone_orders:
        out.append("order gone #" + ", #".join(gone_orders))
    return out


def obs_delta(prev_view: dict, view: dict) -> dict:
    """{'flag', 'delta'} describing what the last action observably changed.

    Flags (all computable at deployment time — no grounded labels involved):
      state-change  a persistent backend/probe signal moved;
      nav           URL changed, no state signal moved;
      update        same URL, page/form/UI content changed without a persistent
                    signal (menu opened, tab switched, form edited);
      no-effect     nothing observable changed — the caller's cue to try a
                    DIFFERENT action instead of repeating this one.
    """
    from .fingerprint import fingerprint, state_distance

    d = state_distance(fingerprint(prev_view or {}), fingerprint(view or {}))
    # ``key_entities`` also contains order-looking text scraped from the
    # current AXTree.  Moving from an order list to an existing order detail
    # therefore changes that set even though no order was created.  It is
    # evidence about page visibility, not persistence.  Only a structured
    # backend channel or an explicitly supplied probe signal may justify the
    # stronger ``state-change`` history flag.
    backend_observed = (prev_view.get("backend_state") is not None and
                        view.get("backend_state") is not None)
    persistent_before = prev_view.get("persistent_signal")
    persistent_after = view.get("persistent_signal")
    persistent_observed = (persistent_before is not None and
                           persistent_after is not None)
    persistent_changed = (persistent_observed and
                          persistent_before != persistent_after)
    signals = _entity_signals(d) if backend_observed else []
    label = _page_label(view)
    if d["backend_changed"] is True or persistent_changed:
        if not signals:
            signals = ["persistent site signal changed"]
        delta = "; ".join(signals)
        if d["url_changed"] and label:
            delta += f" (now on: {label})"
        return {"flag": "state-change", "delta": delta}
    if d["url_changed"]:
        where = label or "new page"
        return {"flag": "nav", "delta": f"{where} ({_short_url(view.get('url', ''))})"}
    if d["form_changed"]:
        return {"flag": "update", "delta": "form values changed (same page)"}
    if d["axtree_changed"] or d["title_changed"] or \
            d["entities_added"] or d["entities_removed"]:
        return {"flag": "update",
                "delta": "page content changed (same URL)"}
    return {"flag": "no-effect", "delta": "no visible change"}


def history_entry(action: str, prev_view: dict, view: dict) -> dict:
    """One P2 history entry: {'action', 'delta', 'flag'} (see prompts.py)."""
    return {"action": action, **obs_delta(prev_view, view)}
