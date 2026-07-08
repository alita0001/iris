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

from ..config import MAX_AXTREE_CHARS_SNAPSHOT

_BID_LINE_RE = re.compile(r"\[(\d+)\]\s*(.*)")


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


def prune_axtree_txt(text: str, max_chars: int = MAX_AXTREE_CHARS_SNAPSHOT) -> str:
    """Cap length while preserving reading order (truncate the tail).

    Original order matters: static text (review bodies, descriptions) is
    interleaved with interactive elements, so reordering would break context.
    """
    if not text or len(text) <= max_chars:
        return text or ""
    kept, size = [], 0
    for ln in text.splitlines():
        if size + len(ln) + 1 > max_chars:
            break
        kept.append(ln)
        size += len(ln) + 1
    kept.append("... (axtree truncated)")
    return "\n".join(kept)


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
