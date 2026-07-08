"""S1: state fingerprint + distance + restore check.

A StateFingerprint is a compact, JSON-serializable summary of a web state used
for two purposes:
  1. replay-to-state verification (did replaying the prefix land us on the same
     state we recorded?);
  2. later (S5) reversibility grounding (did undoing an action restore the
     pre-action state?).

Pure stdlib. Operates on an "obs view" dict: {url, title, axtree_txt,
backend_state?, form_values?} produced by obs_utils.to_obs_view().

Design notes
------------
* Accessibility-tree element ids (``[123]``) are volatile across reloads/replays,
  so they are stripped before hashing. Otherwise a faithful replay would look
  "different" purely because bids were renumbered.
* When a backend_state (DB/API view) is available it is the authoritative
  channel for restore checks; UI channels are advisory.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from ..config import DEFAULT_TOL, FingerprintTolerance

_BID_RE = re.compile(r"\[\d+\]")
_MULTISPACE_RE = re.compile(r"[ \t]+")
_ORDER_RE = re.compile(r"order\s*#?\s*(\d+)", re.IGNORECASE)
_SHINGLE_N = 4


def _normalize_axtree(text: str) -> str:
    """Strip volatile tokens (bids, extra whitespace) for stable hashing."""
    if not text:
        return ""
    text = _BID_RE.sub("", text)
    lines = []
    for ln in text.splitlines():
        ln = _MULTISPACE_RE.sub(" ", ln).strip()
        if ln:
            lines.append(ln)
    return "\n".join(lines)


def _shingles(text: str, n: int = _SHINGLE_N) -> frozenset[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    if len(toks) < n:
        return frozenset(toks)
    return frozenset(" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1))


def _extract_entities(text: str, backend_state: dict | None) -> frozenset[str]:
    ents: set[str] = set()
    for m in _ORDER_RE.finditer(text or ""):
        ents.add(f"order:{m.group(1)}")
    if backend_state:
        # Deterministic, structured entities from the backend view.
        for order in backend_state.get("orders", []) or []:
            ents.add(f"order:{order}")
        ents.add(f"cart_size:{len(backend_state.get('cart', []) or [])}")
        ents.add(f"n_addresses:{len(backend_state.get('addresses', []) or [])}")
    return frozenset(ents)


def _jaccard_distance(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return 1.0 - inter / union


@dataclass(frozen=True)
class StateFingerprint:
    url: str
    title: str
    axtree_hash: str
    text_shingles: frozenset[str] = field(repr=False)
    form_values: dict[str, str] = field(default_factory=dict)
    key_entities: frozenset[str] = frozenset()
    backend_state: dict[str, Any] | None = None

    # -- serialization ----------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "axtree_hash": self.axtree_hash,
            "text_shingles": sorted(self.text_shingles),
            "form_values": self.form_values,
            "key_entities": sorted(self.key_entities),
            "backend_state": self.backend_state,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StateFingerprint":
        return cls(
            url=d.get("url", ""),
            title=d.get("title", ""),
            axtree_hash=d.get("axtree_hash", ""),
            text_shingles=frozenset(d.get("text_shingles", [])),
            form_values=dict(d.get("form_values", {})),
            key_entities=frozenset(d.get("key_entities", [])),
            backend_state=d.get("backend_state"),
        )


def fingerprint(obs_view: dict) -> StateFingerprint:
    axtree_txt = obs_view.get("axtree_txt", "") or ""
    normalized = _normalize_axtree(axtree_txt)
    backend_state = obs_view.get("backend_state")
    return StateFingerprint(
        url=obs_view.get("url", "") or "",
        title=obs_view.get("title", "") or "",
        axtree_hash=hashlib.sha1(normalized.encode("utf-8")).hexdigest(),
        text_shingles=_shingles(normalized),
        form_values=dict(obs_view.get("form_values", {}) or {}),
        key_entities=_extract_entities(axtree_txt, backend_state),
        backend_state=backend_state,
    )


def state_distance(a: StateFingerprint, b: StateFingerprint) -> dict:
    return {
        "url_changed": a.url != b.url,
        "title_changed": a.title != b.title,
        "axtree_changed": a.axtree_hash != b.axtree_hash,
        "text_jaccard_dist": round(_jaccard_distance(a.text_shingles, b.text_shingles), 4),
        "form_changed": a.form_values != b.form_values,
        "entities_added": sorted(b.key_entities - a.key_entities),
        "entities_removed": sorted(a.key_entities - b.key_entities),
        "backend_changed": (
            None
            if a.backend_state is None or b.backend_state is None
            else a.backend_state != b.backend_state
        ),
    }


def is_restored(
    pre: StateFingerprint,
    post: StateFingerprint,
    tol: FingerprintTolerance = DEFAULT_TOL,
) -> bool:
    """Whether ``post`` is (approximately) the same state as ``pre``.

    Backend channel is authoritative when present; otherwise fall back to UI
    channels + a small text-drift tolerance.
    """
    if pre.backend_state is not None and post.backend_state is not None:
        return pre.backend_state == post.backend_state
    d = state_distance(pre, post)
    return (
        not d["url_changed"]
        and not d["axtree_changed"]
        and not d["form_changed"]
        and not d["entities_added"]
        and not d["entities_removed"]
        and d["text_jaccard_dist"] <= tol.text_jaccard
    )
