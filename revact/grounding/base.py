"""Grounding core: probe protocol, registry, result schema, destructive gating.

A *probe* measures the reversibility of ONE action class by actually executing
the action in the live environment, running its undo controller, and comparing
a BACKEND signal (cart count, order set, wishlist count, ...) before/after.

Every probe declares:
  * ``destructive``: NON_DESTRUCTIVE (no lasting state change),
    SELF_RECOVERING (mutates state but the probe itself restores it, e.g.
    add-then-delete an address WE created), or DESTRUCTIVE (leaves a lasting
    change, e.g. a real order).
  * ``grounding``: which backend signal decides the label.
  * ``undo``: what the undo controller does.

Gating policy (enforced by ``run_probe``, not left to callers):
  DESTRUCTIVE probes execute their mutating step only when BOTH the explicit
  ``commit=True`` argument AND the environment opt-in
  ``REVACT_ALLOW_DESTRUCTIVE=1`` are present; otherwise they run in dry-run
  mode (navigate up to the mutating control, never click it) and return
  UNKNOWN with provenance.
"""
from __future__ import annotations

import os
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .. import config

DESTRUCTIVE_ENV_GATE = "REVACT_ALLOW_DESTRUCTIVE"

NON_DESTRUCTIVE = "non_destructive"
SELF_RECOVERING = "self_recovering"
DESTRUCTIVE = "destructive"

LABELS = ("REVERSIBLE", "PARTIALLY_RECOVERABLE", "IRREVERSIBLE",
          "NO_EFFECT", "UNKNOWN")


@dataclass
class ReversibilityResult:
    action_type: str
    label: str            # one of LABELS
    grounding: str        # backend signal used
    destructive: bool     # did this run mutate lasting state?
    evidence: dict = field(default_factory=dict)
    # provenance (defaults keep old JSONL rows loadable)
    probe_id: str = ""
    timestamp: str = ""
    commit_mode: bool = False
    site: str = ""
    probe_name: str = ""


def mk_result(action_type: str, label: str, grounding: str, destructive: bool,
              evidence: dict, commit_mode: bool = False, site: str = "",
              probe_name: str = "") -> ReversibilityResult:
    return ReversibilityResult(
        action_type=action_type, label=label, grounding=grounding,
        destructive=destructive, evidence=evidence,
        probe_id=f"{action_type}-{uuid.uuid4().hex[:8]}",
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        commit_mode=commit_mode, site=site, probe_name=probe_name,
    )


def destructive_allowed() -> bool:
    """Environment opt-in gate for destructive probes."""
    return os.environ.get(DESTRUCTIVE_ENV_GATE, "") == "1"


# --------------------------------------------------------------------------- #
# Probe context + spec + registry
# --------------------------------------------------------------------------- #
@dataclass
class ProbeContext:
    renv: Any                 # RevActEnv (or mock-wrapped equivalent)
    base: str                 # site base url (WA_SHOPPING / WA_REDDIT / ...)
    product_url: str = ""     # a valid product page (shopping probes)
    admin_base: str = ""      # WA_SHOPPING_ADMIN
    commit: bool = False      # explicit request to run the destructive step
    budget: int = 12          # undo-controller step budget k
    submission_url: str = ""  # a valid reddit submission page (reddit probes)
    forum_url: str = ""       # a valid reddit forum page (reddit probes)


@dataclass(frozen=True)
class ProbeSpec:
    name: str                 # registry key, e.g. "shopping.add_to_cart"
    site: str
    action_type: str
    destructive: str          # NON_DESTRUCTIVE | SELF_RECOVERING | DESTRUCTIVE
    grounding: str            # backend-signal description
    undo: str                 # undo-controller description
    fn: Callable[[ProbeContext], ReversibilityResult]
    expected_spectrum: str = ""  # documentation only, never used as a label


_REGISTRY: dict[str, ProbeSpec] = {}


def register(spec: ProbeSpec) -> ProbeSpec:
    if spec.name in _REGISTRY:
        raise ValueError(f"duplicate probe name {spec.name!r}")
    _REGISTRY[spec.name] = spec
    return spec


def get_probe(name: str) -> ProbeSpec:
    if name not in _REGISTRY:
        raise KeyError(f"unknown probe {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_probes(site: Optional[str] = None) -> list[ProbeSpec]:
    return [s for s in _REGISTRY.values() if site is None or s.site == site]


def run_probe(name: str, ctx: ProbeContext) -> ReversibilityResult:
    """Run one probe with gating + error containment.

    Never raises: an in-probe exception yields an UNKNOWN result carrying the
    traceback tail, so a batch run can proceed and the failure stays auditable.
    """
    spec = get_probe(name)
    gate_note = ""
    if spec.destructive == DESTRUCTIVE and ctx.commit and not destructive_allowed():
        ctx = ProbeContext(**{**asdict_ctx(ctx), "commit": False})
        gate_note = (f"commit requested but {DESTRUCTIVE_ENV_GATE}!=1 -> forced dry-run")
    try:
        res = spec.fn(ctx)
    except Exception:
        res = mk_result(spec.action_type, "UNKNOWN", spec.grounding,
                        destructive=False,
                        evidence={"error": traceback.format_exc(limit=3)[-800:]})
    res.site = spec.site
    res.probe_name = spec.name
    if gate_note:
        res.evidence.setdefault("gate_note", gate_note)
    return res


def asdict_ctx(ctx: ProbeContext) -> dict:
    return {"renv": ctx.renv, "base": ctx.base, "product_url": ctx.product_url,
            "admin_base": ctx.admin_base, "commit": ctx.commit,
            "budget": ctx.budget, "submission_url": ctx.submission_url,
            "forum_url": ctx.forum_url}


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_results(results: list[ReversibilityResult],
                 out_dir: Optional[Path] = None) -> Path:
    import json

    base = Path(out_dir) if out_dir else config.DATA_ROOT
    path = base / "grounded" / "reversibility.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    _append_manifest(base / "grounded", results)
    return path


def _append_manifest(grounded_dir: Path, results: list[ReversibilityResult]) -> None:
    """One manifest line per probe run: enough to audit where labels came from."""
    import json

    path = grounded_dir / "MANIFEST.jsonl"
    with path.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({
                "probe_id": r.probe_id, "probe_name": r.probe_name,
                "site": r.site, "action_type": r.action_type, "label": r.label,
                "timestamp": r.timestamp, "commit_mode": r.commit_mode,
                "controller_version": config.CONTROLLER_VERSION,
            }, ensure_ascii=False) + "\n")


def load_reversibility(path: Path) -> dict:
    """action_type -> grounded label, dry-run-safe (latest non-UNKNOWN wins)."""
    return {at: r["label"] for at, r in load_reversibility_details(path).items()}


def load_reversibility_details(path: Path) -> dict:
    """action_type -> latest-non-UNKNOWN grounded record with the undo
    evidence (undo_steps / undo_actions / residual_diff) that the P2 sample
    format consumes when constructing <rev_check>/<undo>.

    Returns {action_type: {label, grounding, undo_steps, undo_actions,
                           residual_diff, probe_id}}.
    """
    import json

    rows_by_type: dict[str, list] = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            rows_by_type.setdefault(r["action_type"], []).append(r)
    out = {}
    for at, rows in rows_by_type.items():
        grounded = [r for r in rows if r.get("label") != "UNKNOWN"]
        pick = grounded[-1] if grounded else rows[-1]
        ev = pick.get("evidence") or {}
        out[at] = {
            "label": pick["label"],
            "grounding": pick.get("grounding", ""),
            "undo_steps": ev.get("undo_steps"),
            "undo_actions": ev.get("undo_actions") or [],
            "residual_diff": ev.get("residual_diff"),
            "probe_id": pick.get("probe_id", ""),
        }
    return out
