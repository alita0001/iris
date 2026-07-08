"""Human-review overlays: append-only JSONL, never mutating pipeline files.

One file per target kind under ``data/annotations/``. Every row is a full
audit record; the *effective* annotation for a target is the chronological
merge of its rows (later fields win). Export (see ``export.py``) is the only
consumer that applies overlays to training data — the generation pipeline
itself never reads them, so grounded labels stay behavioral.

Target kinds and their natural keys:
  trajectory    trajectory_id       (S2 rollouts: select/exclude for next stage)
  key_state     state_id            (S2 key states: confirm/reject/type tags)
  state         name                (S3/scale reached states)
  constraint    name__variant       (edited goal text / constraint type)
  candidate     name__index|custom  (added/edited/removed candidate actions)
  grounded      probe_id            (undo-label human verification/override)
  sample        sample_id           (accepted / rejected / needs-review + edits)
  distill       sample_id           (teacher prose review)
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .. import config

KINDS = ("trajectory", "key_state", "state", "constraint", "candidate",
         "grounded", "sample", "distill")

# review_status values shared across kinds
STATUSES = ("accepted", "rejected", "needs-review", "confirmed")


def _dir(root: Path | None = None) -> Path:
    return (root or config.DATA_ROOT) / "annotations"


def _path(kind: str, root: Path | None = None) -> Path:
    if kind not in KINDS:
        raise ValueError(f"unknown annotation kind {kind!r}; known: {KINDS}")
    return _dir(root) / f"{kind}.jsonl"


def add(kind: str, target_id: str, payload: dict,
        author: str = "workbench", root: Path | None = None) -> dict:
    """Append one annotation row. `payload` carries the edited fields, e.g.
    {"review_status": "rejected", "note": "..."} or
    {"reversibility_override": "PARTIALLY_RECOVERABLE", "confidence": 0.8}."""
    if not target_id:
        raise ValueError("target_id required")
    row = {
        "ann_id": f"{kind}-{int(time.time() * 1000)}",
        "kind": kind, "target_id": target_id,
        "payload": payload, "author": author,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    p = _path(kind, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def history(kind: str, root: Path | None = None) -> list[dict]:
    p = _path(kind, root)
    if not p.exists():
        return []
    out = []
    for ln in p.open(encoding="utf-8"):
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return out


def effective(kind: str, root: Path | None = None) -> dict[str, dict]:
    """target_id -> chronologically merged payload (+ _last_ts, _n_rows)."""
    merged: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for row in history(kind, root):
        tid = row["target_id"]
        cur = merged.setdefault(tid, {})
        cur.update(row.get("payload") or {})
        cur["_last_ts"] = row.get("timestamp", "")
        counts[tid] = counts.get(tid, 0) + 1
    for tid, cur in merged.items():
        cur["_n_rows"] = counts[tid]
    return merged


def all_effective(root: Path | None = None) -> dict[str, dict[str, dict]]:
    return {kind: effective(kind, root) for kind in KINDS}
