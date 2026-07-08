"""List WebArena shopping-only task ids.

`list_shopping_task_ids` reads the WebArena raw config (`webarena/test.raw.json`)
and keeps tasks whose sites == ["shopping"]. Results are cached to
config.TASK_LIST_PATH so scripts that run outside the `agentlab` env (where
`webarena` is installed) can still read the list.
"""
from __future__ import annotations

import json

from .. import config


def _load_from_webarena() -> list[str]:
    import importlib.resources  # lazy
    import webarena  # provided by the browsergym webarena extra

    raw = importlib.resources.files(webarena).joinpath("test.raw.json").read_text()
    data = json.loads(raw)
    ids = [f"webarena.{d['task_id']}" for d in data if d.get("sites") == ["shopping"]]
    return ids


def list_shopping_task_ids(use_cache: bool = True, write_cache: bool = True) -> list[str]:
    if use_cache and config.TASK_LIST_PATH.exists():
        return json.loads(config.TASK_LIST_PATH.read_text())
    ids = _load_from_webarena()
    if write_cache:
        config.TASK_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        config.TASK_LIST_PATH.write_text(json.dumps(ids, ensure_ascii=False, indent=2))
    return ids


# --------------------------------------------------------------------------- #
# Pilot task selection: bucket shopping tasks by which pilot state they reach /
# how tractable they are, so we don't waste budget on hard, off-target review-QA.
# --------------------------------------------------------------------------- #
def bucket_intent(intent: str) -> str:
    t = (intent or "").lower()
    # Structurally infeasible in our single-tab setup: WebArena "open tabs"
    # tasks need pre-opened browser tabs; no agent/policy can do them here.
    if "open tab" in t or "across the tabs" in t or "opened tabs" in t:
        return "infeasible/multi-tab"
    if any(k in t for k in ["add to cart", "add to the cart", " to the shopping cart"]):
        return "cart/add"          # -> cart + add_to_cart states
    if any(k in t for k in ["address", "shipping address", "my account"]):
        return "account/address"   # -> address book (delete_address affordance)
    if any(k in t for k in ["checkout", "place order", "purchase", "cancel"]):
        return "order/checkout"    # -> order / checkout pages
    if any(k in t for k in ["review", "reviewer", "rating", "comment"]):
        return "review-QA"         # hard for a11y-only agents; skip in pilot
    return "other-info"            # order counts, statuses, product attributes


# tractable + on-target first; review-QA excluded by default (count 0)
DEFAULT_PER_BUCKET = {
    "cart/add": 10,
    "account/address": 5,
    "order/checkout": 8,
    "other-info": 7,
    "review-QA": 0,
}


def select_pilot_tasks(per_bucket: dict | None = None) -> dict:
    """Return {bucket: [(task_id, intent), ...]} sampled per DEFAULT_PER_BUCKET."""
    import importlib.resources
    import webarena

    per_bucket = per_bucket or DEFAULT_PER_BUCKET
    data = json.loads(
        importlib.resources.files(webarena).joinpath("test.raw.json").read_text()
    )
    shop = [d for d in data if d.get("sites") == ["shopping"]]
    buckets: dict[str, list] = {}
    for d in shop:
        b = bucket_intent(d["intent"])
        buckets.setdefault(b, []).append((f"webarena.{d['task_id']}", d["intent"]))
    return {b: items[: per_bucket.get(b, 0)] for b, items in buckets.items()}
