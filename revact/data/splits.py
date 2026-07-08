"""Split assembled samples into train/test by PRODUCT (held-out products).

Splitting by product (not by sample) tests whether the model generalizes the
reversibility->decision mapping to UNSEEN product pages rather than memorizing.
NOTE (honest limitation, kept from the audit): checkout pages are largely
product-independent, so the product split is a WEAK generalization test; the
real test is the cross-action-class split, which becomes available once
multiple grounded action classes exist (see meta.action_type).
"""
from __future__ import annotations

import collections
import json
from pathlib import Path


def parse_sid(sid: str):
    p = sid.split("__")
    action = p[0]
    slug = p[1] if len(p) >= 3 else "_base"
    variant = p[-1]
    return action, slug, variant


def _site_of(row: dict) -> str:
    meta = row.get("meta") or {}
    if meta.get("site"):
        return meta["site"]
    at = meta.get("action_type", "")
    if at.startswith("reddit_"):
        return "reddit"
    if at.startswith("admin_"):
        return "shopping_admin"
    return "shopping"


def build_splits(sft_path: Path, dpo_path: Path, out_dir: Path,
                 holdout_frac: float = 0.25) -> dict:
    sft = [json.loads(ln) for ln in sft_path.open()]
    dpo = [json.loads(ln) for ln in dpo_path.open()] if dpo_path.exists() else []

    prods = collections.defaultdict(set)
    for r in sft:
        a, slug, _ = parse_sid(r["sample_id"])
        prods[a].add(slug)
    test_slugs = set()
    for a, slugs in prods.items():
        s = sorted(slugs)
        k = max(1, int(len(s) * holdout_frac))
        test_slugs |= set(s[-k:])

    def is_test(sid):
        return parse_sid(sid)[1] in test_slugs

    sft_train = [r for r in sft if not is_test(r["sample_id"])]
    sft_test = [r for r in sft if is_test(r["sample_id"])]
    dpo_train = [r for r in dpo if not is_test(r["pair_id"])]

    out_files = {"sft_train": sft_train, "sft_test": sft_test,
                 "dpo_train": dpo_train}

    # Cross-site generalization split (available once >=2 sites are grounded):
    # hold out an ENTIRE site's samples so the test measures transfer of the
    # reversibility->decision mapping across sites, not product memorization.
    sites = sorted({_site_of(r) for r in sft})
    cross_site = {}
    if len(sites) >= 2:
        held_site = sites[-1]
        cross_site = {
            "held_site": held_site,
            "sites": sites,
            "n_test": sum(_site_of(r) == held_site for r in sft),
        }
        out_files["sft_train_cross_site"] = [r for r in sft
                                             if _site_of(r) != held_site]
        out_files["sft_test_cross_site"] = [r for r in sft
                                            if _site_of(r) == held_site]

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in out_files.items():
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dist = {
        split: dict(collections.Counter(
            (r["meta"]["action_type"], r["meta"]["decision"]) for r in rows))
        for split, rows in [("train", sft_train), ("test", sft_test)]
    }
    by_site = dict(collections.Counter(_site_of(r) for r in sft))
    return {"test_slugs": sorted(test_slugs), "n_train": len(sft_train),
            "n_test": len(sft_test), "n_dpo_train": len(dpo_train),
            "distribution": dist, "by_site": by_site,
            "cross_site": cross_site, "out_dir": str(out_dir)}
