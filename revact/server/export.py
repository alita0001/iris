"""Dataset export: apply annotation overlays to the assembled splits and write
a versioned, auditable release under ``outputs/workbench/exports/``.

Overlay semantics (deliberately conservative):
  * sample review_status == rejected      -> excluded (listed in excluded.jsonl)
  * sample review_status == needs-review  -> excluded unless include_needs_review
  * grounded human override that CONTRADICTS a sample's pinned label
    -> excluded as label_conflict (we never silently rewrite a behaviorally
       grounded label; resolve by re-probing or re-assembling instead)
  * meta gains review provenance fields; message content is exported verbatim.

Original files under ``data/`` are never touched.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from datetime import datetime, timezone

from .. import config
from . import annotations
from .datasets import DataStore, _jsonl

EXPORTS_DIR = config.OUTPUTS_DIR / "workbench" / "exports"


def _val_split(sample_id: str, val_frac: float) -> bool:
    h = int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest(), 16) % 1000
    return h < int(val_frac * 1000)


def export_dataset(store: DataStore | None = None, params: dict | None = None) -> dict:
    store = store or DataStore()
    params = params or {}
    name = re.sub(r"[^A-Za-z0-9_.-]", "-", str(params.get("name", "release")))[:40]
    val_frac = min(0.5, max(0.0, float(params.get("val_frac", 0.15))))
    include_needs_review = bool(params.get("include_needs_review", False))
    prefer_distilled = bool(params.get("prefer_distilled", True))

    splits_dir = store.root / "train" / "splits"
    sft_train = _jsonl(splits_dir / "sft_train.jsonl")
    sft_test = _jsonl(splits_dir / "sft_test.jsonl")
    dpo_train = _jsonl(splits_dir / "dpo_train.jsonl")
    if not sft_train and not sft_test:
        return {"ok": False, "note": "没有 splits 产物：先运行 assemble + split"}

    sample_ann = annotations.effective("sample", store.root)
    grounded_ann = annotations.effective("grounded", store.root)
    distill_ann = annotations.effective("distill", store.root)
    distilled_by_id = {r.get("sample_id"): r for r in _jsonl(
        store.root / "train" / "sft" / "revact_sft_distilled.jsonl")}
    # map action_type -> human override label (via probe_id rows)
    probe_by_id = {g["probe_id"]: g for g in store.grounded_runs() if g["probe_id"]}
    override_by_type: dict[str, str] = {}
    for pid, a in grounded_ann.items():
        ov = a.get("reversibility_override")
        g = probe_by_id.get(pid)
        if ov and g:
            override_by_type[g["action_type"]] = ov

    excluded: list[dict] = []

    def keep(row: dict, sid: str, allow_distill: bool = True) -> dict | None:
        a = sample_ann.get(sid) or {}
        status = a.get("review_status", "")
        if status == "rejected":
            excluded.append({"id": sid, "reason": "human-rejected",
                             "note": a.get("note", "")})
            return None
        if status == "needs-review" and not include_needs_review:
            excluded.append({"id": sid, "reason": "needs-review"})
            return None
        meta = dict(row.get("meta") or {})
        at = meta.get("action_type", "")
        ov = override_by_type.get(at)
        if ov and meta.get("reversibility") and ov != meta["reversibility"]:
            excluded.append({"id": sid, "reason": "label_conflict",
                             "note": f"human override {ov} != pinned {meta['reversibility']}"})
            return None
        out = dict(row)
        if allow_distill and prefer_distilled and sid in distilled_by_id \
                and (distill_ann.get(sid) or {}).get("review_status") != "rejected":
            out = dict(distilled_by_id[sid])
        meta = dict(out.get("meta") or {})
        meta["review_status"] = status or "unreviewed"
        if a.get("note"):
            meta["review_note"] = a["note"]
        if ov:
            meta["human_verified_label"] = ov
        out["meta"] = meta
        return out

    train, val = [], []
    for r in sft_train:
        k = keep(r, r.get("sample_id", ""))
        if k is None:
            continue
        (val if _val_split(r.get("sample_id", ""), val_frac) else train).append(k)
    test = [k for r in sft_test if (k := keep(r, r.get("sample_id", "")))]
    dpo = []
    for r in dpo_train:
        sid = r.get("pair_id", "").rsplit("__", 1)[0]
        k = keep(dict(r), sid, allow_distill=False)
        if k is not None:
            dpo.append(k)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = EXPORTS_DIR / f"{stamp}__{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(fname: str, rows: list[dict]) -> None:
        with (out_dir / fname).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl("sft_train.jsonl", train)
    write_jsonl("sft_val.jsonl", val)
    write_jsonl("sft_test.jsonl", test)
    write_jsonl("dpo_train.jsonl", dpo)
    write_jsonl("excluded.jsonl", excluded)

    csv_buf = io.StringIO()
    w = csv.writer(csv_buf)
    w.writerow(["sample_id", "split", "action_type", "variant", "decision",
                "reversibility", "constraint_style", "review_status", "prose_source"])
    for split_name, rows in [("train", train), ("val", val), ("test", test)]:
        for r in rows:
            m = r.get("meta") or {}
            w.writerow([r.get("sample_id"), split_name, m.get("action_type"),
                        m.get("variant"), m.get("decision"), m.get("reversibility"),
                        m.get("constraint_style"), m.get("review_status"),
                        m.get("prose_source", "template")])
    (out_dir / "samples.csv").write_text(csv_buf.getvalue(), encoding="utf-8")

    from .quality import compute_quality
    quality = compute_quality(store)
    (out_dir / "stats.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=1), encoding="utf-8")
    # Provenance: snapshot the EFFECTIVE prompt set (registry + overrides) so
    # the release is attributable to the exact prompts that produced it.
    from .. import prompts as _prompts
    (out_dir / "prompts.json").write_text(
        json.dumps({"fingerprint": _prompts.fingerprint(),
                    "prompts": _prompts.effective()},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "dataset_card.md").write_text(
        _dataset_card(name, stamp, train, val, test, dpo, excluded, quality, store),
        encoding="utf-8")

    try:
        rel = str(out_dir.relative_to(config.PROJECT_ROOT))
    except ValueError:
        rel = str(out_dir)
    return {"ok": True, "dir": str(out_dir),
            "n_train": len(train), "n_val": len(val), "n_test": len(test),
            "n_dpo": len(dpo), "n_excluded": len(excluded),
            "files": sorted(p.name for p in out_dir.iterdir()),
            "note": f"export -> {rel}"}


def _dataset_card(name: str, stamp: str, train: list, val: list, test: list,
                  dpo: list, excluded: list, quality: dict, store: DataStore) -> str:
    labels = store.effective_labels()
    manifest = store.manifest()
    ctrl_versions = sorted({m.get("controller_version", "?") for m in manifest})
    dist = quality["distributions"]
    lines = [
        f"# IRIS grounded-reversibility dataset — `{name}` ({stamp} UTC)",
        "",
        "Invertibility-aware safe-web-agent training data. Reversibility labels are",
        "**behaviorally measured** (execute-then-undo probes against live WebArena",
        "Magento), never LLM/human opinion; teacher prose is conditionally distilled",
        "with conclusions pinned (see `docs/plan/IRIS项目计划书.md`).",
        "",
        "## Splits",
        "",
        "| split | rows |", "|---|---|",
        f"| sft_train | {len(train)} |", f"| sft_val | {len(val)} |",
        f"| sft_test | {len(test)} |", f"| dpo_train | {len(dpo)} |",
        f"| excluded (audit trail) | {len(excluded)} |",
        "",
        "## Grounded label provenance",
        "",
        f"- effective labels: `{json.dumps(labels, ensure_ascii=False)}`",
        f"- probe runs recorded: {len(manifest)} (MANIFEST.jsonl), "
        f"controller versions: {', '.join(ctrl_versions) or '—'}",
        f"- human grounded-label overrides: {quality.get('grounded_human_overrides', 0)} "
        "(conflicting samples are EXCLUDED, never rewritten)",
        "",
        "## Distribution snapshot",
        "",
        f"- decision: `{json.dumps(dist['decision'], ensure_ascii=False)}`",
        f"- constraint style: `{json.dumps(dist['constraint_style'], ensure_ascii=False)}`",
        f"- DPO pair types: `{json.dumps(dist['pair_type'], ensure_ascii=False)}`",
        f"- teacher distill coverage: {quality['rates']['distill_coverage']:.1%}, "
        f"pinned-label agreement: {quality['teacher']['pinned_label_agreement']:.1%}",
        "",
        "## Known limitations (honest)",
        "",
        "- Spectrum mid-band (PARTIALLY / WITH_COST) and admin-side labels pending",
        "  destructive-probe approval; current decision labels are collinear with",
        "  action_type × goal variant until the cross-action-class split lands.",
        "- Product-level split is a weak generalization test (see revact/data/splits.py).",
        "",
        "Generated by the IRIS dataset workbench (`python -m revact.cli serve`).",
    ]
    return "\n".join(lines) + "\n"
