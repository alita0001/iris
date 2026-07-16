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

import collections
import csv
import io
import json
import re
import shutil
from datetime import datetime, timezone

from .. import config
from ..data.governance import (formal_derivation_reasons,
                               formal_prompt_content_reasons,
                               formal_release_context,
                               formal_release_reasons)
from ..data.splits import (audit_split_leakage, source_sample_id,
                           stable_group_is_val, FORMAL_ISOLATION_AXES)
from . import annotations
from .datasets import DataStore, _jsonl

EXPORTS_DIR = config.OUTPUTS_DIR / "workbench" / "exports"

_DEPLOYMENT_NEGATIVE_SOURCES = frozenset({"legal_candidate", "on_policy"})


def _formal_split_gate(splits_dir, partitions: dict[str, list[dict]],
                       source_rows: list[dict] | None = None) -> dict:
    """Fail closed unless one joint, non-empty train/dev/test split is intact."""
    errors: list[str] = []
    report_path = splits_dir / "SPLIT_REPORT.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        report = {}
        errors.append("split_report_missing_or_invalid")
    if report and (report.get("formal") is not True or
                   report.get("available") is not True):
        errors.append("split_report_not_formal_available")
    for side in ("train", "dev", "test"):
        if not partitions.get(side):
            errors.append(f"formal_{side}_split_empty")
    ids = {
        side: [source_sample_id(row) for row in rows]
        for side, rows in partitions.items()
    }
    for side, side_ids in ids.items():
        if any(not sample_id for sample_id in side_ids):
            errors.append(f"formal_{side}_split_missing_sample_id")
        if len(side_ids) != len(set(side_ids)):
            errors.append(f"formal_{side}_split_duplicate_sample_id")
        expected = report.get(f"n_{side}") if report else None
        if isinstance(expected, int) and expected != len(side_ids):
            errors.append(
                f"formal_{side}_split_count_mismatch:{len(side_ids)}!={expected}")
    if source_rows is not None:
        source_ids = [source_sample_id(row) for row in source_rows]
        if any(not sample_id for sample_id in source_ids):
            errors.append("formal_source_missing_sample_id")
        if len(source_ids) != len(set(source_ids)):
            errors.append("formal_source_duplicate_sample_id")
        materialized = [sample_id for side_ids in ids.values()
                        for sample_id in side_ids]
        if collections.Counter(materialized) != collections.Counter(source_ids):
            errors.append("formal_split_does_not_exactly_cover_source")
    for left, right in (("train", "dev"), ("train", "test"),
                        ("dev", "test")):
        duplicates = sorted(set(ids.get(left, [])) & set(ids.get(right, [])))
        if duplicates:
            errors.append(
                f"split_membership_overlap:{left}:{right}:" +
                ",".join(duplicates[:10]))
        leakage = audit_split_leakage(
            partitions.get(left, []), partitions.get(right, []))
        bad = [axis for axis in FORMAL_ISOLATION_AXES
               if leakage[axis]["n_overlap"]]
        if bad:
            errors.append(f"split_axis_overlap:{left}:{right}:" + ",".join(bad))
    return {"passes": not errors, "errors": errors, "report": report}


def _formal_dpo_source_gate(rows: list[dict], minimum: float = .50) -> dict:
    """Audit the release-wide DPO negative provenance threshold.

    The gate is intentionally computed *after* point/prompt/source-split
    validation and across both single- and multi-turn families.  Computing it
    per row or per family would allow a small compliant shard to make a mostly
    synthetic release look deployment-shaped.
    """
    sources = collections.Counter(
        str((row.get("meta") or {}).get("negative_source") or "missing")
        for row in rows)
    deployment = sum(sources[source] for source in _DEPLOYMENT_NEGATIVE_SOURCES)
    total = len(rows)
    share = deployment / total if total else None
    return {
        "enforced": True,
        "available": bool(total),
        "passes": None if not total else share >= minimum,
        "n_pairs": total,
        "source_counts": dict(sorted(sources.items())),
        "legal_or_on_policy_n": deployment,
        "legal_or_on_policy_share": share,
        "minimum_required_share": minimum,
    }


def export_dataset(store: DataStore | None = None, params: dict | None = None) -> dict:
    store = store or DataStore()
    params = params or {}
    name = re.sub(r"[^A-Za-z0-9_.-]", "-", str(params.get("name", "release")))[:40]
    val_frac = min(0.5, max(0.0, float(params.get("val_frac", 0.15))))
    include_needs_review = bool(params.get("include_needs_review", False))
    prefer_distilled = bool(params.get("prefer_distilled", True))
    formal = bool(params.get("formal", True))
    dry_run = bool(params.get("dry_run", False))

    splits_dir = store.root / "train" / ("formal/splits" if formal else "splits")
    sft_train = _jsonl(splits_dir / "sft_train.jsonl")
    sft_dev = _jsonl(splits_dir / "sft_dev.jsonl")
    sft_test = _jsonl(splits_dir / "sft_test.jsonl")
    dpo_train = _jsonl(splits_dir / "dpo_train.jsonl")
    dpo_dev = _jsonl(splits_dir / "dpo_dev.jsonl")
    mt_train = _jsonl(splits_dir / "sft_train_multiturn.jsonl")
    mt_dev = _jsonl(splits_dir / "sft_dev_multiturn.jsonl")
    mt_test = _jsonl(splits_dir / "sft_test_multiturn.jsonl")
    mt_dpo_train = _jsonl(splits_dir / "dpo_train_multiturn.jsonl")
    mt_dpo_dev = _jsonl(splits_dir / "dpo_dev_multiturn.jsonl")
    if formal:
        source_rows = _jsonl(store.root / "train" / "formal" /
                             config.FORMAL_SFT_PATH.name)
        source_rows += _jsonl(store.root / "train" / "formal" /
                              config.FORMAL_MULTITURN_SFT_PATH.name)
        split_gate = _formal_split_gate(splits_dir, {
            "train": sft_train + mt_train,
            "dev": sft_dev + mt_dev,
            "test": sft_test + mt_test,
        }, source_rows=source_rows)
        if not split_gate["passes"]:
            return {"ok": False, "formal": True,
                    "note": "formal split gate failed",
                    "split_gate": split_gate}
    else:
        split_gate = {"passes": None, "errors": [],
                      "report": {"formal": False}}
        if not sft_train and not sft_test and not mt_train and not mt_test:
            return {"ok": False,
                    "note": "没有 legacy splits 产物：先运行 assemble + split"}

    sample_ann = annotations.effective("sample", store.root)
    grounded_ann = annotations.effective("grounded", store.root)
    distill_ann = annotations.effective("distill", store.root)
    distilled_paths = ([
        store.root / "train" / "formal" /
        config.FORMAL_DISTILLED_SFT_PATH.name,
        store.root / "train" / "formal" /
        config.FORMAL_MULTITURN_DISTILLED_SFT_PATH.name,
    ] if formal else [
        store.root / "train" / "sft" / "revact_sft_distilled.jsonl",
    ])
    distilled_by_id = {
        row.get("sample_id"): row
        for path in distilled_paths for row in _jsonl(path)
    }
    # map action_type -> human override label (via probe_id rows)
    probe_by_id = {g["probe_id"]: g for g in store.grounded_runs() if g["probe_id"]}
    override_by_type: dict[str, str] = {}
    for pid, a in grounded_ann.items():
        ov = a.get("reversibility_override")
        g = probe_by_id.get(pid)
        if ov and g:
            override_by_type[g["action_type"]] = ov

    excluded: list[dict] = []
    formal_context = formal_release_context(store.root) if formal else None

    def exclude(sid: str, reasons: list[str], *, note: str = "") -> None:
        item = {"id": sid, "reason": reasons[0] if reasons else "excluded"}
        if len(reasons) > 1:
            item["reasons"] = reasons
        if note:
            item["note"] = note
        excluded.append(item)

    def keep(row: dict, sid: str, allow_distill: bool = True,
             audit_id: str = "") -> dict | None:
        excluded_id = audit_id or sid
        a = sample_ann.get(sid) or {}
        status = a.get("review_status", "")
        if status == "rejected":
            exclude(excluded_id, ["human-rejected"], note=a.get("note", ""))
            return None
        if status == "needs-review" and not include_needs_review:
            exclude(excluded_id, ["needs-review"])
            return None
        out = dict(row)
        used_distilled = False
        if allow_distill and prefer_distilled and sid in distilled_by_id \
                and (distill_ann.get(sid) or {}).get("review_status") != "rejected":
            out = dict(distilled_by_id[sid])
            used_distilled = True
        meta = dict(out.get("meta") or {})
        if formal:
            assert formal_context is not None
            reasons = formal_release_reasons(meta, formal_context)
            reasons.extend(formal_prompt_content_reasons(out, formal_context))
            if used_distilled:
                source_meta = dict(row.get("meta") or {})
                reasons.extend(formal_derivation_reasons(source_meta, meta))
                if out.get("sample_id") != row.get("sample_id"):
                    reasons.append("derived_sample_id_mismatch")
                if out.get("messages", [])[:-1] != row.get("messages", [])[:-1]:
                    reasons.append("derived_input_messages_mismatch")
            if reasons:
                exclude(excluded_id, reasons)
                return None

        at = meta.get("action_type", "")
        if formal:
            # A class-level opinion from the legacy smoke table must not be
            # projected onto a formal point.  A point-keyed annotation can only
            # trigger exclusion; it never rewrites probe evidence.
            ov = (grounded_ann.get(str(meta.get("probe_point_id") or "")) or {}).get(
                "reversibility_override")
        else:
            ov = override_by_type.get(at)
        if ov and meta.get("reversibility") and ov != meta["reversibility"]:
            exclude(excluded_id, ["label_conflict"],
                    note=f"human override {ov} != pinned {meta['reversibility']}")
            return None

        meta["review_status"] = status or "unreviewed"
        if a.get("note"):
            meta["review_note"] = a["note"]
        if ov:
            meta["human_verified_label"] = ov
        out["meta"] = meta
        return out

    def export_sft_family(train_rows: list[dict], test_rows: list[dict],
                          dev_rows: list[dict] | None = None):
        kept_train = []
        for r in train_rows:
            k = keep(r, r.get("sample_id", ""))
            if k is None:
                continue
            kept_train.append(k)
        if formal:
            train_out = kept_train
            val_out = [k for row in (dev_rows or [])
                       if (k := keep(row, row.get("sample_id", "")))]
            val_report = {
                "available": bool(val_out),
                "reason": "materialized strict formal dev split" if val_out else
                          "formal dev split is empty",
                "n_train": len(train_out), "n_val": len(val_out),
                "leakage": audit_split_leakage(train_out, val_out),
            }
        else:
            train_out, val_out = [], []
            for row in kept_train:
                (val_out if stable_group_is_val(row, val_frac)
                 else train_out).append(row)
            val_report = {
                "available": bool(val_out), "reason": "legacy entity hash",
                "n_train": len(train_out), "n_val": len(val_out),
            }
        test_out = [k for r in test_rows
                    if (k := keep(r, r.get("sample_id", "")))]
        return train_out, val_out, test_out, val_report

    train, val, test, val_split_report = export_sft_family(
        sft_train, sft_test, sft_dev)
    mt_train_out, mt_val, mt_test_out, mt_val_split_report = export_sft_family(
        mt_train, mt_test, mt_dev)
    allowed_train_by_id = {r.get("sample_id"): r for r in train + mt_train_out}
    allowed_train_ids = set(allowed_train_by_id)
    allowed_dev_by_id = {r.get("sample_id"): r for r in val + mt_val}
    allowed_dev_ids = set(allowed_dev_by_id)

    def export_dpo_family(rows: list[dict], allowed_ids: set,
                          allowed_by_id: dict) -> list[dict]:
        out = []
        for r in rows:
            sid = source_sample_id(r)
            pair_id = str(r.get("pair_id") or sid)
            # DPO derived from a held-out validation/test sample must not leak
            # its chosen response into training.
            if sid not in allowed_ids:
                exclude(pair_id, ["dpo_source_not_in_formal_train" if formal
                                  else "dpo_source_not_in_train"])
                continue
            if formal:
                source = allowed_by_id[sid]
                pre_reasons = formal_derivation_reasons(
                    source.get("meta") or {}, r.get("meta") or {})
                if r.get("prompt") != source.get("messages", [])[:-1]:
                    pre_reasons.append("derived_input_messages_mismatch")
                if pre_reasons:
                    exclude(pair_id, pre_reasons)
                    continue
            k = keep(dict(r), sid, allow_distill=False, audit_id=pair_id)
            if k is not None:
                if formal:
                    source = allowed_by_id[sid]
                    reasons = formal_derivation_reasons(
                        source.get("meta") or {}, k.get("meta") or {})
                    if k.get("prompt") != source.get("messages", [])[:-1]:
                        reasons.append("dpo_prompt_source_mismatch")
                    if reasons:
                        exclude(r.get("pair_id") or sid, reasons)
                        continue
                out.append(k)
        return out

    dpo = export_dpo_family(dpo_train, allowed_train_ids, allowed_train_by_id)
    dpo_val = export_dpo_family(dpo_dev, allowed_dev_ids, allowed_dev_by_id)
    mt_dpo = export_dpo_family(
        mt_dpo_train, allowed_train_ids, allowed_train_by_id)
    mt_dpo_val = export_dpo_family(
        mt_dpo_dev, allowed_dev_ids, allowed_dev_by_id)
    if formal:
        post_filter_split_gate = _formal_split_gate(splits_dir, {
            "train": train + mt_train_out,
            "dev": val + mt_val,
            "test": test + mt_test_out,
        })
        # The report describes the source materialization; pairwise audits are
        # recomputed over kept rows.  Filtering may empty a side, which blocks
        # publication rather than silently producing a two-way release.
        if not post_filter_split_gate["passes"]:
            return {"ok": False, "formal": True,
                    "note": "formal post-filter split gate failed",
                    "split_gate": post_filter_split_gate,
                    "n_train": len(train), "n_val": len(val),
                    "n_test": len(test), "n_dpo": 0,
                    "n_multiturn_train": len(mt_train_out),
                    "n_multiturn_val": len(mt_val),
                    "n_multiturn_test": len(mt_test_out),
                    "n_multiturn_dpo": 0,
                    "n_excluded": len(excluded), "excluded": excluded,
                    "formal_grounding_error": (
                        formal_context.grounding_error
                        if formal_context else "")}
    if formal:
        dpo_source_gate = _formal_dpo_source_gate(
            dpo + dpo_val + mt_dpo + mt_dpo_val)
        if dpo_source_gate["passes"] is False:
            share = dpo_source_gate["legal_or_on_policy_share"]
            reason = ("formal_dpo_legal_or_on_policy_share_below_0.500:"
                      f"{share:.3f}")
            failed_rows = dpo + dpo_val + mt_dpo + mt_dpo_val
            for row in failed_rows:
                exclude(str(row.get("pair_id") or source_sample_id(row)), [reason])
            return {
                "ok": False, "formal": True,
                "note": "formal DPO negative-source gate failed",
                "formal_dpo_source_gate": dpo_source_gate,
                "n_excluded": len(excluded), "excluded": excluded,
            }
    else:
        dpo_source_gate = {
            "enforced": False, "available": bool(
                dpo or dpo_val or mt_dpo or mt_dpo_val),
            "passes": None, "n_pairs": (len(dpo) + len(dpo_val) +
                                           len(mt_dpo) + len(mt_dpo_val)),
            "reason": "legacy-development export",
        }

    if dry_run:
        # All source, lineage, prompt, split, annotation and DPO-provenance
        # gates above have executed.  Return the exact would-write inventory
        # before allocating a timestamped directory or touching any file.
        return {
            "ok": True, "dry_run": True, "formal": formal,
            "note": "all export gates passed; no files written",
            "output_dir": None,
            "n_train": len(train), "n_val": len(val), "n_test": len(test),
            "n_dpo": len(dpo), "n_dpo_val": len(dpo_val),
            "n_multiturn_train": len(mt_train_out),
            "n_multiturn_val": len(mt_val),
            "n_multiturn_test": len(mt_test_out),
            "n_multiturn_dpo": len(mt_dpo),
            "n_multiturn_dpo_val": len(mt_dpo_val),
            "n_excluded": len(excluded), "excluded": excluded,
            "split_gate": split_gate,
            "formal_dpo_source_gate": dpo_source_gate,
            "would_write": [
                "sft_train.jsonl", "sft_val.jsonl", "sft_test.jsonl",
                "dpo_train.jsonl", "dpo_val.jsonl",
                "sft_train_multiturn.jsonl", "sft_val_multiturn.jsonl",
                "sft_test_multiturn.jsonl", "dpo_train_multiturn.jsonl",
                "dpo_val_multiturn.jsonl", "excluded.jsonl", "samples.csv",
                "stats.json", "prompts.json", "prompt_bundles/",
            ],
        }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    out_dir = EXPORTS_DIR / f"{stamp}__{name}"
    out_dir.mkdir(parents=True, exist_ok=False)

    def write_jsonl(fname: str, rows: list[dict]) -> None:
        with (out_dir / fname).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl("sft_train.jsonl", train)
    write_jsonl("sft_val.jsonl", val)
    write_jsonl("sft_test.jsonl", test)
    write_jsonl("dpo_train.jsonl", dpo)
    write_jsonl("dpo_val.jsonl", dpo_val)
    write_jsonl("sft_train_multiturn.jsonl", mt_train_out)
    write_jsonl("sft_val_multiturn.jsonl", mt_val)
    write_jsonl("sft_test_multiturn.jsonl", mt_test_out)
    write_jsonl("dpo_train_multiturn.jsonl", mt_dpo)
    write_jsonl("dpo_val_multiturn.jsonl", mt_dpo_val)
    write_jsonl("excluded.jsonl", excluded)

    csv_buf = io.StringIO()
    w = csv.writer(csv_buf)
    w.writerow(["sample_id", "family", "split", "action_type", "variant", "decision",
                "reversibility", "constraint_style", "review_status", "prose_source"])
    for family, split_name, rows in [
            ("single", "train", train), ("single", "val", val),
            ("single", "test", test), ("multiturn", "train", mt_train_out),
            ("multiturn", "val", mt_val), ("multiturn", "test", mt_test_out)]:
        for r in rows:
            m = r.get("meta") or {}
            w.writerow([r.get("sample_id"), family, split_name, m.get("action_type"),
                        m.get("variant"), m.get("decision"), m.get("reversibility"),
                        m.get("constraint_style"), m.get("review_status"),
                        m.get("prose_source", "template")])
    (out_dir / "samples.csv").write_text(csv_buf.getvalue(), encoding="utf-8")

    from .quality import compute_release_quality
    release_sft = train + val + test + mt_train_out + mt_val + mt_test_out
    release_dpo = dpo + dpo_val + mt_dpo + mt_dpo_val
    split_by_id = {
        str(row.get("sample_id") or ""): side
        for side, rows in (("train", train + mt_train_out),
                           ("dev", val + mt_val),
                           ("test", test + mt_test_out))
        for row in rows
    }
    quality = compute_release_quality(
        release_sft, release_dpo, split_by_id=split_by_id)
    (out_dir / "stats.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=1), encoding="utf-8")
    # Provenance: content-addressed prompt bundles make every prompts_fp
    # recoverable.  Missing historical bundles are reported, never substituted
    # with today's prompt text.
    from .. import prompts as _prompts
    from ..prompt_store import (bundle_dir, generation_bundle_dir,
                                store_bundle)
    current_bundle = store_bundle(_prompts.effective(), root=store.root,
                                  author="workbench-export")
    (out_dir / "prompts.json").write_text(
        json.dumps({"fingerprint": _prompts.fingerprint(),
                    "prompts": _prompts.effective()},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    prompt_fps = sorted({str((r.get("meta") or {}).get("prompts_fp"))
                         for r in train + val + test + mt_train_out + mt_val + mt_test_out
                         if (r.get("meta") or {}).get("prompts_fp")})
    bundle_out = out_dir / "prompt_bundles"
    bundle_out.mkdir()
    missing_prompt_bundles = []
    for fp in prompt_fps:
        src = bundle_dir(store.root) / f"{fp}.json"
        if src.exists():
            shutil.copyfile(src, bundle_out / src.name)
        else:
            missing_prompt_bundles.append(fp)
    # Always include the current registry snapshot, while keeping it distinct
    # from historical sample fingerprints.
    shutil.copyfile(current_bundle, bundle_out / current_bundle.name)
    generation_fps = sorted({
        str(fp)
        for row in train + val + test + mt_train_out + mt_val + mt_test_out
        for fp in ((row.get("meta") or {}).get("prompt_generation_fp"),
                   (row.get("meta") or {}).get("teacher_prompt_generation_fp"))
        if fp
    })
    generation_out = out_dir / "prompt_generation_bundles"
    generation_out.mkdir()
    missing_generation_bundles = []
    for fp in generation_fps:
        src = generation_bundle_dir(store.root) / f"{fp}.json"
        if src.exists():
            shutil.copyfile(src, generation_out / src.name)
        else:
            missing_generation_bundles.append(fp)
    (out_dir / "provenance.json").write_text(json.dumps({
        "formal": formal, "sample_prompt_fingerprints": prompt_fps,
        "missing_prompt_bundles": missing_prompt_bundles,
        "sample_prompt_generation_fingerprints": generation_fps,
        "missing_prompt_generation_bundles": missing_generation_bundles,
        "current_prompt_fingerprint": _prompts.fingerprint(),
        "formal_grounding_error": (
            formal_context.grounding_error if formal_context else ""),
        "formal_grounding_points": (
            len(formal_context.points) if formal_context else None),
        "formal_dpo_source_gate": dpo_source_gate,
        "formal_split_gate": split_gate,
        "validation_split": val_split_report,
        "multiturn_validation_split": mt_val_split_report,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "dataset_card.md").write_text(
        _dataset_card(name, stamp, train, val, test, dpo, excluded, quality, store,
                      mt_train_out, mt_val, mt_test_out, mt_dpo, formal,
                      missing_prompt_bundles, missing_generation_bundles,
                      dpo_source_gate,
                      val_split_report, mt_val_split_report),
        encoding="utf-8")

    try:
        rel = str(out_dir.relative_to(config.PROJECT_ROOT))
    except ValueError:
        rel = str(out_dir)
    return {"ok": True, "dir": str(out_dir),
            "n_train": len(train), "n_val": len(val), "n_test": len(test),
            "n_dpo": len(dpo), "n_dpo_val": len(dpo_val),
            "n_multiturn_train": len(mt_train_out),
            "n_multiturn_val": len(mt_val),
            "n_multiturn_test": len(mt_test_out),
            "n_multiturn_dpo": len(mt_dpo),
            "n_multiturn_dpo_val": len(mt_dpo_val),
            "n_excluded": len(excluded),
            "formal": formal, "missing_prompt_bundles": missing_prompt_bundles,
            "missing_prompt_generation_bundles": missing_generation_bundles,
            "formal_dpo_source_gate": dpo_source_gate,
            "formal_split_gate": split_gate,
            "validation_split": val_split_report,
            "multiturn_validation_split": mt_val_split_report,
            "files": sorted(p.name for p in out_dir.iterdir()),
            "note": f"export -> {rel}"}


def _dataset_card(name: str, stamp: str, train: list, val: list, test: list,
                  dpo: list, excluded: list, quality: dict, store: DataStore,
                  mt_train: list | None = None, mt_val: list | None = None,
                  mt_test: list | None = None, mt_dpo: list | None = None,
                  formal: bool = True,
                  missing_prompt_bundles: list[str] | None = None,
                  missing_generation_bundles: list[str] | None = None,
                  dpo_source_gate: dict | None = None,
                  validation_split: dict | None = None,
                  multiturn_validation_split: dict | None = None) -> str:
    mt_train, mt_val, mt_test, mt_dpo = (mt_train or [], mt_val or [],
                                        mt_test or [], mt_dpo or [])
    missing_prompt_bundles = missing_prompt_bundles or []
    missing_generation_bundles = missing_generation_bundles or []
    dpo_source_gate = dpo_source_gate or {}
    validation_split = validation_split or {}
    multiturn_validation_split = multiturn_validation_split or {}
    legacy_labels = store.effective_labels()
    legacy_manifest = store.manifest()
    formal_grounding = store.formal_grounding()
    formal_items = formal_grounding["items"]
    labels = ({p["probe_point_id"]: {
        "effect_status": p["effect_status"],
        "recovery_status": p["recovery_status"],
    } for p in formal_items} if formal else legacy_labels)
    ctrl_versions = sorted({p.get("controller_version", "?") for p in formal_items}) \
        if formal else sorted({m.get("controller_version", "?")
                               for m in legacy_manifest})
    dist = quality["distributions"]
    lines = [
        f"# IRIS grounded-reversibility dataset — `{name}` ({stamp} UTC)",
        "",
        "Operational-recoverability training artifacts. Labels are scoped to the",
        "recorded environment, signals, controller set, privilege and step budget;",
        "they are not universal safety or mathematical irreversibility claims.",
        "",
        "## Splits",
        "",
        "| split | rows |", "|---|---|",
        f"| sft_train | {len(train)} |", f"| sft_val | {len(val)} |",
        f"| sft_test | {len(test)} |", f"| dpo_train | {len(dpo)} |",
        f"| sft_train_multiturn | {len(mt_train)} |",
        f"| sft_val_multiturn | {len(mt_val)} |",
        f"| sft_test_multiturn | {len(mt_test)} |",
        f"| dpo_train_multiturn | {len(mt_dpo)} |",
        f"| excluded (audit trail) | {len(excluded)} |",
        "",
        "## Grounded label provenance",
        "",
        f"- release labels keyed by {'probe_point_id' if formal else 'legacy action_type'}: "
        f"`{json.dumps(labels, ensure_ascii=False)}`",
        f"- formal points / point manifest: {formal_grounding['n_points']} / "
        f"{formal_grounding['n_manifest']}; exact 1:1={formal_grounding['one_to_one']}; "
        f"integrity error={formal_grounding['error'] or 'none'}",
        f"- legacy class-smoke rows / manifest (not formal supervision): "
        f"{len(store.grounded_runs())} / {len(legacy_manifest)}",
        f"- controller versions for selected tier: {', '.join(ctrl_versions) or '—'}",
        f"- human grounded-label overrides: {quality.get('grounded_human_overrides', 0)} "
        "(conflicting samples are EXCLUDED, never rewritten)",
        f"- formal governance gates enabled: {formal}",
        "- formal DPO legal/on-policy source gate: "
        f"`{json.dumps(dpo_source_gate, ensure_ascii=False)}`",
        "- validation component isolation: "
        f"`{json.dumps(validation_split, ensure_ascii=False)}`",
        "- multiturn validation component isolation: "
        f"`{json.dumps(multiturn_validation_split, ensure_ascii=False)}`",
        f"- missing immutable prompt bundles: {missing_prompt_bundles or 'none'}",
        "- missing immutable prompt-generation bundles: "
        f"{missing_generation_bundles or 'none'}",
        "",
        "## Distribution snapshot",
        "",
        f"- decision: `{json.dumps(dist['decision'], ensure_ascii=False)}`",
        f"- constraint style: `{json.dumps(dist['constraint_style'], ensure_ascii=False)}`",
        f"- DPO pair types: `{json.dumps(dist['pair_type'], ensure_ascii=False)}`",
        "- teacher distill coverage: " +
        (f"{quality['rates']['distill_coverage']:.1%}" if
         quality['rates']['distill_coverage'] is not None else "undefined") +
        ", pinned-label agreement: " +
        (f"{quality['teacher']['pinned_label_agreement']:.1%}" if
         quality['teacher']['pinned_label_agreement'] is not None else
         "undefined (no teacher rows)"),
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
