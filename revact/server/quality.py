"""Workbench quality report: dataset volumes, distributions, coverage, QC.

Everything is computed from pipeline artifacts + annotation overlays; the QC
pass reuses ``revact.train.distill.qc_check`` (the same contradiction rules
that gate teacher prose) so the workbench and the pipeline cannot disagree
about what "contradicts a pinned label" means.
"""
from __future__ import annotations

import collections

from ..train.distill import qc_check
from . import annotations
from .datasets import DataStore


def _dist(items) -> dict:
    return dict(collections.Counter(items))


def compute_release_quality(sft_rows: list[dict], dpo_rows: list[dict],
                            *, split_by_id: dict[str, str] | None = None) -> dict:
    """Statistics over the exact kept release rows, never the default tier."""
    split_by_id = split_by_id or {}
    metas = [row.get("meta") or {} for row in sft_rows]
    dpo_metas = [row.get("meta") or {} for row in dpo_rows]
    teacher_rows = [meta for meta in metas if
                    meta.get("prose_source") == "teacher" or
                    meta.get("teacher_prompts_fp")]
    n_sft = len(sft_rows)
    coverage = len(teacher_rows) / n_sft if n_sft else None
    return {
        "scope": "exact_export_rows",
        "volumes": {
            "sft_samples": n_sft,
            "dpo_pairs": len(dpo_rows),
            "distilled_samples": len(teacher_rows),
        },
        "rates": {"distill_coverage": coverage},
        "teacher": {
            "n_distilled": len(teacher_rows),
            # Every retained formal teacher row has already passed the pinned
            # completion/provenance gate.  With no teacher rows the statistic
            # is undefined, not 0% agreement.
            "pinned_label_agreement": 1.0 if teacher_rows else None,
        },
        "distributions": {
            "decision": _dist(meta.get("decision", "") for meta in metas),
            "constraint_style": _dist(
                meta.get("constraint_style", "") for meta in metas),
            "pair_type": _dist(meta.get("pair_type", "") for meta in dpo_metas),
            "action_type": _dist(meta.get("action_type", "") for meta in metas),
            "site": _dist(meta.get("site", "") for meta in metas),
            "effect_status": _dist(
                meta.get("effect_status", "") for meta in metas),
            "recovery_status": _dist(
                meta.get("recovery_status", "") for meta in metas),
            "environment_origin": _dist(
                meta.get("environment_origin", "") for meta in metas),
            "split": _dist(split_by_id.get(str(row.get("sample_id") or ""),
                                            "unassigned")
                           for row in sft_rows),
        },
    }


def compute_quality(store: DataStore | None = None) -> dict:
    store = store or DataStore()
    # The workbench quality page is a release-readiness view.  Historical
    # assets are inventoried separately and never enter unqualified metrics.
    sft_single = store.sft(tier="formal")
    sft_multi = store.sft(family="multiturn", tier="formal")
    sft = sft_single + sft_multi
    dpo_single = store.dpo(tier="formal")
    dpo_multi = store.dpo(family="multiturn", tier="formal")
    dpo = dpo_single + dpo_multi
    distilled = store.sft(distilled=True, family="all", tier="formal")
    trajs = store.trajectory_index()
    formal_grounding = store.formal_grounding()
    grounded = formal_grounding["items"]
    grounded_action_types = {
        str(point.get("action_type") or "") for point in grounded
        if str(point.get("action_type") or "")}
    legacy_grounded = store.grounded_runs()
    legacy_labels = store.effective_labels()
    legacy_sft_single = store.sft(tier="legacy")
    legacy_sft_multi = store.sft(family="multiturn", tier="legacy")
    legacy_dpo_single = store.dpo(tier="legacy")
    legacy_dpo_multi = store.dpo(family="multiturn", tier="legacy")
    legacy_distilled = store.sft(
        distilled=True, family="all", tier="legacy")
    ann = annotations.all_effective(store.root)

    # -- volumes ------------------------------------------------------------ #
    volumes = {
        "trajectories": len(trajs),
        "trajectories_success": sum(t["success"] for t in trajs),
        "key_states": len(store.key_states()),
        "reached_states": len(store.reached_states()),
        "formal_probe_points": len(grounded),
        "formal_point_manifest": formal_grounding["n_manifest"],
        "formal_grounding_ok": formal_grounding["ok"],
        # Compatibility field names now refer to the formal point tier.
        "grounded_probe_runs": len(grounded),
        "grounded_action_classes": len(grounded_action_types),
        "sft_samples": len(sft), "sft_single": len(sft_single),
        "sft_multiturn": len(sft_multi),
        "dpo_pairs": len(dpo), "dpo_single": len(dpo_single),
        "dpo_multiturn": len(dpo_multi),
        "distilled_samples": len(distilled),
        "splits": store.splits_report(),
    }
    rates = {
        "traj_success_rate": round(volumes["trajectories_success"]
                                   / max(volumes["trajectories"], 1), 3),
        "distill_coverage": (
            round(len(distilled) / len(sft), 3) if sft else None),
    }

    # -- distributions ------------------------------------------------------ #
    distributions = {
        "effect_status": _dist(point.get("effect_status", "")
                               for point in grounded),
        "recovery_status": _dist(point.get("recovery_status", "")
                                 for point in grounded),
        "grounding_action_type": _dist(point.get("action_type", "")
                                       for point in grounded),
        "grounding_site": _dist(point.get("site", "") for point in grounded),
        "decision": _dist(s["decision"] for s in sft),
        "constraint_style": _dist(s["constraint_style"] for s in sft),
        "goal_template": _dist(s["goal_template"] for s in sft),
        "pair_type": _dist(p["pair_type"] for p in dpo),
        "action_type": _dist(s["action_type"] for s in sft),
        "sample_family": _dist(s.get("family", "single") for s in sft),
        "environment_origin": _dist(s.get("environment_origin", "unknown") for s in sft),
        "is_mock": _dist(str(s.get("is_mock")) for s in sft),
        "collector_success": _dist(str(s.get("collector_success")) for s in sft),
        "split": _dist(s["split"] for s in sft),
        "decision_matrix": [
            {"action_type": a, "variant": v, "decision": d, "n": n}
            for (a, v, d), n in sorted(collections.Counter(
                (s["action_type"], s["variant"], s["decision"]) for s in sft).items())],
    }

    # -- counterfactual coverage -------------------------------------------- #
    pairs_by_sample: dict[str, set] = collections.defaultdict(set)
    for p in dpo:
        sid = p["pair_id"].rsplit("__", 1)[0]
        pairs_by_sample[sid].add(p["pair_type"])
    covered = [s for s in sft if pairs_by_sample.get(s["sample_id"])]
    coverage = {
        "samples_with_pairs": len(covered),
        "coverage_rate": round(len(covered) / max(len(sft), 1), 3),
        "avg_pairs_per_sample": round(len(dpo) / max(len(sft), 1), 2),
        "pair_types_per_sample": _dist(
            len(v) for v in pairs_by_sample.values()),
    }

    # -- teacher vs template (pinned-label fidelity) ------------------------- #
    tmpl_by_id = {s["sample_id"]: s for s in sft}
    agree = prose_changed = label_drift = 0
    for d in distilled:
        t = tmpl_by_id.get(d["sample_id"])
        if not t:
            continue
        pinned_ok = (d["reversibility"] == t["reversibility"]
                     and d.get("undo") == t.get("undo")
                     and d["decision"] == t["decision"]
                     and d["answer"] == t["answer"])
        agree += pinned_ok
        label_drift += not pinned_ok
        prose_changed += (d["observation"], d["reasoning"], d["prediction"],
                          d.get("rev_check")) != \
            (t["observation"], t["reasoning"], t["prediction"], t.get("rev_check"))
    teacher = {
        "n_distilled": len(distilled),
        "pinned_label_agreement": (
            round(agree / len(distilled), 3) if distilled else None),
        "pinned_label_drift": label_drift,
        "prose_changed": prose_changed,
    }

    # -- low-quality list ---------------------------------------------------- #
    sample_ann = ann.get("sample", {})
    grounded_ann = ann.get("grounded", {})
    low_quality = []

    def flag(sid: str, reason: str):
        low_quality.append({"sample_id": sid, "reason": reason})

    for s in sft:
        if s["reversibility"] in ("UNKNOWN", "NO_EFFECT"):
            flag(s["sample_id"], f"non-grounded reversibility: {s['reversibility']}")
        if not s["answer"]:
            flag(s["sample_id"], "missing <answer>")
        if not s["obs"].strip():
            flag(s["sample_id"], "empty observation input")
        qc = qc_check(s["assistant"], s["reversibility"], s["decision"])
        if qc:
            flag(s["sample_id"], f"QC: {qc}")
        st = (sample_ann.get(s["sample_id"]) or {}).get("review_status")
        if st == "rejected":
            flag(s["sample_id"], "human-rejected")
        elif st == "needs-review":
            flag(s["sample_id"], "needs-review")
    for d in distilled:
        qc = qc_check(d["assistant"], d["reversibility"], d["decision"])
        if qc:
            flag(d["sample_id"], f"distilled QC: {qc}")

    # -- annotation summary --------------------------------------------------#
    ann_summary = {
        kind: {"n_targets": len(items),
               "statuses": _dist((v.get("review_status") or "edited")
                                 for v in items.values())}
        for kind, items in ann.items() if items
    }
    human_overrides = sum(1 for v in grounded_ann.values()
                          if v.get("reversibility_override"))

    legacy_assets = {
        "asset_tier": "legacy",
        "formal_supervision": False,
        "grounded_class_smoke_rows": len(legacy_grounded),
        "grounded_action_classes": len(legacy_labels),
        "sft_samples": len(legacy_sft_single) + len(legacy_sft_multi),
        "sft_single": len(legacy_sft_single),
        "sft_multiturn": len(legacy_sft_multi),
        "dpo_pairs": len(legacy_dpo_single) + len(legacy_dpo_multi),
        "dpo_single": len(legacy_dpo_single),
        "dpo_multiturn": len(legacy_dpo_multi),
        "distilled_samples": len(legacy_distilled),
        "reversibility_display": _dist(legacy_labels.values()),
    }

    return {"scope": "formal", "asset_tier": "formal",
            "volumes": volumes, "rates": rates, "distributions": distributions,
            "counterfactual_coverage": coverage, "teacher": teacher,
            "legacy_assets": legacy_assets,
            "low_quality": low_quality[:300],
            "n_low_quality": len(low_quality),
            "annotations": ann_summary,
            "grounded_human_overrides": human_overrides}
