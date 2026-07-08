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


def compute_quality(store: DataStore | None = None) -> dict:
    store = store or DataStore()
    sft = store.sft()
    dpo = store.dpo()
    distilled = store.sft(distilled=True)
    trajs = store.trajectory_index()
    grounded = store.grounded_runs()
    labels = store.effective_labels()
    ann = annotations.all_effective(store.root)

    # -- volumes ------------------------------------------------------------ #
    volumes = {
        "trajectories": len(trajs),
        "trajectories_success": sum(t["success"] for t in trajs),
        "key_states": len(store.key_states()),
        "reached_states": len(store.reached_states()),
        "grounded_probe_runs": len(grounded),
        "grounded_action_classes": len(labels),
        "sft_samples": len(sft), "dpo_pairs": len(dpo),
        "distilled_samples": len(distilled),
        "splits": store.splits_report(),
    }
    rates = {
        "traj_success_rate": round(volumes["trajectories_success"]
                                   / max(volumes["trajectories"], 1), 3),
        "distill_coverage": round(len(distilled) / max(len(sft), 1), 3),
    }

    # -- distributions ------------------------------------------------------ #
    rev_effective = _dist(labels.values())
    n_rev = sum(v for k, v in rev_effective.items() if k.startswith("REVERSIBLE"))
    n_irr = sum(v for k, v in rev_effective.items()
                if k in ("IRREVERSIBLE", "PARTIALLY_RECOVERABLE"))
    distributions = {
        "reversibility_effective": rev_effective,
        "reversibility_runs": _dist(g["label"] for g in grounded),
        "reversible_vs_irreversible": {"reversible": n_rev, "irreversible": n_irr,
                                       "other": len(labels) - n_rev - n_irr},
        "decision": _dist(s["decision"] for s in sft),
        "constraint_style": _dist(s["constraint_style"] for s in sft),
        "goal_template": _dist(s["goal_template"] for s in sft),
        "pair_type": _dist(p["pair_type"] for p in dpo),
        "action_type": _dist(s["action_type"] for s in sft),
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
                     and d["decision"] == t["decision"]
                     and d["answer"] == t["answer"])
        agree += pinned_ok
        label_drift += not pinned_ok
        prose_changed += (d["observation"], d["reasoning"], d["prediction"]) != \
            (t["observation"], t["reasoning"], t["prediction"])
    teacher = {
        "n_distilled": len(distilled),
        "pinned_label_agreement": round(agree / max(len(distilled), 1), 3),
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

    return {"volumes": volumes, "rates": rates, "distributions": distributions,
            "counterfactual_coverage": coverage, "teacher": teacher,
            "low_quality": low_quality[:300],
            "n_low_quality": len(low_quality),
            "annotations": ann_summary,
            "grounded_human_overrides": human_overrides}
