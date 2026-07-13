"""Leakage-aware dataset splits.

The original implementation hashed/parsing product slugs only.  That allowed
the request and constraint views of one state (and, for multi-turn rows, the
same trajectory decision point) to be assigned independently by downstream
export code.  This module treats a *state group* as the atomic unit and emits
separate leave-one-axis-out challenge splits for action, privilege, template,
and environment whenever the relevant metadata actually has >=2 values.

No function invents missing metadata: an unavailable challenge split is
reported as unavailable rather than advertised as a completed experiment.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
from typing import Callable


def parse_sid(sid: str):
    """Legacy single-step id parser kept for external callers."""
    p = sid.split("__")
    action = p[0]
    slug = p[1] if len(p) >= 3 else "_base"
    variant = p[-1]
    return action, slug, variant


def source_sample_id(row: dict) -> str:
    """SFT sample id for either an SFT row or a derived DPO pair."""
    if row.get("sample_id"):
        return str(row["sample_id"])
    meta = row.get("meta") or {}
    if meta.get("source_sample_id"):
        return str(meta["source_sample_id"])
    pair_id = str(row.get("pair_id", ""))
    return pair_id.rsplit("__", 1)[0] if pair_id else ""


def state_group_of(row: dict) -> str:
    """Atomic split key; request/constraint variants always share it."""
    meta = row.get("meta") or {}
    for key in ("state_group_id", "state_id", "probe_point_id"):
        if meta.get(key):
            return str(meta[key])
    sid = source_sample_id(row)
    return sid.rsplit("__", 1)[0] if "__" in sid else sid


def entity_of(row: dict) -> str:
    meta = row.get("meta") or {}
    for key in ("canonical_entity_id", "entity_id", "product_id", "product_slug"):
        if meta.get(key):
            return str(meta[key])
    sid = source_sample_id(row)
    if not sid.startswith("mt__"):
        return parse_sid(sid)[1]
    # A multi-turn row without an explicit entity cannot honestly be grouped by
    # product.  Its state group is the conservative no-leak fallback.
    return state_group_of(row)


def template_of(row: dict) -> str:
    return str((row.get("meta") or {}).get("goal_template", ""))


def canonical_entity_of(row: dict) -> str:
    """Formal entity key; unlike :func:`entity_of`, never guesses from IDs."""
    return str((row.get("meta") or {}).get("canonical_entity_id") or "")


def page_template_of(row: dict) -> str:
    return str((row.get("meta") or {}).get("page_template_id") or "")


def environment_of(row: dict) -> str:
    meta = row.get("meta") or {}
    for key in ("environment_instance", "environment_family", "environment_origin"):
        if meta.get(key):
            return str(meta[key])
    return ""


def site_of(row: dict) -> str:
    """Concrete application/site axis; do not collapse every site to webarena."""
    return str((row.get("meta") or {}).get("site") or "")


def privilege_of(row: dict) -> str:
    meta = row.get("meta") or {}
    return str(meta.get("privilege") or meta.get("account_privilege") or "")


def action_of(row: dict) -> str:
    return str((row.get("meta") or {}).get("action_type", ""))


AXES: dict[str, Callable[[dict], str]] = {
    "state_group": state_group_of,
    "entity": entity_of,
    "canonical_entity": canonical_entity_of,
    "template": template_of,
    "goal_template": template_of,
    "page_template": page_template_of,
    "environment": environment_of,
    "site": site_of,
    "action": action_of,
    "privilege": privilege_of,
}

# A publication split is only a split when no identifying family crosses it.
# Assigning each axis independently cannot provide that guarantee: for example,
# two different products may still share a page template or environment.  The
# formal splitter therefore partitions connected components in the graph
# induced jointly by these four axes.
FORMAL_ISOLATION_AXES = (
    "state_group", "canonical_entity", "goal_template", "page_template",
    "environment",
)


def audit_split_leakage(train: list[dict], test: list[dict]) -> dict:
    """Return exact overlap values for every auditable leakage axis."""
    report = {}
    for name, fn in AXES.items():
        left = {v for r in train if (v := fn(r))}
        right = {v for r in test if (v := fn(r))}
        overlap = sorted(left & right)
        report[name] = {"n_train": len(left), "n_test": len(right),
                        "n_overlap": len(overlap), "overlap": overlap[:100]}
    return report


def stable_group_is_val(row: dict, val_frac: float) -> bool:
    """Group-stable validation assignment used by workbench export."""
    # Entity is the stronger key: it also contains all state/goal variants.
    key = entity_of(row)
    h = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % 1000
    return h < int(max(0.0, min(0.5, val_frac)) * 1000)


def _read(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    return [json.loads(ln) for ln in path.open(encoding="utf-8") if ln.strip()]


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _partition(rows: list[dict], key: Callable[[dict], str], frac: float) \
        -> tuple[list[dict], list[dict], list[str]]:
    values = sorted({key(r) for r in rows if key(r)})
    if len(values) < 2:
        return list(rows), [], []
    n_test = max(1, int(len(values) * frac))
    n_test = min(n_test, len(values) - 1)
    held = set(values[-n_test:])
    return ([r for r in rows if key(r) not in held],
            [r for r in rows if key(r) in held], sorted(held))


def _strict_formal_partition(
    rows: list[dict], frac: float,
) -> tuple[list[dict], list[dict], list[str], str]:
    """Partition joint leakage components for a formal publication split.

    Every row must expose every isolation axis.  Rows are connected when they
    share *any* state group, canonical entity, goal/page template, or
    environment.  A connected component is indivisible, which is the only way
    to make overlap zero on all four axes simultaneously.  If the metadata
    graph has only one component, the requested generalisation experiment is
    unavailable; returning the input as a nominal train split would hide that
    fact, so this function deliberately returns two empty sides.
    """
    if not rows:
        return [], [], [], "formal source contains zero rows"

    missing: list[str] = []
    values_by_row: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        values = {name: AXES[name](row) for name in FORMAL_ISOLATION_AXES}
        absent = [name for name, value in values.items() if not value]
        if absent:
            ident = source_sample_id(row) or f"row{index}"
            missing.append(f"{ident}:{','.join(absent)}")
        values_by_row.append(values)
    if missing:
        preview = "; ".join(missing[:10])
        return [], [], [], (
            "formal isolation metadata is missing "
            f"(sample:axes): {preview}")

    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    owner: dict[tuple[str, str], int] = {}
    for index, values in enumerate(values_by_row):
        for axis, value in values.items():
            key = (axis, value)
            if key in owner:
                union(index, owner[key])
            else:
                owner[key] = index

    components: dict[int, list[int]] = collections.defaultdict(list)
    for index in range(len(rows)):
        components[find(index)].append(index)
    groups = list(components.values())
    if len(groups) < 2:
        return [], [], [], (
            "joint state/entity/goal-template/page-template/environment graph has one connected "
            "component; a non-leaking train/test split is unavailable")

    def component_key(indices: list[int]) -> tuple[str, str]:
        identities = sorted(source_sample_id(rows[i]) or f"row{i}" for i in indices)
        digest = hashlib.sha256("\n".join(identities).encode("utf-8")).hexdigest()
        return digest, identities[0]

    groups.sort(key=component_key)
    n_test = max(1, int(len(groups) * frac))
    n_test = min(n_test, len(groups) - 1)
    test_indices = {index for group in groups[-n_test:] for index in group}
    train = [row for index, row in enumerate(rows) if index not in test_indices]
    test = [row for index, row in enumerate(rows) if index in test_indices]
    held = sorted(source_sample_id(rows[group[0]]) or f"row{group[0]}"
                  for group in groups[-n_test:])
    leakage = audit_split_leakage(train, test)
    leaked = [name for name in FORMAL_ISOLATION_AXES
              if leakage[name]["n_overlap"]]
    if not train or not test or leaked:  # defensive: union-find should prevent this
        detail = ",".join(leaked) or "empty-side"
        return [], [], [], f"strict formal partition invariant failed: {detail}"
    return train, test, held, ""


def formal_validation_partition(
    rows: list[dict], val_frac: float,
) -> tuple[list[dict], list[dict], dict]:
    """Create an optional strict validation side from an existing formal train.

    A one-component training set can still be released for fitting, but it
    cannot honestly claim an independent validation split.  In that case all
    rows remain in train, validation is empty, and the structured report marks
    the split unavailable rather than leaking a shared template/environment.
    """
    if val_frac <= 0:
        return list(rows), [], {
            "available": False, "reason": "validation split disabled",
            "n_train": len(rows), "n_val": 0,
            "leakage": audit_split_leakage(rows, []),
        }
    train, val, held, reason = _strict_formal_partition(rows, val_frac)
    if reason:
        return list(rows), [], {
            "available": False, "reason": reason,
            "n_train": len(rows), "n_val": 0,
            "leakage": audit_split_leakage(rows, []),
        }
    return train, val, {
        "available": True, "reason": "", "held_components": held,
        "n_train": len(train), "n_val": len(val),
        "leakage": audit_split_leakage(train, val),
    }


def _challenge_partition(rows: list[dict], axis: Callable[[dict], str],
                         frac: float) -> tuple[list[dict], list[dict], list[str]]:
    """Leave one axis out without leaking the same state/entity into train.

    Rows sharing a test entity but carrying another template/action are dropped
    from this challenge's train side rather than reassigned to test, preserving
    the semantic meaning of the held-out axis.
    """
    values = sorted({axis(r) for r in rows if axis(r)})
    n_test = max(1, int(len(values) * frac))
    n_test = min(n_test, len(values) - 1)
    # Prefer the groups touching the fewest entities; this avoids making a
    # template challenge empty merely because a ubiquitous request template
    # sorts last lexicographically.
    ranked = sorted(values, key=lambda value: (
        len({entity_of(r) for r in rows if axis(r) == value}), value))
    held = set(ranked[:n_test])
    test = [r for r in rows if axis(r) in held]
    test_states = {state_group_of(r) for r in test}
    test_entities = {entity_of(r) for r in test}
    train = [r for r in rows if axis(r) not in held
             and state_group_of(r) not in test_states
             and entity_of(r) not in test_entities]
    return train, test, sorted(held)


def _dpo_for_train(dpo: list[dict], train: list[dict]) -> list[dict]:
    ids = {source_sample_id(r) for r in train}
    return [r for r in dpo if source_sample_id(r) in ids]


def _family_splits(sft: list[dict], dpo: list[dict], holdout_frac: float,
                   suffix: str = "", *, formal: bool = False
                   ) -> tuple[dict[str, list[dict]], dict]:
    # Entity is the default atomic key.  A state cannot cross when its enclosing
    # entity cannot cross; multi-turn rows without entity metadata conservatively
    # fall back to their state-group id.
    if formal:
        non_formal = [source_sample_id(row) for row in sft
                      if (row.get("meta") or {}).get("formal_dataset") is not True]
        if non_formal:
            reason = ("formal source mixes rows without formal_dataset=true: "
                      + ", ".join((item or "<missing-id>")
                                  for item in non_formal[:10]))
            train, test, held = [], [], []
        else:
            train, test, held, reason = _strict_formal_partition(
                sft, holdout_frac)
    else:
        train, test, held = _partition(sft, entity_of, holdout_frac)
        reason = ""
    dev: list[dict] = []
    validation = {
        "available": False,
        "reason": (f"base formal split unavailable: {reason}"
                   if formal and reason else
                   "legacy split has no formal dev side"),
        "n_train": len(train), "n_val": 0,
        "leakage": audit_split_leakage(train, []),
    }
    if formal and not reason:
        train, dev, validation = formal_validation_partition(train, holdout_frac)
        if not validation["available"]:
            reason = "formal dev split unavailable: " + validation["reason"]
            train, dev, test, held = [], [], [], []
    files = {
        f"sft_train{suffix}": train,
        f"sft_dev{suffix}": dev,
        f"sft_test{suffix}": test,
        f"dpo_train{suffix}": _dpo_for_train(dpo, train),
        f"dpo_dev{suffix}": _dpo_for_train(dpo, dev),
    }
    challenges = {}
    if formal and reason:
        for label in ("cross_action", "cross_privilege", "cross_template",
                      "cross_site"):
            files[f"sft_train_{label}{suffix}"] = []
            files[f"sft_test_{label}{suffix}"] = []
            files[f"dpo_train_{label}{suffix}"] = []
            challenges[label] = {
                "available": False,
                "reason": f"base formal split unavailable: {reason}",
            }
        return files, {
            "available": False,
            "formal": True,
            "reason": reason,
            "n_train": 0,
            "n_dev": 0,
            "n_test": 0,
            "n_dpo_train": 0,
            "held_state_groups": [],
            "leakage": audit_split_leakage([], []),
            "validation": validation,
            "challenges": challenges,
        }
    for label, axis in (("cross_action", action_of),
                        ("cross_privilege", privilege_of),
                        ("cross_template", template_of),
                        ("cross_site", site_of)):
        values = sorted({axis(r) for r in sft if axis(r)})
        if len(values) < 2:
            files[f"sft_train_{label}{suffix}"] = []
            files[f"sft_test_{label}{suffix}"] = []
            files[f"dpo_train_{label}{suffix}"] = []
            challenges[label] = {"available": False, "values": values,
                                 "reason": "requires at least two non-empty groups"}
            continue
        ctrain, ctest, cheld = _challenge_partition(sft, axis, holdout_frac)
        leakage = audit_split_leakage(ctrain, ctest)
        strict_leaks = ([name for name in FORMAL_ISOLATION_AXES
                         if leakage[name]["n_overlap"]] if formal else [])
        if not ctrain or not ctest or strict_leaks:
            files[f"sft_train_{label}{suffix}"] = []
            files[f"sft_test_{label}{suffix}"] = []
            files[f"dpo_train_{label}{suffix}"] = []
            challenges[label] = {
                "available": False, "values": values, "held_out": cheld,
                "reason": (
                    "axis is entangled with formal isolation components"
                    if strict_leaks else
                    "axis is entangled with state/entity groups; train or test is empty"),
                "n_train": len(ctrain), "n_test": len(ctest),
                "strict_overlap_axes": strict_leaks,
                "leakage": leakage,
            }
            continue
        files[f"sft_train_{label}{suffix}"] = ctrain
        files[f"sft_test_{label}{suffix}"] = ctest
        files[f"dpo_train_{label}{suffix}"] = _dpo_for_train(dpo, ctrain)
        challenges[label] = {
            "available": True, "values": values, "held_out": cheld,
            "n_train": len(ctrain), "n_test": len(ctest),
            "leakage": leakage,
        }
    return files, {
        # Legacy/development callers historically used ``available`` to mean
        # that the family asset exists, even when its tiny fixture cannot
        # populate a test side.  Keep that compatibility explicit; formal mode
        # uses the stronger publication meaning (both sides are non-empty).
        "available": bool(train and dev and test) if formal else bool(sft),
        "formal": formal,
        "reason": reason,
        "n_train": len(train), "n_test": len(test),
        "n_dev": len(dev),
        "n_dpo_train": len(files[f"dpo_train{suffix}"]),
        "held_state_groups": held,
        "leakage": audit_split_leakage(train, test),
        "validation": validation,
        "challenges": challenges,
    }


def build_splits(sft_path: Path, dpo_path: Path, out_dir: Path,
                 holdout_frac: float = 0.25,
                 multiturn_sft_path: Path | None = None,
                 multiturn_dpo_path: Path | None = None,
                 formal: bool | None = None) -> dict:
    """Build single- and multi-turn grouped splits plus leakage audits.

    Multi-turn paths auto-resolve beside the canonical single-step files.  They
    are emitted to distinct ``*_multiturn.jsonl`` files, never silently mixed.
    """
    sft, dpo = _read(sft_path), _read(dpo_path)
    if formal is None:
        formal = (any((row.get("meta") or {}).get("formal_dataset") is True
                      for row in sft)
                  or "formal" in sft_path.parts or "formal" in out_dir.parts)
    if multiturn_sft_path is None:
        multiturn_sft_path = sft_path.with_name("revact_sft_multiturn.jsonl")
    if multiturn_dpo_path is None:
        multiturn_dpo_path = dpo_path.with_name("revact_dpo_multiturn.jsonl")
    mt_sft, mt_dpo = _read(multiturn_sft_path), _read(multiturn_dpo_path)

    if formal:
        # Publication splits are assigned once over both message families.
        # Splitting single and multi-turn rows independently can put the same
        # entity/template/environment in different partitions across families.
        single_list = [source_sample_id(row) for row in sft]
        multi_list = [source_sample_id(row) for row in mt_sft]
        if any(not sample_id for sample_id in single_list + multi_list):
            raise ValueError("formal SFT rows require non-empty sample_id")
        if len(single_list) != len(set(single_list)):
            raise ValueError("duplicate sample_id within formal single SFT source")
        if len(multi_list) != len(set(multi_list)):
            raise ValueError("duplicate sample_id within formal multi-turn SFT source")
        pair_ids = [str(row.get("pair_id") or "") for row in dpo + mt_dpo]
        if any(not pair_id for pair_id in pair_ids) or len(pair_ids) != len(set(pair_ids)):
            raise ValueError("formal DPO rows require unique non-empty pair_id")
        single_ids = set(single_list)
        multi_ids = set(multi_list)
        duplicate_ids = sorted(single_ids & multi_ids)
        if duplicate_ids:
            raise ValueError(
                "formal single/multiturn sample_id collision: " +
                ", ".join(duplicate_ids[:10]))
        joint_files, single_report = _family_splits(
            sft + mt_sft, dpo + mt_dpo, holdout_frac, formal=True)
        files: dict[str, list[dict]] = {}
        for name, rows in joint_files.items():
            files[name] = [row for row in rows
                           if source_sample_id(row) in single_ids]
            # Always materialize the multi-turn side, even when empty, so a
            # rebuild cannot leave stale rows from an older run.
            files[f"{name}_multiturn"] = [
                row for row in rows if source_sample_id(row) in multi_ids]
        multi_report = {
            "available": bool(mt_sft) and single_report["available"],
            "reason": (single_report.get("reason", "") if mt_sft else
                       "no formal multi-turn rows"),
            "n_train": len(files["sft_train_multiturn"]),
            "n_dev": len(files["sft_dev_multiturn"]),
            "n_test": len(files["sft_test_multiturn"]),
            "n_dpo_train": len(files["dpo_train_multiturn"]),
        }
        report_sft = sft + mt_sft
        report_train = joint_files["sft_train"]
        report_dev = joint_files["sft_dev"]
        report_test = joint_files["sft_test"]
        materialized_ids = [source_sample_id(row) for name in (
            "sft_train", "sft_dev", "sft_test") for row in joint_files[name]]
        if single_report["available"] and collections.Counter(
                materialized_ids) != collections.Counter(single_list + multi_list):
            raise ValueError(
                "formal split partitions do not exactly cover the SFT source")
    else:
        files, single_report = _family_splits(
            sft, dpo, holdout_frac, formal=False)
        multi_report = {"available": bool(mt_sft), "n_train": 0,
                        "n_dev": 0, "n_test": 0}
        if mt_sft:
            mt_files, multi_report = _family_splits(
                mt_sft, mt_dpo, holdout_frac, suffix="_multiturn",
                formal=False)
            files.update(mt_files)
        else:
            # Clear every base/challenge multi-turn shard on rebuild.
            for name in list(files):
                files[f"{name}_multiturn"] = []
        report_sft = sft
        report_train = files["sft_train"]
        report_dev = files["sft_dev"]
        report_test = files["sft_test"]

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in files.items():
        _write(out_dir / f"{name}.jsonl", rows)

    dist = {
        split: dict(collections.Counter(
            ((r.get("meta") or {}).get("action_type", ""),
             (r.get("meta") or {}).get("decision", "")) for r in rows))
        for split, rows in (("train", report_train),
                            ("dev", report_dev),
                            ("test", report_test))
    }
    by_site = dict(collections.Counter(environment_of(r) for r in report_sft))
    report = {
        "available": single_report["available"],
        "formal": formal,
        "reason": single_report.get("reason", ""),
        "n_train": single_report["n_train"],
        "n_dev": single_report["n_dev"],
        "n_test": single_report["n_test"],
        "n_dpo_train": single_report["n_dpo_train"],
        "test_slugs": sorted({entity_of(r) for r in report_test}),
        "distribution": dist, "by_site": by_site,
        "leakage": single_report["leakage"],
        "validation": single_report["validation"],
        "challenges": single_report["challenges"],
        # Backward-compatible report field, now honest when unavailable.
        "cross_site": single_report["challenges"]["cross_site"],
        "multiturn": multi_report,
        "out_dir": str(out_dir),
    }
    # Keep an auditable, machine-readable verdict beside the generated files.
    # Distribution uses tuple keys for backward-compatible Python callers, so
    # the persisted report intentionally contains only JSON-native fields.
    persisted = {
        key: report[key] for key in (
            "available", "formal", "reason", "n_train", "n_dev", "n_test",
            "n_dpo_train", "test_slugs", "by_site", "leakage",
            "validation", "challenges", "cross_site", "multiturn", "out_dir")
    }
    (out_dir / "SPLIT_REPORT.json").write_text(
        json.dumps(persisted, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report
