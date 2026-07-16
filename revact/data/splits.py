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
import os
from pathlib import Path
from typing import Callable


FORMAL_DPO_SUPPLEMENT_MANIFEST_SCHEMA_VERSION = \
    "iris.formal_dpo_supplement_manifest.v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_sha256(values: list[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(values)).encode("utf-8")).hexdigest()


def _supplement_source_id(row: dict) -> str:
    """Require explicit DPO lineage; never guess a release join from pair_id."""
    return str((row.get("meta") or {}).get("source_sample_id") or "").strip()


def formal_dpo_supplement_manifest_path(body: Path) -> Path:
    return Path(str(Path(body)) + ".manifest.json")


def write_formal_dpo_supplement_manifest(
    body: Path, *, release_id: str,
) -> Path:
    """Pin one immutable supplemental DPO body to a named release.

    The helper is intentionally explicit rather than automatically called by
    :func:`build_splits`: release authors must freeze a completed supplement
    before it can participate in publication splits.
    """
    body = Path(body)
    rows = _read(body)
    release_id = str(release_id or "").strip()
    if not body.is_file():
        raise ValueError(f"supplement body does not exist: {body}")
    if not release_id:
        raise ValueError("formal DPO supplement release_id is required")
    pair_ids = [str(row.get("pair_id") or "") for row in rows]
    source_ids = [_supplement_source_id(row) for row in rows]
    if not rows:
        raise ValueError("formal DPO supplement cannot be empty")
    if any(not pair_id for pair_id in pair_ids) or \
            len(pair_ids) != len(set(pair_ids)):
        raise ValueError("formal DPO supplement requires unique non-empty pair_id")
    if any(not source_id for source_id in source_ids):
        raise ValueError("formal DPO supplement requires source_sample_id")
    if any((row.get("meta") or {}).get("formal_dataset") is not True
           for row in rows):
        raise ValueError("formal DPO supplement contains a non-formal row")
    manifest = {
        "schema_version": FORMAL_DPO_SUPPLEMENT_MANIFEST_SCHEMA_VERSION,
        "release_id": release_id,
        "body_name": body.name,
        "body_sha256": _file_sha256(body),
        "n_rows": len(rows),
        "pair_ids_sha256": _ids_sha256(pair_ids),
        "source_sample_ids_sha256": _ids_sha256(source_ids),
    }
    path = formal_dpo_supplement_manifest_path(body)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing == manifest:
            return path
        raise ValueError(
            f"refusing to overwrite non-matching supplement manifest: {path}")
    text = json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        # A concurrent publisher won the race; never replace its release pin.
        raise ValueError(
            f"supplement manifest appeared concurrently: {path}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Directory fsync is unavailable on some filesystems; the file
            # itself has still been flushed and fsynced.
            pass
    except BaseException:
        if path.exists():
            path.unlink()
        raise
    return path


def load_formal_dpo_supplement(body: Path) -> tuple[list[dict], dict]:
    """Load one manifest-pinned formal DPO supplement fail-closed."""
    body = Path(body)
    manifest_path = formal_dpo_supplement_manifest_path(body)
    if not body.is_file() or not manifest_path.is_file():
        raise ValueError(
            f"formal DPO supplement body/manifest are both required: {body}")
    rows = _read(body)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid supplement manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"supplement manifest must be an object: {manifest_path}")
    pair_ids = [str(row.get("pair_id") or "") for row in rows]
    source_ids = [_supplement_source_id(row) for row in rows]
    expected = {
        "schema_version": FORMAL_DPO_SUPPLEMENT_MANIFEST_SCHEMA_VERSION,
        "release_id": str(manifest.get("release_id") or "").strip(),
        "body_name": body.name,
        "body_sha256": _file_sha256(body),
        "n_rows": len(rows),
        "pair_ids_sha256": _ids_sha256(pair_ids),
        "source_sample_ids_sha256": _ids_sha256(source_ids),
    }
    if not expected["release_id"]:
        raise ValueError(f"supplement manifest lacks release_id: {manifest_path}")
    if manifest != expected:
        raise ValueError(
            f"supplement manifest does not exactly pin body/pairs/sources: {body}")
    if not rows:
        raise ValueError(f"formal DPO supplement is empty: {body}")
    if any(not pair_id for pair_id in pair_ids) or \
            len(pair_ids) != len(set(pair_ids)):
        raise ValueError(
            f"formal DPO supplement requires unique non-empty pair_id: {body}")
    if any(not source_id for source_id in source_ids):
        raise ValueError(
            f"formal DPO supplement requires source_sample_id: {body}")
    non_formal = [pair_id for pair_id, row in zip(pair_ids, rows)
                  if (row.get("meta") or {}).get("formal_dataset") is not True]
    if non_formal:
        raise ValueError(
            "formal DPO supplement contains non-formal rows: " +
            ", ".join(non_formal[:10]))
    return rows, manifest


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

SPLIT_REPORT_SCHEMA_VERSION = "iris.split_report.v2"
FORMAL_JOINT_GRAPH_SCHEMA_VERSION = "iris.formal_joint_graph_diagnostics.v1"

# Diagnostics deliberately include the two challenge axes ``site`` and
# ``action`` in addition to the five axes that actually induce the graph.  They
# do not affect connectivity; they make confounds such as "one component spans
# two sites only because of a shared goal template" visible in the report.
_COMPONENT_DIAGNOSTIC_AXES = (
    "state_group", "canonical_entity", "goal_template", "page_template",
    "environment", "site", "action",
)


def _component_sort_key(rows: list[dict], indices: list[int]) -> tuple[str, str]:
    identities = sorted(source_sample_id(rows[i]) or f"row{i}" for i in indices)
    digest = hashlib.sha256("\n".join(identities).encode("utf-8")).hexdigest()
    return digest, identities[0]


def _axis_distribution(rows: list[dict], indices: list[int], axis: str) \
        -> dict[str, int]:
    counts = collections.Counter(
        AXES[axis](rows[index]) or "<missing>" for index in indices)
    return dict(sorted(counts.items()))


def _formal_joint_graph(
    rows: list[dict],
) -> tuple[list[list[int]], list[str], dict]:
    """Build the exact graph used by formal partitioning plus diagnostics.

    Rows are unioned only through :data:`FORMAL_ISOLATION_AXES`.  The returned
    JSON-native diagnostics are evidence about that graph, never an alternate
    or relaxed partitioning rule.
    """
    missing: list[str] = []
    missing_rows: list[dict] = []
    values_by_row: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        values = {name: AXES[name](row) for name in FORMAL_ISOLATION_AXES}
        absent = [name for name, value in values.items() if not value]
        if absent:
            ident = source_sample_id(row) or f"row{index}"
            missing.append(f"{ident}:{','.join(absent)}")
            missing_rows.append({"sample_id": ident, "missing_axes": absent})
        values_by_row.append(values)

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
        # Missing metadata is reported above and invalidates formal partitioning.
        # It must not accidentally connect two incomplete rows through "".
        for axis, value in values.items():
            if not value:
                continue
            key = (axis, value)
            if key in owner:
                union(index, owner[key])
            else:
                owner[key] = index

    grouped: dict[int, list[int]] = collections.defaultdict(list)
    for index in range(len(rows)):
        grouped[find(index)].append(index)
    groups = list(grouped.values())
    groups.sort(key=lambda indices: _component_sort_key(rows, indices))

    component_ids: dict[int, str] = {}
    for indices in groups:
        digest, _first = _component_sort_key(rows, indices)
        component_id = f"component-{digest[:16]}"
        for index in indices:
            component_ids[index] = component_id

    templates: dict[str, list[int]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        template = template_of(row)
        if template:
            templates[template].append(index)

    bridge_details: list[dict] = []
    for template, indices in sorted(templates.items()):
        environments = _axis_distribution(rows, indices, "environment")
        sites = _axis_distribution(rows, indices, "site")
        non_missing_environments = {
            value for value in environments if value != "<missing>"}
        non_missing_sites = {value for value in sites if value != "<missing>"}
        cross_environment = len(non_missing_environments) > 1
        cross_site = len(non_missing_sites) > 1
        if not (cross_environment or cross_site):
            continue
        bridge_details.append({
            "goal_template": template,
            "n_rows": len(indices),
            "environments": environments,
            "sites": sites,
            "cross_environment": cross_environment,
            "cross_site": cross_site,
            "component_ids": sorted({component_ids[index] for index in indices}),
        })

    cross_environment_values = [
        detail["goal_template"] for detail in bridge_details
        if detail["cross_environment"]]
    cross_site_values = [
        detail["goal_template"] for detail in bridge_details
        if detail["cross_site"]]

    components: list[dict] = []
    for indices in groups:
        identities = sorted(
            source_sample_id(rows[index]) or f"row{index}" for index in indices)
        distributions = {
            axis: _axis_distribution(rows, indices, axis)
            for axis in _COMPONENT_DIAGNOSTIC_AXES
        }
        shared_isolation_values = {
            axis: [
                {"value": value, "n_rows": count}
                for value, count in distributions[axis].items()
                if value != "<missing>" and count > 1
            ]
            for axis in FORMAL_ISOLATION_AXES
        }
        shared_isolation_values = {
            axis: values for axis, values in shared_isolation_values.items()
            if values
        }
        component_id = component_ids[indices[0]]
        component_bridges = [
            detail["goal_template"] for detail in bridge_details
            if component_id in detail["component_ids"]]
        components.append({
            "component_id": component_id,
            "n_rows": len(indices),
            "sample_ids_sha256": hashlib.sha256(
                "\n".join(identities).encode("utf-8")).hexdigest(),
            "sample_id_preview": identities[:20],
            "distributions": distributions,
            "merge_evidence": {
                "shared_isolation_values": shared_isolation_values,
                "cross_environment_or_site_goal_templates": component_bridges,
            },
        })

    if not rows:
        explanation = "formal source contains zero rows"
    elif missing:
        explanation = (
            "formal isolation metadata is incomplete; components are diagnostic "
            "only and partitioning remains unavailable")
    elif len(groups) == 1:
        explanation = (
            "all rows are transitively connected by one or more formal isolation "
            "values; shared goal-template bridges are listed explicitly")
    else:
        explanation = (
            f"the source contains {len(groups)} indivisible components under the "
            "unchanged formal isolation axes")

    diagnostics = {
        "schema_version": FORMAL_JOINT_GRAPH_SCHEMA_VERSION,
        "isolation_axes": list(FORMAL_ISOLATION_AXES),
        "diagnostic_axes": list(_COMPONENT_DIAGNOSTIC_AXES),
        "n_rows": len(rows),
        "n_components": len(groups),
        "partition_metadata_complete": not missing,
        "missing_isolation_metadata": missing_rows,
        "components": components,
        "goal_template_bridges": {
            "cross_environment_values": cross_environment_values,
            "cross_site_values": cross_site_values,
            "details": bridge_details,
        },
        "explanation": explanation,
    }
    return groups, missing, diagnostics


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
) -> tuple[list[dict], list[dict], list[str], str, dict]:
    """Partition joint leakage components for a formal publication split.

    Every row must expose every isolation axis.  Rows are connected when they
    share *any* state group, canonical entity, goal/page template, or
    environment.  A connected component is indivisible, which is the only way
    to make overlap zero on all four axes simultaneously.  If the metadata
    graph has only one component, the requested generalisation experiment is
    unavailable; returning the input as a nominal train split would hide that
    fact, so this function deliberately returns two empty sides.
    """
    groups, missing, diagnostics = _formal_joint_graph(rows)
    if not rows:
        return [], [], [], "formal source contains zero rows", diagnostics
    if missing:
        preview = "; ".join(missing[:10])
        return [], [], [], (
            "formal isolation metadata is missing "
            f"(sample:axes): {preview}"), diagnostics
    if len(groups) < 2:
        return [], [], [], (
            "joint state/entity/goal-template/page-template/environment graph has one connected "
            "component; a non-leaking train/test split is unavailable"), diagnostics

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
        return [], [], [], (
            f"strict formal partition invariant failed: {detail}"), diagnostics
    return train, test, held, "", diagnostics


def formal_validation_partition(
    rows: list[dict], val_frac: float,
) -> tuple[list[dict], list[dict], dict]:
    """Create an optional strict validation side from an existing formal train.

    A one-component training set can still be released for fitting, but it
    cannot honestly claim an independent validation split.  In that case all
    rows remain in train, validation is empty, and the structured report marks
    the split unavailable rather than leaking a shared template/environment.
    """
    _groups, _missing, joint_graph = _formal_joint_graph(rows)
    if val_frac <= 0:
        return list(rows), [], {
            "available": False, "reason": "validation split disabled",
            "n_train": len(rows), "n_val": 0,
            "leakage": audit_split_leakage(rows, []),
            "joint_graph": joint_graph,
        }
    train, val, held, reason, joint_graph = _strict_formal_partition(
        rows, val_frac)
    if reason:
        return list(rows), [], {
            "available": False, "reason": reason,
            "n_train": len(rows), "n_val": 0,
            "leakage": audit_split_leakage(rows, []),
            "joint_graph": joint_graph,
        }
    return train, val, {
        "available": True, "reason": "", "held_components": held,
        "n_train": len(train), "n_val": len(val),
        "leakage": audit_split_leakage(train, val),
        "joint_graph": joint_graph,
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


def _strict_challenge_partition(
        rows: list[dict], axis: Callable[[dict], str], frac: float
        ) -> tuple[list[dict], list[dict], list[str]]:
    """Hold out one challenge axis and drop every leaking train component.

    A base train/dev/test split may be unavailable even though an independently
    auditable leave-one-site/action challenge exists.  Start from the requested
    held axis, then retain a whole train state group only when none of its
    formal isolation values appears in the test side.  Dropped rows are
    reported by the resulting counts; they are never reassigned across sides.
    """
    values = sorted({axis(row) for row in rows if axis(row)})
    if len(values) < 2:
        return [], [], []
    ranked = sorted(values, key=lambda value: (
        len({state_group_of(row) for row in rows if axis(row) == value}), value))
    n_test = max(1, int(len(values) * frac))
    n_test = min(n_test, len(values) - 1)
    held = set(ranked[:n_test])
    test = [row for row in rows if axis(row) in held]
    forbidden = {
        name: {AXES[name](row) for row in test if AXES[name](row)}
        for name in FORMAL_ISOLATION_AXES
    }
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        if axis(row) not in held:
            grouped[state_group_of(row)].append(row)
    train: list[dict] = []
    for group in grouped.values():
        leaks = any(
            AXES[name](row) and AXES[name](row) in forbidden[name]
            for row in group for name in FORMAL_ISOLATION_AXES
        )
        if not leaks:
            train.extend(group)
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
    joint_graph = None
    if formal:
        _groups, _missing, joint_graph = _formal_joint_graph(sft)
        non_formal = [source_sample_id(row) for row in sft
                      if (row.get("meta") or {}).get("formal_dataset") is not True]
        if non_formal:
            reason = ("formal source mixes rows without formal_dataset=true: "
                      + ", ".join((item or "<missing-id>")
                                  for item in non_formal[:10]))
            train, test, held = [], [], []
        else:
            train, test, held, reason, joint_graph = _strict_formal_partition(
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
        "joint_graph": None,
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
        partition = _strict_challenge_partition if formal else _challenge_partition
        ctrain, ctest, cheld = partition(sft, axis, holdout_frac)
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
        "joint_graph": joint_graph,
        "validation": validation,
        "challenges": challenges,
    }


def build_splits(sft_path: Path, dpo_path: Path, out_dir: Path,
                 holdout_frac: float = 0.25,
                 multiturn_sft_path: Path | None = None,
                 multiturn_dpo_path: Path | None = None,
                 additional_dpo_paths: tuple[Path, ...] = (),
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
    normalized_additional_paths = tuple(Path(path) for path in additional_dpo_paths)
    if len({path.resolve() for path in normalized_additional_paths}) != \
            len(normalized_additional_paths):
        raise ValueError("duplicate additional DPO supplement path")
    if normalized_additional_paths and not formal:
        raise ValueError("additional DPO artifacts are formal-only")
    additional_releases: list[dict] = []
    additional_dpo: list[dict] = []
    for path in normalized_additional_paths:
        rows, manifest = load_formal_dpo_supplement(path)
        additional_dpo.extend(rows)
        additional_releases.append({
            "path": str(path),
            "manifest_path": str(formal_dpo_supplement_manifest_path(path)),
            "release_id": manifest["release_id"],
            "body_sha256": manifest["body_sha256"],
            "n_rows": manifest["n_rows"],
        })

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
        formal_source_ids = single_ids | multi_ids
        unknown_supplement_sources = sorted({
            source_sample_id(row) for row in additional_dpo
            if source_sample_id(row) not in formal_source_ids})
        if unknown_supplement_sources:
            raise ValueError(
                "additional DPO supplement references unknown formal SFT sources: " +
                ", ".join(unknown_supplement_sources[:10]))
        dpo.extend(additional_dpo)
        pair_ids = [str(row.get("pair_id") or "") for row in dpo + mt_dpo]
        if any(not pair_id for pair_id in pair_ids) or \
                len(pair_ids) != len(set(pair_ids)):
            raise ValueError("formal DPO rows require unique non-empty pair_id")
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
        "schema_version": SPLIT_REPORT_SCHEMA_VERSION,
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
        "joint_graph": single_report["joint_graph"],
        "validation": single_report["validation"],
        "challenges": single_report["challenges"],
        # Backward-compatible report field, now honest when unavailable.
        "cross_site": single_report["challenges"]["cross_site"],
        "multiturn": multi_report,
        "additional_dpo_paths": [str(path) for path in normalized_additional_paths],
        "additional_dpo_releases": additional_releases,
        "n_additional_dpo": len(additional_dpo),
        "out_dir": str(out_dir),
    }
    # Keep an auditable, machine-readable verdict beside the generated files.
    # Distribution uses tuple keys for backward-compatible Python callers, so
    # the persisted report intentionally contains only JSON-native fields.
    persisted = {
        key: report[key] for key in (
            "schema_version", "available", "formal", "reason", "n_train",
            "n_dev", "n_test",
            "n_dpo_train", "test_slugs", "by_site", "leakage",
            "joint_graph", "validation", "challenges", "cross_site", "multiturn",
            "additional_dpo_paths", "additional_dpo_releases",
            "n_additional_dpo", "out_dir")
    }
    (out_dir / "SPLIT_REPORT.json").write_text(
        json.dumps(persisted, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report
