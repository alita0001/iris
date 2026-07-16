"""Task-independent, fixture-safe state-mutation discovery.

The miner does not know about ``add_to_cart`` or any other action class.  It
enumerates every legal accessibility control at a resettable state, executes
each through a caller-supplied *fixture/dry-run* executor, compares persistent
UI/API/DB/external signals, and restores the anchor state after every trial.

Keyword and LLM components may only reorder the full control set.  Returning a
subset is rejected because ranking must never become a hidden ground-truth
filter.  This module intentionally refuses live executors; a separately
reviewed approval gate is required before production execution is introduced.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
from urllib.parse import urlsplit


MUTATION_SAMPLE_SCHEMA_VERSION = "iris.mutation_miner.live_sample.v1"
MUTATION_REPORT_SCHEMA_VERSION = "iris.mutation_miner.live_report.v3"
MUTATION_MANIFEST_SCHEMA_VERSION = \
    "iris.mutation_miner.live_report_manifest.v3"
MUTATION_REPORT_BODY_NAME = "mutation-miner-live-report.v3.json"
MUTATION_REPORT_MANIFEST_NAME = \
    "MUTATION_MINER_LIVE_REPORT.v3.manifest.json"
MUTATION_CENSUS_MIN_CONTROLS = 200
MUTATION_PREFLIGHT_SCHEMA_VERSION = "iris.mutation_miner.preflight.v1"
MUTATION_SNAPSHOT_SCHEMA_VERSION = "iris.mutation_miner.full_snapshot.v1"
MUTATION_SNAPSHOT_MANIFEST_SCHEMA_VERSION = \
    "iris.mutation_miner.full_snapshot_manifest.v1"

_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_SIGNAL_CHANNELS = frozenset({"ui", "api", "db", "external"})
_SAFETY_CLASSES = frozenset({"non_destructive", "destructive"})
_REFERENCE_SOURCES = {
    "independent_api_diff": "api",
    "independent_db_diff": "db",
    "independent_external_receipt": "external",
    "independent_manual_replay": None,
}

# These strings prioritize *human review* only.  They never declare a control
# safe, mutated, or recovered and never enter the live-census truth fields.
_ELEVATED_REVIEW_SURFACES = (
    "ban", "delete", "email", "pay", "place order", "publish", "refund",
    "remove account", "send", "submit order",
)
_SELF_RECOVERING_REVIEW_SURFACES = (
    "add to cart", "add to compare", "add to wish list", "downvote",
    "subscribe", "upvote",
)


class MutationMiningError(RuntimeError):
    """Raised when reset, enumeration, or safety invariants are violated."""


@dataclass(frozen=True)
class InteractiveControl:
    bid: str
    canonical_action: str
    role: str = ""
    name: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalSnapshot:
    """Signals ordered from UI-visible to stronger persistent side effects."""

    ui: Any = None
    api: Any = None
    db: Any = None
    external: Any = None
    async_pending: bool = False


@dataclass(frozen=True)
class SignalDiff:
    ui_changed: bool
    api_changed: bool
    db_changed: bool
    external_changed: bool
    async_pending: bool

    @property
    def changed_channels(self) -> tuple[str, ...]:
        return tuple(name.removesuffix("_changed") for name, changed in (
            ("ui_changed", self.ui_changed),
            ("api_changed", self.api_changed),
            ("db_changed", self.db_changed),
            ("external_changed", self.external_changed),
        ) if changed)

    @property
    def any_persistent_change(self) -> bool:
        return bool(self.changed_channels)


def diff_signals(before: SignalSnapshot, after: SignalSnapshot) -> SignalDiff:
    return SignalDiff(
        ui_changed=before.ui != after.ui,
        api_changed=before.api != after.api,
        db_changed=before.db != after.db,
        external_changed=before.external != after.external,
        async_pending=bool(before.async_pending or after.async_pending),
    )


class SafeMutationExecutor(Protocol):
    """Explicitly injectable executor; implementations own their fixtures."""

    safety_mode: str

    def reset_to_anchor(self) -> None: ...

    def enumerate_interactive_controls(self) -> Sequence[InteractiveControl]: ...

    def capture_signals(self) -> SignalSnapshot: ...

    def execute_control(self, control: InteractiveControl) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class MutationTrial:
    bid: str
    canonical_action: str
    before: SignalSnapshot
    after: SignalSnapshot
    after_reset: SignalSnapshot
    signal_diff: SignalDiff
    mutation_candidate: bool
    reset_restored: bool
    execution_evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MutationMiner:
    """Enumerate -> execute -> diff -> reset, without action-class keywords."""

    SAFE_MODES = frozenset({"fixture", "dry_run"})

    def __init__(
        self,
        executor: SafeMutationExecutor,
        *,
        ranker: Callable[[Sequence[InteractiveControl]], Sequence[InteractiveControl]] | None = None,
    ) -> None:
        if getattr(executor, "safety_mode", "") not in self.SAFE_MODES:
            raise MutationMiningError(
                "mutation miner refuses non-fixture/non-dry-run executors; "
                "live execution requires a separately reviewed approval gate")
        self.executor = executor
        self.ranker = ranker

    def _rank_without_filtering(
        self, controls: Sequence[InteractiveControl],
    ) -> list[InteractiveControl]:
        original = list(controls)
        if not self.ranker:
            return original
        ranked = list(self.ranker(tuple(original)))
        original_ids = [(c.bid, c.canonical_action) for c in original]
        ranked_ids = [(c.bid, c.canonical_action) for c in ranked]
        if len(ranked_ids) != len(original_ids) or sorted(ranked_ids) != sorted(original_ids):
            raise MutationMiningError(
                "ranker must return a permutation of every enumerated control; "
                "keywords/LLMs may rank but may not filter")
        return ranked

    def mine(self) -> list[MutationTrial]:
        self.executor.reset_to_anchor()
        anchor = self.executor.capture_signals()
        controls = self._rank_without_filtering(
            self.executor.enumerate_interactive_controls())
        identities = [(c.bid, c.canonical_action) for c in controls]
        if len(identities) != len(set(identities)):
            raise MutationMiningError("enumerator returned duplicate controls")

        trials: list[MutationTrial] = []
        for control in controls:
            self.executor.reset_to_anchor()
            before = self.executor.capture_signals()
            if before != anchor:
                raise MutationMiningError(
                    f"anchor drift before [{control.bid}]; refusing further execution")
            evidence = dict(self.executor.execute_control(control))
            after = self.executor.capture_signals()
            signal_diff = diff_signals(before, after)
            self.executor.reset_to_anchor()
            after_reset = self.executor.capture_signals()
            restored = after_reset == anchor
            trial = MutationTrial(
                bid=control.bid,
                canonical_action=control.canonical_action,
                before=before,
                after=after,
                after_reset=after_reset,
                signal_diff=signal_diff,
                mutation_candidate=signal_diff.any_persistent_change,
                reset_restored=restored,
                execution_evidence=evidence,
            )
            trials.append(trial)
            if not restored:
                raise MutationMiningError(
                    f"reset failed after [{control.bid}]; stopping to avoid state contamination")
        return trials


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    """Two-sided Wilson score interval (95% by default), stdlib-only."""
    if successes < 0 or total < 0 or successes > total:
        raise ValueError("Wilson counts require 0 <= successes <= total")
    if total == 0:
        return 0.0, 1.0
    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        (p * (1.0 - p) + z2 / (4.0 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def mutation_detection_report(
    trials: Sequence[MutationTrial], ground_truth_mutated_bids: set[str],
) -> dict[str, Any]:
    """Fixture precision/recall with counts and Wilson intervals."""
    detected = {t.bid for t in trials if t.mutation_candidate}
    enumerated = {t.bid for t in trials}
    outside = set(ground_truth_mutated_bids) - enumerated
    if outside:
        raise MutationMiningError(
            f"ground-truth controls were not enumerated: {sorted(outside)}")
    true_positive = detected & ground_truth_mutated_bids
    false_positive = detected - ground_truth_mutated_bids
    false_negative = ground_truth_mutated_bids - detected
    precision_n = len(detected)
    recall_n = len(ground_truth_mutated_bids)
    precision = len(true_positive) / precision_n if precision_n else 0.0
    recall = len(true_positive) / recall_n if recall_n else 0.0
    return {
        "enumerated_n": len(enumerated),
        "detected_n": len(detected),
        "ground_truth_positive_n": recall_n,
        "true_positive_bids": sorted(true_positive),
        "false_positive_bids": sorted(false_positive),
        "false_negative_bids": sorted(false_negative),
        "precision": precision,
        "precision_wilson_95": list(wilson_interval(len(true_positive), precision_n)),
        "recall": recall,
        "recall_wilson_95": list(wilson_interval(len(true_positive), recall_n)),
    }


def signal_sufficiency_audit(
    before: SignalSnapshot, after: SignalSnapshot,
) -> dict[str, Any]:
    """Expose UI-vs-API/DB agreement, false positives, and blind spots.

    API and DB are stronger observations of application state, while external
    side effects form a separate channel that may remain even after a DB reset.
    Async pending marks the result inconclusive rather than treating delayed
    consistency as evidence of no effect.
    """
    diff = diff_signals(before, after)
    backend_changed = diff.api_changed or diff.db_changed
    ui_backend_agree = diff.ui_changed == backend_changed
    blind_spots: list[str] = []
    if backend_changed and not diff.ui_changed:
        blind_spots.append("ui_missed_backend_change")
    if diff.ui_changed and not backend_changed:
        blind_spots.append("ui_only_transient_or_backend_blind_spot")
    if diff.external_changed and not backend_changed:
        blind_spots.append("external_side_effect_not_reflected_in_api_or_db")
    if diff.async_pending:
        blind_spots.append("async_state_unsettled")
    return {
        "changed_channels": list(diff.changed_channels),
        "ui_changed": diff.ui_changed,
        "api_changed": diff.api_changed,
        "db_changed": diff.db_changed,
        "external_changed": diff.external_changed,
        "ui_backend_agree": ui_backend_agree,
        "ui_vs_api_agree": diff.ui_changed == diff.api_changed,
        "ui_vs_db_agree": diff.ui_changed == diff.db_changed,
        "blind_spots": blind_spots,
        "conclusive": not diff.async_pending,
    }


# ---------------------------------------------------------------------------
# Read-only live-census preflight
# ---------------------------------------------------------------------------


def _origin(url: str) -> str:
    parsed = urlsplit(str(url or ""))
    return (f"{parsed.scheme}://{parsed.netloc}"
            if parsed.scheme in {"http", "https"} and parsed.netloc else "")


def _snapshot_manifest_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": MUTATION_SNAPSHOT_MANIFEST_SCHEMA_VERSION,
        "state_id": row["state_id"],
        "snapshot_hash": row["snapshot_hash"],
        "record_sha256": _json_sha256(dict(row)),
    }


def save_full_mutation_snapshots(
    rows: Iterable[Mapping[str, Any]], *, body: Path, manifest: Path,
) -> str:
    """Freeze dedicated unpruned AX snapshots with a 1:1 hash manifest."""
    materialized = sorted((dict(row) for row in rows), key=lambda row: row["state_id"])
    if not materialized:
        raise MutationMiningError("cannot save zero full mutation snapshots")
    identities: set[str] = set()
    for row in materialized:
        if row.get("schema_version") != MUTATION_SNAPSHOT_SCHEMA_VERSION:
            raise MutationMiningError("bad full mutation snapshot schema_version")
        state_id = str(row.get("state_id") or "")
        snapshot = str(row.get("axtree_snapshot") or "")
        if not state_id or state_id in identities:
            raise MutationMiningError("full mutation snapshot state_id must be unique")
        identities.add(state_id)
        if row.get("axtree_complete") is not True or not snapshot:
            raise MutationMiningError("full mutation snapshot lacks completeness attestation")
        if "... (axtree truncated)" in snapshot:
            raise MutationMiningError("full mutation snapshot contains truncation marker")
        if row.get("snapshot_hash") != hashlib.sha256(snapshot.encode()).hexdigest():
            raise MutationMiningError("full mutation snapshot hash mismatch")
        if row.get("navigation_only_capture") is not True:
            raise MutationMiningError("snapshot capture must be navigation-only")
    body_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                        for row in materialized)
    manifest_text = "".join(json.dumps(
        _snapshot_manifest_row(row), ensure_ascii=False, sort_keys=True) + "\n"
        for row in materialized)
    body, manifest = Path(body), Path(manifest)
    if body.exists() != manifest.exists():
        raise MutationMiningError(
            "full snapshot body/manifest must both exist or both be absent")
    if body.exists():
        if (body.read_text(encoding="utf-8") != body_text or
                manifest.read_text(encoding="utf-8") != manifest_text):
            raise MutationMiningError(
                f"refusing to overwrite full mutation snapshot asset {body}")
        return "already_identical"
    body.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    body_fd = os.open(body, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(body_fd, "w", encoding="utf-8") as handle:
            handle.write(body_text)
            handle.flush()
            os.fsync(handle.fileno())
        manifest_fd = os.open(
            manifest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(manifest_fd, "w", encoding="utf-8") as handle:
            handle.write(manifest_text)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        if body.exists() and not manifest.exists():
            body.unlink()
        raise
    return "created"


def capture_full_mutation_snapshots(
    targets: Sequence[tuple[str, str]], *, body: Path, manifest: Path,
    code_version: str, headless: bool = True,
    env_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Navigate only to registered sandbox URLs and capture the full AXTree.

    This function performs no candidate action, label inference, or mutation.
    It exists because policy-pruned state banks cannot define a complete-control
    recall denominator.  Each target uses the registered site's authenticated
    session task and is constrained to that site's configured URL origin.
    """
    if not targets:
        raise MutationMiningError("at least one site=url target is required")
    if not str(code_version or "").strip():
        raise MutationMiningError("code_version is required")
    from .. import config
    from ..envs.harness import RevActEnv, make_env
    from .candidates import interactive_elements

    factory = env_factory or make_env
    rows: list[dict[str, Any]] = []
    captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    seen_targets: set[tuple[str, str]] = set()
    for site, url in targets:
        site, url = str(site).strip(), str(url).strip()
        spec = config.SITES.get(site)
        if spec is None:
            raise MutationMiningError(f"unknown registered site {site!r}")
        configured_base = config.site_base(site)
        if not configured_base or not _origin(configured_base):
            raise MutationMiningError(f"site {site!r} has no configured sandbox URL")
        if _origin(url) != _origin(configured_base):
            raise MutationMiningError(
                f"target URL is outside registered sandbox origin for {site}")
        target_identity = (site, url)
        if target_identity in seen_targets:
            raise MutationMiningError(f"duplicate full snapshot target {target_identity}")
        seen_targets.add(target_identity)

        env = factory(spec.session_task, headless=headless)
        renv = RevActEnv(env, task_id=spec.session_task, site=site)
        trajectory_id = (
            "mutation-fullax-" + hashlib.sha256(
                f"{site}\0{url}\0{captured_at}".encode()).hexdigest()[:16])
        try:
            renv.reset(
                seed=0, trajectory_id=trajectory_id,
                run_id=trajectory_id, code_version=code_version)
            raw_action = f"goto({url!r})"
            renv.step(raw_action)
            observed_url = str(renv._last_obs_view.get("url") or "")
            if _origin(observed_url) != _origin(configured_base):
                raise MutationMiningError(
                    f"navigation left registered sandbox origin: {observed_url}")
            snapshot = str(renv._last_obs_view.get("axtree_txt") or "")
            if not snapshot:
                raise MutationMiningError(f"empty full AXTree at {url}")
            if "... (axtree truncated)" in snapshot:
                raise MutationMiningError(
                    f"environment returned an already-truncated AXTree at {url}")
            elements = [
                element for element in interactive_elements(snapshot)
                if not re.search(
                    r"(?:^|[,\s])(?:aria-)?disabled\s*=\s*true(?:[,\s]|$)",
                    str(element.get("line") or "").lower())]
            identities = [(element["bid"], element["role"], element["name"])
                          for element in elements]
            if not elements or len(identities) != len(set(identities)):
                raise MutationMiningError(
                    f"full AXTree has zero or duplicate interactive controls at {url}")
            snapshot_hash = hashlib.sha256(snapshot.encode()).hexdigest()
            state_id = "mutation-fullax-" + _json_sha256({
                "site": site, "url": observed_url,
                "snapshot_hash": snapshot_hash,
            })[:20]
            environment_instance = _origin(observed_url)
            rows.append({
                "schema_version": MUTATION_SNAPSHOT_SCHEMA_VERSION,
                "state_id": state_id,
                "task_id": spec.session_task,
                "site": site,
                "environment_family": spec.environment_family,
                "environment_instance": environment_instance,
                "environment_origin": "webarena",
                "is_mock": False,
                "trajectory_id": trajectory_id,
                "run_id": trajectory_id,
                "seed": 0,
                "url": observed_url,
                "requested_url": url,
                "replay_prefix": [raw_action],
                "axtree_snapshot": snapshot,
                "axtree_complete": True,
                "capture_protocol": "browsergym_full_axtree_navigation_only.v1",
                "navigation_only_capture": True,
                "snapshot_hash": snapshot_hash,
                "interactive_control_count": len(elements),
                "control_set_sha256": mutation_control_set_sha256(
                    (element["bid"], f"click('{element['bid']}')")
                    for element in elements),
                "captured_at": captured_at,
                "code_version": code_version,
            })
        finally:
            try:
                env.close()
            except Exception:
                pass
    status = save_full_mutation_snapshots(rows, body=body, manifest=manifest)
    return {
        "schema_version": "iris.mutation_miner.full_snapshot_report.v1",
        "status": status,
        "body": str(body),
        "manifest": str(manifest),
        "n_snapshots": len(rows),
        "n_controls": sum(int(row["interactive_control_count"]) for row in rows),
        "sites": dict(sorted(Counter(row["site"] for row in rows).items())),
        "state_ids": sorted(row["state_id"] for row in rows),
        "navigation_only_capture": True,
        "counts_as_live_mutation_execution": False,
        "code_version": code_version,
    }


def _environment_instance_from_state(row: Mapping[str, Any]) -> str:
    """Return a physical endpoint identity, preferring the recorded URL.

    Legacy state banks often contain ``site='shopping'`` while newer rows use
    the concrete WebArena endpoint.  Treating those strings as different
    environments would pad a physical-snapshot census, so URL origin wins.
    """
    parsed = urlsplit(str(row.get("url") or ""))
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return str(row.get("environment_instance") or row.get("site") or "").strip()


def _review_surface(name: str, role: str) -> str:
    """Prioritize review without asserting safety or mutation ground truth."""
    normalized = re.sub(r"[^a-z0-9]+", " ", f"{name} {role}".lower()).strip()
    if any(re.search(rf"(?:^| ){' '.join(map(re.escape, phrase.split()))}(?: |$)",
                     normalized)
           for phrase in _ELEVATED_REVIEW_SURFACES):
        return "elevated_surface_review"
    if any(re.search(rf"(?:^| ){' '.join(map(re.escape, phrase.split()))}(?: |$)",
                     normalized)
           for phrase in _SELF_RECOVERING_REVIEW_SURFACES):
        return "self_recovering_surface_review"
    return "unclassified_review"


def build_mutation_census_preflight(
    data_root: Path, *, minimum_controls: int = MUTATION_CENSUS_MIN_CONTROLS,
    code_version: str,
) -> dict[str, Any]:
    """Inventory and select complete physical snapshots for live review.

    No browser action is executed.  Every control remains ``PENDING`` and the
    result explicitly does not count toward the release gate.  The preflight
    closes an engineering gap: reviewers receive exact replay contracts and a
    complete, alias-deduplicated control list instead of a keyword-selected
    subset.
    """
    if isinstance(minimum_controls, bool) or not isinstance(
            minimum_controls, int) or minimum_controls <= 0:
        raise MutationMiningError("minimum_controls must be a positive integer")
    if not str(code_version or "").strip():
        raise MutationMiningError("code_version is required")

    from .candidates import interactive_elements, snapshot_sha256

    root = Path(data_root)
    state_bank = root / "raw" / "state_bank"
    physical: dict[tuple[str, str], dict[str, Any]] = {}
    invalid_rows: list[dict[str, Any]] = []
    excluded_mock_rows = 0
    for path in sorted(state_bank.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise MutationMiningError(
                        f"invalid state-bank JSON {path}:{line_no}: {exc}") from exc
                if (row.get("is_mock") is True or
                        str(row.get("environment_origin") or "").lower() == "mock" or
                        str(row.get("site") or "").lower() == "mock"):
                    excluded_mock_rows += 1
                    continue
                state_id = str(row.get("state_id") or "").strip()
                snapshot = str(row.get("axtree_snapshot") or "")
                environment_instance = _environment_instance_from_state(row)
                if not state_id or not snapshot or not environment_instance:
                    invalid_rows.append({
                        "path": str(path.relative_to(root)),
                        "line": line_no,
                        "reason": "missing state_id/axtree_snapshot/environment_instance",
                    })
                    continue
                snapshot_hash = snapshot_sha256(snapshot)
                controls = []
                for element in interactive_elements(snapshot):
                    line_text = str(element.get("line") or "").lower()
                    if re.search(
                            r"(?:^|[,\s])(?:aria-)?disabled\s*=\s*true(?:[,\s]|$)",
                            line_text):
                        continue
                    bid = str(element.get("bid") or "")
                    role = str(element.get("role") or "")
                    name = str(element.get("name") or "")
                    control_id = "control-" + _json_sha256({
                        "environment_instance": environment_instance,
                        "snapshot_hash": snapshot_hash,
                        "bid": bid,
                        "role": role,
                        "name": name,
                    })[:20]
                    controls.append({
                        "control_id": control_id,
                        "bid": bid,
                        "role": role,
                        "name": name,
                        "canonical_action": f"click('{bid}')",
                        "review_priority": _review_surface(name, role),
                        "review_status": "PENDING",
                        "approved_for_execution": False,
                        "counts_as_mutation_evidence": False,
                    })
                controls.sort(key=lambda item: (item["bid"], item["control_id"]))
                key = (environment_instance, snapshot_hash)
                control_signature = [
                    (item["bid"], item["role"], item["name"])
                    for item in controls]
                existing = physical.get(key)
                source = {
                    "path": str(path.relative_to(root)),
                    "line": line_no,
                    "state_id": state_id,
                    "task_id": str(row.get("task_id") or ""),
                    "trajectory_id": str(row.get("trajectory_id") or ""),
                    "run_id": str(row.get("run_id") or ""),
                    "seed": row.get("seed"),
                    "url": str(row.get("url") or ""),
                    "replay_prefix": list(row.get("replay_prefix") or []),
                    "axtree_complete": row.get("axtree_complete") is True,
                    "capture_protocol": str(row.get("capture_protocol") or ""),
                }
                if existing is None:
                    physical[key] = {
                        "environment_instance": environment_instance,
                        "snapshot_hash": snapshot_hash,
                        "state_aliases": [state_id],
                        "source_rows": [source],
                        "controls": controls,
                        "control_signature": control_signature,
                    }
                else:
                    if existing["control_signature"] != control_signature:
                        raise MutationMiningError(
                            "same physical snapshot hash has conflicting controls: "
                            f"{environment_instance} {snapshot_hash}")
                    existing["state_aliases"].append(state_id)
                    existing["source_rows"].append(source)

    inventory: list[dict[str, Any]] = []
    for item in physical.values():
        item["state_aliases"] = sorted(set(item["state_aliases"]))
        item["source_rows"] = sorted(
            item["source_rows"],
            key=lambda source: (source["path"], source["line"]))
        review_counts = Counter(
            control["review_priority"] for control in item["controls"])
        replay_sources = [source for source in item["source_rows"]
                          if source["task_id"] and source["url"] and
                          isinstance(source["seed"], int)]
        complete_replay_sources = [
            source for source in replay_sources if source["axtree_complete"]]
        preferred_replay = (
            complete_replay_sources[0] if complete_replay_sources else
            replay_sources[0] if replay_sources else None)
        preferred_url = urlsplit(str((preferred_replay or {}).get("url") or ""))
        canonical_page = (
            f"{preferred_url.scheme}://{preferred_url.netloc}{preferred_url.path}"
            if preferred_url.scheme and preferred_url.netloc else "")
        item.update({
            "canonical_state_id": item["state_aliases"][0],
            "n_controls": len(item["controls"]),
            "control_set_sha256": mutation_control_set_sha256(
                (control["bid"], control["canonical_action"])
                for control in item["controls"]),
            "review_priority_counts": dict(sorted(review_counts.items())),
            "all_controls_reviewed": False,
            "replay_contract_available": bool(replay_sources),
            "snapshot_projection_complete": bool(complete_replay_sources),
            "preferred_replay_source": preferred_replay,
            "canonical_page": canonical_page,
        })
        item.pop("control_signature")
        inventory.append(item)

    # Prefer replayable snapshots with no elevated surface, then minimize the
    # number of browser resets by taking larger complete control sets.  This is
    # scheduling only; every selected control still requires explicit review.
    eligible = [item for item in inventory
                if item["replay_contract_available"] and
                item["snapshot_projection_complete"]]
    eligible.sort(key=lambda item: (
        int(item["review_priority_counts"].get("elevated_surface_review", 0) > 0),
        item["review_priority_counts"].get("elevated_surface_review", 0),
        -item["n_controls"],
        item["environment_instance"], item["snapshot_hash"],
    ))
    selected: list[dict[str, Any]] = []
    selected_n = 0
    selected_pages: set[str] = set()
    for item in eligible:
        if selected_n >= minimum_controls:
            break
        # Dynamic counters and regenerated AX bids can change a snapshot hash
        # without creating a new page/control surface.  Prefer one complete
        # snapshot per canonical URL path so the review plan cannot reach 200
        # by replaying the same address or product page several times.
        canonical_page = str(item.get("canonical_page") or "")
        if canonical_page and canonical_page in selected_pages:
            continue
        selected.append(item)
        selected_n += item["n_controls"]
        if canonical_page:
            selected_pages.add(canonical_page)

    aggregate_review = Counter(
        control["review_priority"]
        for item in selected for control in item["controls"])
    inventory_digest = _json_sha256([{
        "environment_instance": item["environment_instance"],
        "snapshot_hash": item["snapshot_hash"],
        "state_aliases": item["state_aliases"],
        "control_set_sha256": item["control_set_sha256"],
        "n_controls": item["n_controls"],
    } for item in sorted(inventory, key=lambda row: (
        row["environment_instance"], row["snapshot_hash"]))])
    return {
        "schema_version": MUTATION_PREFLIGHT_SCHEMA_VERSION,
        "code_version": str(code_version),
        "minimum_controls": minimum_controls,
        "state_bank_inventory_sha256": inventory_digest,
        "state_bank_jsonl_n": len(list(state_bank.glob("*.jsonl"))),
        "invalid_state_rows": invalid_rows,
        "excluded_mock_rows": excluded_mock_rows,
        "unique_physical_snapshot_n": len(inventory),
        "enumerable_control_n": sum(item["n_controls"] for item in inventory),
        "replayable_physical_snapshot_n": sum(
            item["replay_contract_available"] for item in inventory),
        "complete_physical_snapshot_n": len(eligible),
        "complete_enumerable_control_n": sum(
            item["n_controls"] for item in eligible),
        "selected_snapshot_n": len(selected),
        "selected_unique_page_n": len(selected_pages),
        "selected_duplicate_canonical_page_n": (
            len(selected) - len(selected_pages)),
        "selected_enumerated_control_n": selected_n,
        "selection_reaches_minimum": selected_n >= minimum_controls,
        "selected_review_priority_counts": dict(sorted(aggregate_review.items())),
        "selected_snapshots": selected,
        "executed_live_n": 0,
        "independent_reference_n": 0,
        "all_controls_reviewed": False,
        "counts_as_live_census": False,
        "gate_satisfied": False,
        "blockers": [
            "selected snapshots must be captured unpruned with axtree_complete=true",
            "every selected control requires explicit non-destructive/destructive review",
            "every control requires live execution with reset_restored=true",
            "every detector result requires an independent reference measurement",
            "only complete physical-snapshot control groups may enter the release census",
        ],
        "interpretation": (
            "read-only replay plan; review_priority is a heuristic queue only, "
            "not effect/recovery/safety evidence"),
    }


def save_mutation_census_preflight(report: Mapping[str, Any], path: Path) -> str:
    """Persist an immutable preflight; an exact rerun is idempotent."""
    if report.get("schema_version") != MUTATION_PREFLIGHT_SCHEMA_VERSION:
        raise MutationMiningError("bad mutation preflight schema_version")
    text = json.dumps(
        dict(report), ensure_ascii=False, indent=2, sort_keys=True,
        allow_nan=False) + "\n"
    path = Path(path)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise MutationMiningError(
                f"refusing to overwrite mutation census preflight {path}")
        return "already_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    return "created"


# ---------------------------------------------------------------------------
# Immutable live-census artifact
# ---------------------------------------------------------------------------


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def mutation_control_set_sha256(
    controls: Iterable[tuple[str, str]],
) -> str:
    """Hash the exact bid/action set enumerated at one recorded snapshot."""
    rows = sorted(f"{bid}\t{action}" for bid, action in controls)
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def make_mutation_sample_id(
    *, collection_run_id: str, environment_instance: str, state_id: str,
    snapshot_hash: str, control_id: str, bid: str, canonical_action: str,
) -> str:
    """Derive the immutable identity; callers cannot pad by renaming a row."""
    payload = {
        "collection_run_id": collection_run_id,
        "environment_instance": environment_instance,
        "state_id": state_id,
        "snapshot_hash": snapshot_hash,
        "control_id": control_id,
        "bid": bid,
        "canonical_action": canonical_action,
    }
    return "mutation-" + _json_sha256(payload)[:24]


@dataclass(frozen=True)
class MutationCensusSample:
    """One executed control plus an independent mutation reference.

    This is deliberately not a grounding label.  It measures whether the
    state-change *detector* found a persistent change for one legal control.
    Precision/recall are always recomputed from ``mutation_candidate`` and
    ``reference_mutated``; no caller-supplied aggregate is trusted.
    """

    schema_version: str
    mutation_sample_id: str
    collection_run_id: str
    state_id: str
    control_id: str
    bid: str
    canonical_action: str
    role: str
    name: str
    environment_family: str
    environment_instance: str
    snapshot_hash: str
    snapshot_control_count: int
    control_set_sha256: str
    legal_at_snapshot: bool
    executed_live: bool
    is_fixture: bool
    safety_class: str
    review_status: str
    destructive_commit_authorized: bool
    destructive_env_gate: bool
    detector_channels: tuple[str, ...]
    changed_channels: tuple[str, ...]
    mutation_candidate: bool
    reference_mutated: bool
    reference_source: str
    reference_evidence_sha256: str
    execution_evidence_sha256: str
    pre_signal_sha256: str
    post_signal_sha256: str
    reset_signal_sha256: str
    reset_restored: bool
    async_pending: bool

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["detector_channels"] = list(self.detector_channels)
        row["changed_channels"] = list(self.changed_channels)
        return row

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "MutationCensusSample":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(row) - known)
        missing = sorted(known - set(row))
        if unknown:
            raise MutationMiningError(
                f"unknown mutation census sample fields: {unknown}")
        if missing:
            raise MutationMiningError(
                f"missing mutation census sample fields: {missing}")
        normalized = dict(row)
        for field_name in ("detector_channels", "changed_channels"):
            value = normalized[field_name]
            if not isinstance(value, (list, tuple)):
                raise MutationMiningError(f"{field_name} must be a list")
            normalized[field_name] = tuple(value)
        try:
            sample = cls(**normalized)
        except TypeError as exc:
            raise MutationMiningError(str(exc)) from exc
        sample.validate()
        return sample

    def validate(self) -> None:
        errors: list[str] = []
        if self.schema_version != MUTATION_SAMPLE_SCHEMA_VERSION:
            errors.append("bad sample schema_version")
        required = (
            "mutation_sample_id", "collection_run_id", "state_id",
            "control_id", "bid", "canonical_action", "role",
            "environment_family", "environment_instance", "snapshot_hash",
            "control_set_sha256", "reference_source",
            "reference_evidence_sha256", "execution_evidence_sha256",
            "pre_signal_sha256", "post_signal_sha256", "reset_signal_sha256",
        )
        for field_name in required:
            if not str(getattr(self, field_name) or "").strip():
                errors.append(f"missing {field_name}")
        for field_name in (
            "snapshot_hash", "control_set_sha256", "reference_evidence_sha256",
            "execution_evidence_sha256", "pre_signal_sha256",
            "post_signal_sha256", "reset_signal_sha256",
        ):
            if not _HEX_64.fullmatch(str(getattr(self, field_name) or "")):
                errors.append(f"bad {field_name}")
        expected_id = make_mutation_sample_id(
            collection_run_id=self.collection_run_id,
            environment_instance=self.environment_instance,
            state_id=self.state_id,
            snapshot_hash=self.snapshot_hash,
            control_id=self.control_id,
            bid=self.bid,
            canonical_action=self.canonical_action,
        )
        if self.mutation_sample_id != expected_id:
            errors.append("mutation_sample_id does not match immutable identity")
        if isinstance(self.snapshot_control_count, bool) or not isinstance(
                self.snapshot_control_count, int) or self.snapshot_control_count <= 0:
            errors.append("snapshot_control_count must be a positive integer")
        for field_name in (
            "legal_at_snapshot", "executed_live", "is_fixture",
            "destructive_commit_authorized", "destructive_env_gate",
            "mutation_candidate", "reference_mutated", "reset_restored",
            "async_pending",
        ):
            if not isinstance(getattr(self, field_name), bool):
                errors.append(f"{field_name} must be boolean")
        if self.legal_at_snapshot is not True:
            errors.append("control was not legal at snapshot")
        if self.executed_live is not True:
            errors.append("control lacks live execution")
        if self.is_fixture is not False:
            errors.append("fixture/unspecified sample cannot enter live census")
        if self.reset_restored is not True:
            errors.append("anchor was not restored after execution")
        if self.async_pending is not False:
            errors.append("async signal is unsettled")
        if self.pre_signal_sha256 != self.reset_signal_sha256:
            errors.append("reset signal does not equal pre signal")
        if self.safety_class not in _SAFETY_CLASSES:
            errors.append(f"bad safety_class {self.safety_class!r}")
        if self.review_status != "approved":
            errors.append("review_status must be approved")
        if self.safety_class == "destructive":
            if not (self.destructive_commit_authorized and
                    self.destructive_env_gate):
                errors.append("destructive sample lacks both authorization gates")
        elif self.destructive_commit_authorized or self.destructive_env_gate:
            errors.append("non-destructive sample must not claim destructive gates")
        detector = tuple(self.detector_channels)
        changed = tuple(self.changed_channels)
        if not detector or len(detector) != len(set(detector)) or any(
                channel not in _SIGNAL_CHANNELS for channel in detector):
            errors.append("detector_channels must be unique supported channels")
        if len(changed) != len(set(changed)) or any(
                channel not in detector for channel in changed):
            errors.append("changed_channels must be a unique detector subset")
        if self.mutation_candidate is not bool(changed):
            errors.append("mutation_candidate does not match changed_channels")
        reference_channel = _REFERENCE_SOURCES.get(self.reference_source, "invalid")
        if reference_channel == "invalid":
            errors.append(f"bad reference_source {self.reference_source!r}")
        elif reference_channel is not None and reference_channel in detector:
            errors.append("reference channel is not independent of detector")
        if self.reference_evidence_sha256 == self.execution_evidence_sha256:
            errors.append("reference and detector execution evidence must be independent")
        if errors:
            raise MutationMiningError(
                f"{self.mutation_sample_id or '<missing sample id>'}: " +
                "; ".join(dict.fromkeys(errors)))


_REPORT_FIELDS = frozenset({
    "schema_version", "release_id", "safety_mode", "collection_timestamp",
    "code_version", "protocol_id", "selection_protocol",
    "collection_run_ids", "snapshot_state_n", "unique_physical_snapshot_n",
    "duplicate_alias_audit", "enumerated_n", "executed_n",
    "detected_n", "ground_truth_positive_n", "true_positive_n",
    "false_positive_n", "false_negative_n", "true_negative_n",
    "true_positive_sample_ids", "false_positive_sample_ids",
    "false_negative_sample_ids", "true_negative_sample_ids", "precision",
    "precision_ci", "recall", "recall_ci", "samples",
})


def _validate_complete_control_groups(
    samples: Sequence[MutationCensusSample],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[MutationCensusSample]] = {}
    # ``state_id`` is a logical authoring identity, not a physical snapshot
    # identity.  The same AX snapshot can legitimately be reached by several
    # action-specific state aliases.  Counting each alias as a fresh census
    # group would let a caller pad one execution surface to the release target.
    physical_snapshot_states: dict[tuple[str, str], str] = {}
    control_identities: set[tuple[str, str, str]] = set()
    bid_identities: set[tuple[str, str, str, str]] = set()
    sample_ids: set[str] = set()
    for sample in samples:
        if sample.mutation_sample_id in sample_ids:
            raise MutationMiningError("duplicate mutation_sample_id")
        sample_ids.add(sample.mutation_sample_id)
        physical_key = (sample.environment_instance, sample.snapshot_hash)
        previous_state = physical_snapshot_states.setdefault(
            physical_key, sample.state_id)
        if previous_state != sample.state_id:
            raise MutationMiningError(
                "one physical snapshot cannot be counted through multiple "
                f"state_id aliases: environment={sample.environment_instance!r}, "
                f"snapshot_hash={sample.snapshot_hash}, "
                f"states={sorted({previous_state, sample.state_id})}")
        control_identity = (*physical_key, sample.control_id)
        bid_identity = (
            *physical_key, sample.bid, sample.canonical_action)
        if control_identity in control_identities or bid_identity in bid_identities:
            raise MutationMiningError(
                "duplicate physical snapshot/control identity cannot pad census")
        control_identities.add(control_identity)
        bid_identities.add(bid_identity)
        key = (sample.environment_instance, sample.state_id,
               sample.snapshot_hash)
        grouped.setdefault(key, []).append(sample)

    for key, rows in grouped.items():
        counts = {row.snapshot_control_count for row in rows}
        hashes = {row.control_set_sha256 for row in rows}
        if len(counts) != 1 or counts != {len(rows)}:
            raise MutationMiningError(
                f"snapshot {key} is not a complete enumerated control set")
        if len({row.bid for row in rows}) != len(rows):
            raise MutationMiningError(f"snapshot {key} repeats a bid")
        expected_hash = mutation_control_set_sha256(
            (row.bid, row.canonical_action) for row in rows)
        if hashes != {expected_hash}:
            raise MutationMiningError(
                f"snapshot {key} control_set_sha256 mismatch")
    return {
        "performed": True,
        "passed": True,
        "physical_snapshot_identity": [
            "environment_instance", "snapshot_hash"],
        "duplicate_physical_snapshot_n": 0,
        "duplicate_state_alias_n": 0,
        "duplicate_state_aliases": [],
    }


def _confusion(samples: Sequence[MutationCensusSample]) -> dict[str, Any]:
    buckets = {"tp": [], "fp": [], "fn": [], "tn": []}
    for sample in samples:
        key = (
            "tp" if sample.mutation_candidate and sample.reference_mutated else
            "fp" if sample.mutation_candidate else
            "fn" if sample.reference_mutated else "tn"
        )
        buckets[key].append(sample.mutation_sample_id)
    for values in buckets.values():
        values.sort()
    tp, fp, fn = len(buckets["tp"]), len(buckets["fp"]), len(buckets["fn"])
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    precision = tp / precision_denominator if precision_denominator else 0.0
    recall = tp / recall_denominator if recall_denominator else 0.0
    return {
        "detected_n": precision_denominator,
        "ground_truth_positive_n": recall_denominator,
        "true_positive_n": tp,
        "false_positive_n": fp,
        "false_negative_n": fn,
        "true_negative_n": len(buckets["tn"]),
        "true_positive_sample_ids": buckets["tp"],
        "false_positive_sample_ids": buckets["fp"],
        "false_negative_sample_ids": buckets["fn"],
        "true_negative_sample_ids": buckets["tn"],
        "precision": precision,
        "precision_ci": list(wilson_interval(tp, precision_denominator)),
        "recall": recall,
        "recall_ci": list(wilson_interval(tp, recall_denominator)),
    }


def build_live_mutation_census(
    samples: Iterable[MutationCensusSample], *, release_id: str,
    collection_timestamp: str, code_version: str, protocol_id: str,
) -> dict[str, Any]:
    """Build aggregates from executed samples; aggregate inputs are impossible."""
    rows = sorted(list(samples), key=lambda row: row.mutation_sample_id)
    if not rows:
        raise MutationMiningError("cannot build a live mutation census with zero samples")
    for row in rows:
        row.validate()
    for field_name, value in (
        ("release_id", release_id), ("collection_timestamp", collection_timestamp),
        ("code_version", code_version), ("protocol_id", protocol_id),
    ):
        if not str(value or "").strip():
            raise MutationMiningError(f"{field_name} is required")
    duplicate_alias_audit = _validate_complete_control_groups(rows)
    metrics = _confusion(rows)
    unique_physical_snapshot_n = len({
        (row.environment_instance, row.snapshot_hash) for row in rows})
    report = {
        "schema_version": MUTATION_REPORT_SCHEMA_VERSION,
        "release_id": release_id,
        "safety_mode": "live",
        "collection_timestamp": collection_timestamp,
        "code_version": code_version,
        "protocol_id": protocol_id,
        "selection_protocol": (
            "all_legal_interactive_controls_at_recorded_snapshot;"
            "physical_snapshot_dedup=environment_instance+snapshot_hash"),
        "collection_run_ids": sorted({row.collection_run_id for row in rows}),
        "snapshot_state_n": len({
            (row.environment_instance, row.state_id, row.snapshot_hash)
            for row in rows}),
        "unique_physical_snapshot_n": unique_physical_snapshot_n,
        "duplicate_alias_audit": duplicate_alias_audit,
        "enumerated_n": len(rows),
        "executed_n": len(rows),
        **metrics,
        "samples": [row.to_dict() for row in rows],
    }
    return report


def validate_live_mutation_census(report: Mapping[str, Any]) \
        -> tuple[dict[str, Any], list[MutationCensusSample]]:
    """Strictly parse and reproduce a report byte-for-byte semantically."""
    if not isinstance(report, Mapping):
        raise MutationMiningError("mutation census report must be an object")
    unknown = sorted(set(report) - _REPORT_FIELDS)
    missing = sorted(_REPORT_FIELDS - set(report))
    if unknown:
        raise MutationMiningError(f"unknown mutation report fields: {unknown}")
    if missing:
        raise MutationMiningError(f"missing mutation report fields: {missing}")
    if report.get("schema_version") != MUTATION_REPORT_SCHEMA_VERSION:
        raise MutationMiningError("bad mutation report schema_version")
    if report.get("safety_mode") != "live":
        raise MutationMiningError("mutation census safety_mode must be live")
    raw_samples = report.get("samples")
    if not isinstance(raw_samples, list):
        raise MutationMiningError("mutation census samples must be a list")
    samples = [MutationCensusSample.from_dict(row) for row in raw_samples]
    rebuilt = build_live_mutation_census(
        samples,
        release_id=str(report.get("release_id") or ""),
        collection_timestamp=str(report.get("collection_timestamp") or ""),
        code_version=str(report.get("code_version") or ""),
        protocol_id=str(report.get("protocol_id") or ""),
    )
    if dict(report) != rebuilt:
        raise MutationMiningError(
            "mutation census aggregates/order do not match recomputed samples")
    if rebuilt["detected_n"] <= 0 or rebuilt["ground_truth_positive_n"] <= 0:
        raise MutationMiningError(
            "precision/recall require detected and reference-positive controls")
    return rebuilt, samples


def _body_text(report: Mapping[str, Any]) -> str:
    return json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True,
        allow_nan=False,
    ) + "\n"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mutation_census_manifest(
    body: Path, report: Mapping[str, Any],
) -> dict[str, Any]:
    sample_ids = sorted(
        str(row["mutation_sample_id"]) for row in report["samples"])
    return {
        "schema_version": MUTATION_MANIFEST_SCHEMA_VERSION,
        "report_schema_version": MUTATION_REPORT_SCHEMA_VERSION,
        "release_id": report["release_id"],
        "code_version": report["code_version"],
        "body_name": Path(body).name,
        "body_sha256": _file_sha256(Path(body)),
        "n_samples": len(sample_ids),
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(sample_ids).encode("utf-8")).hexdigest(),
        "collection_run_ids_sha256": hashlib.sha256(
            "\n".join(report["collection_run_ids"]).encode("utf-8")).hexdigest(),
        "unique_physical_snapshot_n": report["unique_physical_snapshot_n"],
        "duplicate_alias_audit_sha256": _json_sha256(
            report["duplicate_alias_audit"]),
    }


def save_live_mutation_census(
    samples: Iterable[MutationCensusSample], *, body: Path, manifest: Path,
    release_id: str, collection_timestamp: str, code_version: str,
    protocol_id: str,
) -> str:
    """Create an immutable body+manifest pair; exact reruns are idempotent."""
    report = build_live_mutation_census(
        samples, release_id=release_id,
        collection_timestamp=collection_timestamp,
        code_version=code_version, protocol_id=protocol_id)
    body, manifest = Path(body), Path(manifest)
    if body.exists() != manifest.exists():
        raise MutationMiningError(
            "mutation census body/manifest must both exist or both be absent")
    expected_text = _body_text(report)
    if body.exists():
        assert_mutation_census_integrity(body, manifest)
        if body.read_text(encoding="utf-8") != expected_text:
            raise MutationMiningError(
                f"refusing to overwrite versioned mutation census {body}")
        return "already_identical"

    body.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(body, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(expected_text)
            handle.flush()
            os.fsync(handle.fileno())
        pin = mutation_census_manifest(body, report)
        manifest_text = json.dumps(
            pin, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        manifest_fd = os.open(
            manifest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(manifest_fd, "w", encoding="utf-8") as handle:
            handle.write(manifest_text)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        # Only roll back files created by this invocation.
        if body.exists() and not manifest.exists():
            body.unlink()
        raise
    assert_mutation_census_integrity(body, manifest)
    return "created"


def assert_mutation_census_integrity(
    body: Path, manifest: Path,
) -> dict[str, Any]:
    """Verify body schema, recomputed aggregates, and the exact hash pin."""
    body, manifest = Path(body), Path(manifest)
    if not body.exists() or not manifest.exists():
        raise MutationMiningError("mutation census body and manifest must both exist")
    try:
        report = json.loads(body.read_text(encoding="utf-8"))
        pin = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MutationMiningError(f"invalid mutation census JSON: {exc}") from exc
    rebuilt, _samples = validate_live_mutation_census(report)
    expected = mutation_census_manifest(body, rebuilt)
    if pin != expected:
        raise MutationMiningError("mutation census manifest/hash pin mismatch")
    return rebuilt


def _snapshot_control_index(
    data_root: Path,
) -> dict[tuple[str, str], set[str]]:
    """Index exact recorded AX snapshots used to verify complete enumeration."""
    from .candidates import interactive_elements, snapshot_sha256

    indexed: dict[tuple[str, str], set[str]] = {}
    state_bank = Path(data_root) / "raw" / "state_bank"
    for path in sorted(state_bank.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise MutationMiningError(
                        f"invalid state-bank JSON {path}:{line_no}: {exc}") from exc
                state_id = str(row.get("state_id") or "")
                snapshot = str(row.get("axtree_snapshot") or "")
                if not state_id or not snapshot:
                    continue
                # A prefix-pruned AXTree proves only that its retained controls
                # exist; it cannot prove that every legal page control entered
                # the denominator.  Census rows therefore require a dedicated
                # full-AX capture rather than legacy policy/snapshot projections.
                if row.get("axtree_complete") is not True:
                    continue
                key = (state_id, snapshot_sha256(snapshot))
                # AX nodes retain an interactive role even when explicitly
                # disabled.  They are not legal controls and must neither be
                # executed nor used to inflate the census denominator.
                bids = {
                    element["bid"] for element in interactive_elements(snapshot)
                    if not re.search(
                        r"(?:^|[,\s])(?:aria-)?disabled\s*=\s*true(?:[,\s]|$)",
                        str(element.get("line") or "").lower())
                }
                previous = indexed.get(key)
                if previous is not None and previous != bids:
                    raise MutationMiningError(
                        f"conflicting recorded snapshot controls {path}:{line_no}")
                indexed[key] = bids
    return indexed


def validate_census_snapshot_coverage(
    report: Mapping[str, Any], data_root: Path,
) -> None:
    """Prove every recorded legal bid was included exactly once per snapshot."""
    _rebuilt, samples = validate_live_mutation_census(report)
    indexed = _snapshot_control_index(Path(data_root))
    grouped: dict[tuple[str, str], list[MutationCensusSample]] = {}
    for sample in samples:
        grouped.setdefault((sample.state_id, sample.snapshot_hash), []).append(sample)
    for key, rows in grouped.items():
        expected = indexed.get(key)
        if expected is None:
            raise MutationMiningError(
                f"mutation census snapshot is absent from state bank: {key}")
        observed = {row.bid for row in rows}
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise MutationMiningError(
                f"snapshot {key} control census mismatch; missing={missing}, extra={extra}")


def _within_root(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except (OSError, ValueError):
        return False
    return True


def audit_live_mutation_census(data_root: Path) -> dict[str, Any]:
    """Read-only release gate used by both CLI and publication readiness."""
    root = Path(data_root)
    body = root / "audit" / MUTATION_REPORT_BODY_NAME
    manifest = root / "audit" / MUTATION_REPORT_MANIFEST_NAME
    result: dict[str, Any] = {
        "body_path": str(body), "manifest_path": str(manifest),
        "body_exists": body.exists(), "manifest_exists": manifest.exists(),
        "release_id": "", "n_samples": 0, "sample_ids": [],
        "integrity": False, "snapshot_coverage": False,
        "unique_physical_snapshot_n": 0,
        "duplicate_alias_audit": {
            "performed": False,
            "passed": False,
            "physical_snapshot_identity": [
                "environment_instance", "snapshot_hash"],
            "duplicate_physical_snapshot_n": 0,
            "duplicate_state_alias_n": 0,
            "duplicate_state_aliases": [],
        },
        "minimum_controls": MUTATION_CENSUS_MIN_CONTROLS,
        "error": "", "passed": False,
    }
    if body.exists() != manifest.exists():
        result["error"] = "mutation report/manifest must both exist or both be absent"
        return result
    if not body.exists():
        result["error"] = "release-scoped mutation report/manifest are absent"
        return result
    if not _within_root(body, root) or not _within_root(manifest, root):
        result["error"] = "mutation report/manifest resolve outside data root"
        return result
    try:
        report = assert_mutation_census_integrity(body, manifest)
        validate_census_snapshot_coverage(report, root)
    except (MutationMiningError, OSError, ValueError) as exc:
        result["error"] = str(exc)
        return result
    sample_ids = sorted(
        str(row["mutation_sample_id"]) for row in report["samples"])
    enough = len(sample_ids) >= MUTATION_CENSUS_MIN_CONTROLS
    result.update({
        "release_id": report["release_id"],
        "n_samples": len(sample_ids),
        "sample_ids": sample_ids[:250],
        "integrity": True,
        "snapshot_coverage": True,
        "unique_physical_snapshot_n": report["unique_physical_snapshot_n"],
        "duplicate_alias_audit": report["duplicate_alias_audit"],
        "precision": report["precision"],
        "precision_ci": report["precision_ci"],
        "recall": report["recall"],
        "recall_ci": report["recall_ci"],
        "error": "" if enough else (
            f"fewer than {MUTATION_CENSUS_MIN_CONTROLS} executed controls"),
        "passed": enough,
    })
    return result
