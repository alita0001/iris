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

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence


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
