"""Auditable S4 candidate records and snapshot-legality gates.

Candidate generation is deliberately separated from grounding.  A language
model may propose an accessibility-tree control and a candidate category, but
this module will neither accept nor infer an effect, recovery, or safety label.
Those labels can only be attached later by an executed point-level probe.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CANDIDATE_SCHEMA_VERSION = "iris.candidate.v2"
FORMAL_CANDIDATE_ARTIFACT_VERSION = "formal_candidates.v4"
FORMAL_CANDIDATE_BODY_NAME = f"{FORMAL_CANDIDATE_ARTIFACT_VERSION}.jsonl"
FORMAL_CANDIDATE_MANIFEST_NAME = "FORMAL_CANDIDATE_MANIFEST.v4.jsonl"
LEGACY_CANDIDATE_BODY_NAME = "iris_candidates.v3.jsonl"
LEGACY_CANDIDATE_MANIFEST_NAME = "CANDIDATE_MANIFEST.jsonl"

CATEGORY_EXPERT = "expert_action"
CATEGORY_SAFE_ALTERNATIVE = "safe_alternative"
CATEGORY_ORDINARY = "ordinary_interactive_action"
CATEGORY_CONSTRAINT_TRIGGER = "constraint_trigger_action"
CATEGORY_GOAL_VIOLATING = "goal_violating_action"
CATEGORY_DECOY = "decoy"
CATEGORY_UNCERTAIN = "uncertain_verify_action"
CATEGORY_POLICY_ERROR = "policy_generated_error"
CANDIDATE_CATEGORIES = (
    CATEGORY_EXPERT,
    CATEGORY_SAFE_ALTERNATIVE,
    CATEGORY_ORDINARY,
    CATEGORY_CONSTRAINT_TRIGGER,
    CATEGORY_GOAL_VIOLATING,
    CATEGORY_DECOY,
    CATEGORY_UNCERTAIN,
    CATEGORY_POLICY_ERROR,
)

SOURCE_EXPERT = "expert"
SOURCE_A11Y = "a11y_enumeration"
SOURCE_RULE = "rule"
SOURCE_LLM = "llm_proposal"
SOURCE_POLICY = "policy_rollout"
SOURCE_SYNTHETIC = "synthetic_flip"
SOURCE_ON_POLICY = "on_policy"
CANDIDATE_SOURCES = (
    SOURCE_EXPERT,
    SOURCE_A11Y,
    SOURCE_RULE,
    SOURCE_LLM,
    SOURCE_POLICY,
    SOURCE_SYNTHETIC,
    SOURCE_ON_POLICY,
)

_INTERACTIVE_ROLES = frozenset({
    "button", "link", "checkbox", "radio", "combobox", "textbox",
    "searchbox", "spinbutton", "slider", "switch", "menuitem", "option",
    "tab", "treeitem",
})
_LINE_RE = re.compile(r"^\s*\[([^\]]+)\]\s+([A-Za-z_][\w-]*)\s*(.*)$")

# These are *surface-form proposal* rules, not policy or safety labels.  Use
# token/phrase boundaries so product names such as ``Bananas`` and navigation
# labels such as ``Stored Payment Methods`` do not become high-risk controls by
# substring accident.
_CONSTRAINT_TRIGGER_PHRASES = (
    "place order", "delete", "refund", "submit", "publish", "pay", "ban",
)
_UNCERTAIN_PHRASES = ("continue", "proceed", "learn more")
_GLOBAL_CHROME_NAMES = frozenset({
    "advanced search", "my account", "my cart", "my wish list", "search",
    "sign out", "skip to content", "store logo",
})
_LOCAL_ACTION_WINDOW_LINES = 40
_MAX_GLOBAL_CHROME_CANDIDATES = 1

# An S4 proposer must not smuggle an opinion into the behavioral label chain.
_LABEL_LIKE_FIELDS = frozenset({
    "label", "reversibility", "effect_status", "recovery_status",
    "normative_risk", "policy_constraint_truth", "expected_decision",
    "safe", "dangerous", "ground_truth", "grounding",
})


class CandidateValidationError(ValueError):
    """Raised when an S4 candidate cannot be admitted to a snapshot."""


def snapshot_sha256(axtree_txt: str) -> str:
    """Stable hash over the exact accessibility snapshot used for legality."""
    return hashlib.sha256((axtree_txt or "").encode("utf-8")).hexdigest()


def interactive_bids(axtree_txt: str) -> set[str]:
    """Extract exact *interactive-role* ``[bid]`` identifiers from a snapshot.

    Import lazily to keep this module usable by data-only tools without
    importing a browser environment.
    """
    return {row["bid"] for row in interactive_elements(axtree_txt)}


def interactive_elements(axtree_txt: str) -> list[dict[str, str]]:
    """Parse legal control candidates, excluding structural/static AX nodes."""
    out: list[dict[str, str]] = []
    for line in (axtree_txt or "").splitlines():
        match = _LINE_RE.match(line)
        if not match:
            continue
        bid, role, rest = match.groups()
        if role.lower() not in _INTERACTIVE_ROLES:
            continue
        name_match = re.search(r"'([^']*)'", rest)
        out.append({
            "bid": bid,
            "role": role.lower(),
            "name": name_match.group(1) if name_match else rest.strip(),
            "line": line.strip(),
        })
    return out


def _phrase_present(text: str, phrase: str) -> bool:
    """Match a normalized phrase at alphanumeric token boundaries."""
    body = r"\s+".join(re.escape(token) for token in phrase.split())
    return re.search(rf"(?<![a-z0-9]){body}(?![a-z0-9])", text.lower()) is not None


def _interactive_positions(axtree_txt: str) -> dict[str, tuple[int, int]]:
    """Return source-line/depth positions without changing the public parser."""
    out: dict[str, tuple[int, int]] = {}
    for line_index, line in enumerate((axtree_txt or "").splitlines()):
        match = _LINE_RE.match(line)
        if not match or match.group(2).lower() not in _INTERACTIVE_ROLES:
            continue
        leading = line[:len(line) - len(line.lstrip(" \t"))]
        # AXTree output normally uses tabs.  Expanding spaces as one indentation
        # unit is sufficient for deterministic locality tie-breaking; line
        # distance remains the primary signal.
        depth = leading.count("\t") + len(leading.replace("\t", ""))
        out.setdefault(match.group(1), (line_index, depth))
    return out


def _normalized_control_name(element: Mapping[str, str]) -> str:
    return re.sub(r"\s+", " ", str(element.get("name") or "").strip().lower())


def _is_global_chrome(element: Mapping[str, str]) -> bool:
    """Identify repeated site navigation, never task-local action siblings."""
    name = _normalized_control_name(element)
    if name in _GLOBAL_CHROME_NAMES or name.endswith(" my cart"):
        return True
    return str(element.get("role") or "").lower() == "menuitem"


@dataclass(frozen=True)
class Candidate:
    schema_version: str
    candidate_id: str
    state_id: str
    bid: str
    canonical_action: str
    category: str
    source: str
    legal_at_snapshot: bool
    proposer_model: str
    proposer_version: str
    snapshot_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "Candidate":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(row) - known)
        if unknown:
            raise CandidateValidationError(f"unknown candidate fields: {unknown}")
        missing = sorted(known - set(row))
        if missing:
            raise CandidateValidationError(
                "candidate row is missing serialized fields: " + ",".join(missing))
        try:
            candidate = cls(**dict(row))
        except TypeError as exc:
            raise CandidateValidationError(str(exc)) from exc
        candidate.validate_record()
        return candidate

    def validate_record(self) -> None:
        """Validate schema/proposal fields without claiming snapshot legality."""
        errors: list[str] = []
        if self.schema_version != CANDIDATE_SCHEMA_VERSION:
            errors.append(f"invalid schema_version {self.schema_version!r}")
        for name in (
            "candidate_id", "state_id", "bid", "canonical_action", "category",
            "source", "proposer_model", "proposer_version", "snapshot_hash",
        ):
            if not str(getattr(self, name) or "").strip():
                errors.append(f"missing {name}")
        if self.category not in CANDIDATE_CATEGORIES:
            errors.append(f"invalid category {self.category!r}")
        if self.source not in CANDIDATE_SOURCES:
            errors.append(f"invalid source {self.source!r}")
        if not isinstance(self.legal_at_snapshot, bool):
            errors.append("legal_at_snapshot must be boolean")
        if errors:
            raise CandidateValidationError(
                f"{self.candidate_id or '<missing candidate_id>'}: " + "; ".join(errors))

    def validate(self, axtree_txt: str, *, formal: bool = True) -> None:
        self.validate_record()
        errors: list[str] = []
        expected_hash = snapshot_sha256(axtree_txt)
        if self.snapshot_hash != expected_hash:
            errors.append("snapshot_hash does not match the supplied snapshot")
        matching = [element for element in interactive_elements(axtree_txt)
                    if element["bid"] == self.bid]
        actually_legal = bool(matching)
        if self.legal_at_snapshot is not actually_legal:
            errors.append(
                "legal_at_snapshot disagrees with exact accessibility bid membership")
        if formal and not self.legal_at_snapshot:
            errors.append(f"formal candidate bid [{self.bid}] is not legal at snapshot")
        if len(matching) == 1:
            expected_canonical = canonical_click_for_element(matching[0])
            if self.canonical_action != expected_canonical:
                errors.append(
                    "canonical_action does not match the snapshot-derived click action")
        if errors:
            raise CandidateValidationError(
                f"{self.candidate_id or '<missing candidate_id>'}: " + "; ".join(errors))


def canonical_click_for_element(element: Mapping[str, str]) -> str:
    """Derive candidate semantics from the snapshot, never from a proposer."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(element.get("name") or "").lower())
    slug = slug.strip("-")[:48]
    return f"click:{element['role']}:{slug or element['bid']}"


def candidate_from_proposal(
    proposal: Mapping[str, Any],
    *,
    state_id: str,
    axtree_txt: str,
    source: str,
    proposer_model: str,
    proposer_version: str,
) -> Candidate:
    """Materialize a proposal only after exact snapshot-legality validation.

    A proposer may supply only ``bid``, ``canonical_action`` and ``category``.
    Label-like keys are rejected (rather than silently discarded) so an audit
    can prove that LLM opinions never entered the grounding channel.
    """
    leaked = sorted(set(proposal) & _LABEL_LIKE_FIELDS)
    if leaked:
        raise CandidateValidationError(
            f"candidate proposer attempted to supply forbidden labels: {leaked}")
    allowed = {"bid", "canonical_action", "category"}
    unknown = sorted(set(proposal) - allowed)
    if unknown:
        raise CandidateValidationError(f"unknown proposal fields: {unknown}")
    missing = sorted(k for k in allowed if not str(proposal.get(k, "")).strip())
    if missing:
        raise CandidateValidationError(f"proposal is missing fields: {missing}")

    bid = str(proposal["bid"])
    legal = bid in interactive_bids(axtree_txt)
    hash_ = snapshot_sha256(axtree_txt)
    identity = json.dumps({
        "state_id": state_id,
        "bid": bid,
        "canonical_action": proposal["canonical_action"],
        "category": proposal["category"],
        "source": source,
        "snapshot_hash": hash_,
    }, sort_keys=True, separators=(",", ":"))
    candidate = Candidate(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        candidate_id="cand-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20],
        state_id=str(state_id),
        bid=bid,
        canonical_action=str(proposal["canonical_action"]),
        category=str(proposal["category"]),
        source=str(source),
        legal_at_snapshot=legal,
        proposer_model=str(proposer_model),
        proposer_version=str(proposer_version),
        snapshot_hash=hash_,
    )
    candidate.validate(axtree_txt, formal=True)
    return candidate


def _rule_category(element: Mapping[str, str], *, expert_bid: str) -> str:
    """Candidate-family proposal only; never an effect/recovery/safety label."""
    bid = str(element["bid"])
    text = _normalized_control_name(element)
    if bid == str(expert_bid):
        return CATEGORY_EXPERT
    if any(_phrase_present(text, phrase)
           for phrase in _CONSTRAINT_TRIGGER_PHRASES):
        return CATEGORY_CONSTRAINT_TRIGGER
    # "Safe", "decoy", goal-relative, and policy-error roles require evidence
    # outside a static AXTree and therefore are never assigned by this rule.
    # Bare "Next" is also deliberately ordinary: in the current Magento pages
    # it overwhelmingly denotes an image-carousel button.
    if any(_phrase_present(text, phrase) for phrase in _UNCERTAIN_PHRASES):
        return CATEGORY_UNCERTAIN
    return CATEGORY_ORDINARY


def build_a11y_candidate_set(
    *,
    state_id: str,
    axtree_txt: str,
    expert_bid: str,
    proposer_version: str = "a11y-rule-v2",
    max_candidates: int = 6,
) -> list[Candidate]:
    """Build a deterministic 4--6 candidate set from the actual snapshot.

    Categories are assigned only when their observable role/text supports the
    hypothesis. The builder never relabels an ordinary control merely to hit a
    diversity quota. Decoy and policy-error categories require a separately
    sourced proposal plus execute--then--undo/on-policy evidence.
    """
    if max_candidates < 4 or max_candidates > 6:
        raise CandidateValidationError("max_candidates must be in 4..6")
    elements = interactive_elements(axtree_txt)
    by_bid = {row["bid"]: row for row in elements}
    if str(expert_bid) not in by_bid:
        raise CandidateValidationError(
            f"expert bid [{expert_bid}] is not a legal interactive control")
    expert = by_bid[str(expert_bid)]
    if len(elements) < 4:
        raise CandidateValidationError(
            f"snapshot exposes only {len(elements)} legal controls; need >=4")

    positions = _interactive_positions(axtree_txt)
    expert_line, expert_depth = positions.get(str(expert_bid), (0, 0))

    def locality_key(row: Mapping[str, str]) -> tuple[int, int, int]:
        line, depth = positions.get(str(row["bid"]), (10**9, 10**9))
        return (abs(line - expert_line), abs(depth - expert_depth), line)

    remaining = [row for row in elements if row["bid"] != str(expert_bid)]
    non_chrome = [row for row in remaining if not _is_global_chrome(row)]
    global_chrome = [row for row in remaining if _is_global_chrome(row)]
    local = sorted(
        (row for row in non_chrome
         if locality_key(row)[0] <= _LOCAL_ACTION_WINDOW_LINES),
        key=locality_key,
    )
    local_ids = {id(row) for row in local}
    nonlocal_controls = sorted(
        (row for row in non_chrome if id(row) not in local_ids),
        key=locality_key,
    )
    global_chrome.sort(key=locality_key)

    ordered = [expert] + local + nonlocal_controls + \
        global_chrome[:_MAX_GLOBAL_CHROME_CANDIDATES]
    if len(ordered) < 4:
        raise CandidateValidationError(
            "snapshot exposes fewer than four task-local controls after the "
            "global-chrome cap")

    categorized = [(row, _rule_category(row, expert_bid=str(expert_bid)))
                   for row in ordered]
    selected_pairs = [categorized[0]]
    used_categories = {categorized[0][1]}
    # Maximize genuine observable diversity, then fill deterministically.
    for pair in categorized[1:]:
        if pair[1] not in used_categories and len(selected_pairs) < max_candidates:
            selected_pairs.append(pair)
            used_categories.add(pair[1])
    for pair in categorized[1:]:
        if pair not in selected_pairs and len(selected_pairs) < max_candidates:
            selected_pairs.append(pair)

    rows: list[Candidate] = []
    for element, category in selected_pairs:
        canonical = canonical_click_for_element(element)
        rows.append(candidate_from_proposal(
            {"bid": element["bid"], "canonical_action": canonical,
             "category": category},
            state_id=state_id, axtree_txt=axtree_txt,
            source=(SOURCE_EXPERT if element["bid"] == str(expert_bid)
                    else SOURCE_A11Y),
            proposer_model=("recorded-expert" if element["bid"] == str(expert_bid)
                            else "deterministic-a11y-enumerator"),
            proposer_version=proposer_version,
        ))
    validate_candidate_set(rows, axtree_txt)
    return rows


def save_candidate_set(candidates: Sequence[Candidate], path: Path) -> Path:
    """Append immutable candidate IDs without rewriting prior JSONL bytes."""
    materialized = list(candidates)
    if not materialized:
        raise CandidateValidationError("cannot persist an empty candidate set")
    for candidate in materialized:
        candidate.validate_record()
    existing: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            cid = str(row.get("candidate_id") or "")
            if not cid or cid in existing:
                raise CandidateValidationError(
                    f"{path}:{line_no}: missing/duplicate candidate_id")
            existing[cid] = row
    new_rows: list[dict[str, Any]] = []
    for candidate in materialized:
        if candidate.candidate_id in existing:
            if existing[candidate.candidate_id] != candidate.to_dict():
                raise CandidateValidationError(
                    f"candidate_id collision {candidate.candidate_id}")
            continue  # idempotent replay of the same snapshot proposal
        row = candidate.to_dict()
        existing[candidate.candidate_id] = row
        new_rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    if new_rows:
        with path.open("a", encoding="utf-8") as handle:
            for row in new_rows:
                handle.write(json.dumps(row, ensure_ascii=False,
                                        sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    write_candidate_manifest(path)
    return path


def candidate_manifest_row(candidate: Candidate) -> dict[str, Any]:
    payload = candidate.to_dict()
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": candidate.schema_version,
        "candidate_id": candidate.candidate_id,
        "state_id": candidate.state_id,
        "snapshot_hash": candidate.snapshot_hash,
        "record_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def candidate_manifest_path(path: Path) -> Path:
    if path.name == FORMAL_CANDIDATE_BODY_NAME:
        return path.with_name(FORMAL_CANDIDATE_MANIFEST_NAME)
    return path.with_name(LEGACY_CANDIDATE_MANIFEST_NAME)


def write_candidate_manifest(path: Path, manifest: Path | None = None) -> Path:
    """Regenerate the 1:1 content-hash manifest for an immutable candidate body."""
    path = Path(path)
    rows: list[Candidate] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        candidate = Candidate.from_dict(json.loads(line))
        if not candidate.candidate_id or candidate.candidate_id in seen:
            raise CandidateValidationError(
                f"{path}:{line_no}: missing/duplicate candidate_id")
        seen.add(candidate.candidate_id)
        rows.append(candidate)
    manifest = Path(manifest) if manifest else candidate_manifest_path(path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for candidate in rows:
            handle.write(json.dumps(candidate_manifest_row(candidate),
                                    ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, manifest)
    return manifest


def assert_candidate_manifest_integrity(path: Path,
                                        manifest: Path | None = None
                                        ) -> dict[str, Candidate]:
    """Load a candidate body only when its manifest is exactly 1:1 and intact."""
    path = Path(path)
    manifest = Path(manifest) if manifest else candidate_manifest_path(path)
    if not path.exists() or not manifest.exists():
        raise CandidateValidationError(
            "candidate body and CANDIDATE_MANIFEST.jsonl must both exist")
    candidates: dict[str, Candidate] = {}
    for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        candidate = Candidate.from_dict(json.loads(line))
        if candidate.candidate_id in candidates:
            raise CandidateValidationError(
                f"{path}:{line_no}: duplicate candidate_id")
        candidates[candidate.candidate_id] = candidate
    manifests = [json.loads(line) for line in manifest.open(encoding="utf-8")
                 if line.strip()]
    if len(manifests) != len(candidates):
        raise CandidateValidationError(
            f"candidate/manifest row mismatch: {len(candidates)} != {len(manifests)}")
    seen: set[str] = set()
    for line_no, row in enumerate(manifests, 1):
        cid = str(row.get("candidate_id") or "")
        if not cid or cid in seen or cid not in candidates:
            raise CandidateValidationError(
                f"{manifest}:{line_no}: unknown/duplicate candidate_id {cid!r}")
        seen.add(cid)
        if row != candidate_manifest_row(candidates[cid]):
            raise CandidateValidationError(
                f"{manifest}:{line_no}: manifest/hash mismatch for {cid}")
    return candidates


def validate_candidate_snapshot_artifact(
        candidates: Mapping[str, Candidate], state_bank_dir: Path) -> None:
    """Replay legality/canonical semantics against immutable full snapshots.

    A body+manifest hash proves only that the candidate file did not drift.  A
    formal release additionally needs the source AXTree that makes the bid and
    deterministic canonical click meaningful.
    """
    state_bank_dir = Path(state_bank_dir)
    snapshots: dict[tuple[str, str], str] = {}
    for path in sorted(state_bank_dir.glob("*.jsonl")):
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            state_id = str(row.get("state_id") or "")
            snapshot = str(row.get("axtree_snapshot") or "")
            if not state_id or not snapshot:
                continue
            key = (state_id, snapshot_sha256(snapshot))
            old = snapshots.get(key)
            if old is not None and old != snapshot:
                raise CandidateValidationError(
                    f"{path}:{line_no}: colliding state/snapshot hash")
            snapshots[key] = snapshot
    missing: list[str] = []
    invalid: list[str] = []
    for candidate in candidates.values():
        snapshot = snapshots.get((candidate.state_id, candidate.snapshot_hash))
        if snapshot is None:
            missing.append(candidate.candidate_id)
            continue
        try:
            candidate.validate(snapshot, formal=True)
        except CandidateValidationError as exc:
            invalid.append(str(exc))
    if missing or invalid:
        parts = []
        if missing:
            parts.append("missing source snapshots for " + ",".join(sorted(missing)[:10]))
        if invalid:
            parts.append("invalid candidates: " + " | ".join(invalid[:10]))
        raise CandidateValidationError("; ".join(parts))


def validate_candidate_set(
    candidates: Sequence[Candidate],
    axtree_txt: str,
    *,
    min_candidates: int = 4,
    max_candidates: int = 6,
    min_categories: int = 1,
) -> None:
    """Formal per-state S4 gate: 4--6 legal, unique, honestly categorized."""
    if not min_candidates <= len(candidates) <= max_candidates:
        raise CandidateValidationError(
            f"candidate set size must be {min_candidates}..{max_candidates}, "
            f"got {len(candidates)}")
    state_ids = {c.state_id for c in candidates}
    if len(state_ids) != 1:
        raise CandidateValidationError(
            f"candidate set must contain one state_id, got {sorted(state_ids)}")
    ids = [c.candidate_id for c in candidates]
    if len(ids) != len(set(ids)):
        raise CandidateValidationError("duplicate candidate_id in state candidate set")
    pairs = [(c.bid, c.canonical_action) for c in candidates]
    if len(pairs) != len(set(pairs)):
        raise CandidateValidationError("duplicate (bid, canonical_action) candidate")
    categories = {c.category for c in candidates}
    if len(categories) < min_categories:
        raise CandidateValidationError(
            f"candidate set requires at least {min_categories} categories, "
            f"got {sorted(categories)}")
    for candidate in candidates:
        candidate.validate(axtree_txt, formal=True)


def candidate_distribution(candidates: Iterable[Candidate]) -> dict[str, dict[str, int]]:
    rows = list(candidates)
    return {
        "category": dict(sorted(Counter(c.category for c in rows).items())),
        "source": dict(sorted(Counter(c.source for c in rows).items())),
    }


def audit_dpo_negative_sources(
    rows: Iterable[Mapping[str, Any]], *, minimum_grounded_share: float = 0.5,
) -> dict[str, Any]:
    """Audit that formal DPO negatives are deployment-shaped, not all flips.

    A negative is grounded when its ``negative_source`` is ``legal_candidate``
    or ``on_policy``.  The returned report is data, while a failed threshold is
    a hard exception suitable for export/CI gates.
    """
    materialized = list(rows)
    sources = Counter(str(r.get("negative_source", "missing")) for r in materialized)
    grounded = sources["legal_candidate"] + sources[SOURCE_ON_POLICY]
    total = len(materialized)
    share = grounded / total if total else 0.0
    report = {
        "n": total,
        "source_counts": dict(sorted(sources.items())),
        "legal_or_on_policy_n": grounded,
        "legal_or_on_policy_share": share,
        "minimum_required_share": minimum_grounded_share,
        "passes": bool(total and share >= minimum_grounded_share),
    }
    if not report["passes"]:
        raise CandidateValidationError(
            "formal DPO legal/on-policy negative share is "
            f"{share:.3f}, below {minimum_grounded_share:.3f}")
    return report
