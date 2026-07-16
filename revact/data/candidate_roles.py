"""Case-scoped candidate-role hypotheses and behavioral evidence.

``Candidate.category`` is an immutable, snapshot-local proposal attribute.  It
cannot express that the same legal control is an expert action for one goal and
goal-violating for another, and it is not behavioral evidence.  This module
therefore stores roles in a separate versioned asset keyed by
``(evaluation_case_id, candidate_id)``.  Publication readiness may count only
``EVIDENCED`` records; ``PROPOSED`` records remain auditable hypotheses.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping

from .candidates import (
    CATEGORY_CONSTRAINT_TRIGGER,
    CATEGORY_DECOY,
    CATEGORY_EXPERT,
    CATEGORY_GOAL_VIOLATING,
    CATEGORY_ORDINARY,
    CATEGORY_POLICY_ERROR,
    CATEGORY_SAFE_ALTERNATIVE,
    CATEGORY_UNCERTAIN,
    FORMAL_CANDIDATE_BODY_NAME,
    FORMAL_CANDIDATE_MANIFEST_NAME,
    SOURCE_A11Y,
    SOURCE_EXPERT,
    Candidate,
    assert_candidate_manifest_integrity,
)
from .. import config, prompts
from ..eval.on_policy import (
    OnPolicyStepTrace,
    OnPolicyTraceError,
    assert_on_policy_manifest_integrity,
    on_policy_manifest_path,
)
from ..eval.truth import (
    EvaluationTruthRecord,
    assert_truth_manifest_integrity,
    load_truth_records,
)
from ..grounding.schema import (
    EFFECT_CHANGED,
    RECOVERY_RECOVERED,
    GroundingPoint,
    assert_manifest_integrity,
    load_probe_points,
)
from ..train.validators import actions_match, answer_text, parse_action


CANDIDATE_ROLE_SCHEMA_VERSION = "iris.candidate_role.v2"
CANDIDATE_ROLE_MANIFEST_SCHEMA_VERSION = "iris.candidate_role_manifest.v2"
CANDIDATE_ROLE_ARTIFACT_VERSION = "candidate_roles.v8"
CANDIDATE_ROLE_BODY_NAME = f"{CANDIDATE_ROLE_ARTIFACT_VERSION}.jsonl"
CANDIDATE_ROLE_MANIFEST_NAME = "CANDIDATE_ROLE_MANIFEST.v8.jsonl"
CANDIDATE_ROLE_MATERIALIZER_VERSION = "role-protocol-replay.v3"

ROLE_EXPERT = "expert"
ROLE_SAFE_ALTERNATIVE = "safe_alternative"
ROLE_ORDINARY = "ordinary"
ROLE_CONSTRAINT_TRIGGER = "constraint_trigger"
ROLE_GOAL_VIOLATING = "goal_violating"
ROLE_DECOY = "decoy"
ROLE_UNCERTAIN_VERIFY = "uncertain_verify"
ROLE_POLICY_GENERATED_ERROR = "policy_generated_error"
CANDIDATE_ROLES = (
    ROLE_EXPERT,
    ROLE_SAFE_ALTERNATIVE,
    ROLE_ORDINARY,
    ROLE_CONSTRAINT_TRIGGER,
    ROLE_GOAL_VIOLATING,
    ROLE_DECOY,
    ROLE_UNCERTAIN_VERIFY,
    ROLE_POLICY_GENERATED_ERROR,
)

STATUS_PROPOSED = "PROPOSED"
STATUS_EVIDENCED = "EVIDENCED"
CANDIDATE_ROLE_STATUSES = (STATUS_PROPOSED, STATUS_EVIDENCED)

PROTOCOL_PROPOSAL = "proposal_only.v1"
PROTOCOL_EXPERT = "exact_required_expert_action.v1"
PROTOCOL_SAFE_ALTERNATIVE = "exact_goal_preserving_alternative.v1"
PROTOCOL_ORDINARY = "exact_executed_ordinary_control.v2"
PROTOCOL_CONSTRAINT_TRIGGER = "exact_policy_constraint_trigger.v1"
PROTOCOL_GOAL_VIOLATING = "exact_goal_constraint_violation.v1"
PROTOCOL_DECOY = "exact_surface_hazard_recovered_safe.v1"
PROTOCOL_UNCERTAIN_VERIFY = "exact_information_gap_verify.v1"
PROTOCOL_POLICY_GENERATED_ERROR = "exact_legal_policy_action_error.v1"
ROLE_EVIDENCE_PROTOCOLS = {
    ROLE_EXPERT: PROTOCOL_EXPERT,
    ROLE_SAFE_ALTERNATIVE: PROTOCOL_SAFE_ALTERNATIVE,
    ROLE_ORDINARY: PROTOCOL_ORDINARY,
    ROLE_CONSTRAINT_TRIGGER: PROTOCOL_CONSTRAINT_TRIGGER,
    ROLE_GOAL_VIOLATING: PROTOCOL_GOAL_VIOLATING,
    ROLE_DECOY: PROTOCOL_DECOY,
    ROLE_UNCERTAIN_VERIFY: PROTOCOL_UNCERTAIN_VERIFY,
    ROLE_POLICY_GENERATED_ERROR: PROTOCOL_POLICY_GENERATED_ERROR,
}

# A decoy is a *measured* safe/recoverable instance with an independently
# derived hazardous surface form.  Candidate.category is deliberately absent.
_DECOY_SURFACE_PHRASES = (
    "ban", "delete", "pay", "place-order", "publish", "refund", "remove",
    "submit", "upvote", "downvote",
)

_CATEGORY_TO_ROLE = {
    CATEGORY_EXPERT: ROLE_EXPERT,
    CATEGORY_SAFE_ALTERNATIVE: ROLE_SAFE_ALTERNATIVE,
    CATEGORY_ORDINARY: ROLE_ORDINARY,
    CATEGORY_CONSTRAINT_TRIGGER: ROLE_CONSTRAINT_TRIGGER,
    CATEGORY_GOAL_VIOLATING: ROLE_GOAL_VIOLATING,
    CATEGORY_DECOY: ROLE_DECOY,
    CATEGORY_UNCERTAIN: ROLE_UNCERTAIN_VERIFY,
    CATEGORY_POLICY_ERROR: ROLE_POLICY_GENERATED_ERROR,
}
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_EVIDENCE_REF_FIELDS = frozenset({"artifact", "record_id", "record_sha256"})


class CandidateRoleError(ValueError):
    """A role row or its source-evidence join is invalid."""


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def policy_fingerprint(truth: EvaluationTruthRecord) -> str:
    """Fingerprint the normative instrument, not measured recoverability."""
    return canonical_json_sha256({
        "policy_id": truth.policy_id,
        "policy_version": truth.policy_version,
        "truth_source": truth.truth_source,
        "policy_clause": str((truth.evidence or {}).get("policy_clause") or ""),
    })


def candidate_role_paths(data_root: Path) -> tuple[Path, Path]:
    directory = Path(data_root) / "raw" / "candidates"
    return (directory / CANDIDATE_ROLE_BODY_NAME,
            directory / CANDIDATE_ROLE_MANIFEST_NAME)


def _evidence_ref(artifact: str, record_id: str,
                  payload: Mapping[str, Any]) -> dict[str, str]:
    return {
        "artifact": artifact,
        "record_id": str(record_id),
        "record_sha256": canonical_json_sha256(dict(payload)),
    }


def make_candidate_role_id(*, candidate_id: str, evaluation_case_id: str,
                           role: str, status: str, goal_sha256: str,
                           policy_fp: str,
                           evidence_protocol: str = PROTOCOL_PROPOSAL) -> str:
    identity = {
        "candidate_id": candidate_id,
        "evaluation_case_id": evaluation_case_id,
        "role": role,
        "status": status,
        "goal_sha256": goal_sha256,
        "policy_fp": policy_fp,
        "evidence_protocol": evidence_protocol,
    }
    return "role-" + canonical_json_sha256(identity)[:20]


@dataclass(frozen=True)
class CandidateRoleRecord:
    schema_version: str
    candidate_role_id: str
    candidate_id: str
    probe_point_id: str
    state_id: str
    evaluation_case_id: str
    goal_sha256: str
    policy_fp: str
    snapshot_hash: str
    role: str
    status: str
    evidence_protocol: str
    basis: str
    evidence_refs: list[dict[str, str]]
    materializer_version: str
    code_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate_record(self) -> None:
        errors: list[str] = []
        if self.schema_version != CANDIDATE_ROLE_SCHEMA_VERSION:
            errors.append("bad schema_version")
        for name in (
                "candidate_role_id", "candidate_id", "probe_point_id",
                "state_id", "evaluation_case_id", "basis",
                "materializer_version", "code_version"):
            if not str(getattr(self, name) or "").strip():
                errors.append(f"missing {name}")
        for name in ("goal_sha256", "policy_fp", "snapshot_hash"):
            if not _HEX_64.fullmatch(str(getattr(self, name) or "")):
                errors.append(f"{name} must be a lowercase sha256")
        if self.role not in CANDIDATE_ROLES:
            errors.append(f"invalid role {self.role!r}")
        if self.status not in CANDIDATE_ROLE_STATUSES:
            errors.append(f"invalid status {self.status!r}")
        expected_protocol = (PROTOCOL_PROPOSAL if self.status == STATUS_PROPOSED
                             else ROLE_EVIDENCE_PROTOCOLS.get(self.role))
        if self.evidence_protocol != expected_protocol:
            errors.append(
                f"evidence_protocol must be {expected_protocol!r} for "
                f"{self.status}/{self.role}")
        if not isinstance(self.evidence_refs, list) or not self.evidence_refs:
            errors.append("evidence_refs must be a non-empty list")
        else:
            identities: set[tuple[str, str]] = set()
            for index, ref in enumerate(self.evidence_refs):
                if not isinstance(ref, dict) or set(ref) != _EVIDENCE_REF_FIELDS:
                    errors.append(f"evidence_refs[{index}] has invalid fields")
                    continue
                if not str(ref.get("artifact") or "").strip() or not str(
                        ref.get("record_id") or "").strip():
                    errors.append(f"evidence_refs[{index}] has an empty identity")
                if not _HEX_64.fullmatch(str(ref.get("record_sha256") or "")):
                    errors.append(f"evidence_refs[{index}] has an invalid hash")
                identity = (str(ref.get("artifact")), str(ref.get("record_id")))
                if identity in identities:
                    errors.append(f"duplicate evidence ref {identity}")
                identities.add(identity)
            if self.status == STATUS_EVIDENCED and \
                    self.role == ROLE_SAFE_ALTERNATIVE:
                formal_candidate_ids = [
                    str(ref.get("record_id") or "")
                    for ref in self.evidence_refs
                    if isinstance(ref, dict) and
                    ref.get("artifact") == "formal_candidate"]
                if (len(formal_candidate_ids) != 2 or
                        len(set(formal_candidate_ids)) != 2 or
                        self.candidate_id not in formal_candidate_ids):
                    errors.append(
                        "evidenced safe alternative requires executed and "
                        "distinct alternative formal-candidate refs")
        expected_id = make_candidate_role_id(
            candidate_id=self.candidate_id,
            evaluation_case_id=self.evaluation_case_id,
            role=self.role, status=self.status,
            goal_sha256=self.goal_sha256, policy_fp=self.policy_fp,
            evidence_protocol=self.evidence_protocol)
        if self.candidate_role_id != expected_id:
            errors.append("candidate_role_id does not match immutable identity")
        if errors:
            raise CandidateRoleError(
                f"{self.candidate_role_id or '<missing role id>'}: " +
                "; ".join(errors))

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "CandidateRoleRecord":
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(row) - known)
        missing = sorted(known - set(row))
        if unknown:
            raise CandidateRoleError(f"unknown candidate-role fields: {unknown}")
        if missing:
            raise CandidateRoleError(f"missing candidate-role fields: {missing}")
        try:
            record = cls(**dict(row))
        except TypeError as exc:
            raise CandidateRoleError(str(exc)) from exc
        record.validate_record()
        return record


def candidate_role_manifest_row(record: CandidateRoleRecord) -> dict[str, Any]:
    return {
        "schema_version": CANDIDATE_ROLE_MANIFEST_SCHEMA_VERSION,
        "candidate_role_id": record.candidate_role_id,
        "candidate_id": record.candidate_id,
        "evaluation_case_id": record.evaluation_case_id,
        "role": record.role,
        "status": record.status,
        "record_sha256": canonical_json_sha256(record.to_dict()),
    }


def _serialized(records: Iterable[CandidateRoleRecord]) -> tuple[str, str]:
    ordered = sorted(records, key=lambda item: item.candidate_role_id)
    body = "".join(json.dumps(item.to_dict(), ensure_ascii=False,
                              sort_keys=True) + "\n" for item in ordered)
    manifest = "".join(json.dumps(candidate_role_manifest_row(item),
                                  ensure_ascii=False, sort_keys=True) + "\n"
                       for item in ordered)
    return body, manifest


def save_candidate_role_records(records: Iterable[CandidateRoleRecord],
                                body: Path, manifest: Path) -> tuple[Path, Path]:
    """Create an immutable body+manifest; an exact rerun is idempotent."""
    materialized = list(records)
    if not materialized:
        raise CandidateRoleError("cannot save an empty candidate-role asset")
    indexed: dict[str, CandidateRoleRecord] = {}
    for record in materialized:
        record.validate_record()
        old = indexed.get(record.candidate_role_id)
        if old is not None:
            if old != record:
                raise CandidateRoleError(
                    f"candidate_role_id collision {record.candidate_role_id}")
            raise CandidateRoleError(
                f"duplicate candidate_role_id {record.candidate_role_id}")
        indexed[record.candidate_role_id] = record
    body_text, manifest_text = _serialized(indexed.values())
    body, manifest = Path(body), Path(manifest)
    if body.exists() != manifest.exists():
        raise CandidateRoleError(
            "candidate-role body and manifest must both exist or both be absent")
    if body.exists():
        assert_candidate_role_manifest_integrity(body, manifest)
        if (body.read_text(encoding="utf-8") != body_text or
                manifest.read_text(encoding="utf-8") != manifest_text):
            raise CandidateRoleError(
                f"refusing to overwrite versioned candidate-role asset {body}")
        return body, manifest

    body.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    created_body = False
    created_manifest = False
    body_fd = os.open(body, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    created_body = True
    try:
        with os.fdopen(body_fd, "w", encoding="utf-8") as handle:
            handle.write(body_text)
            handle.flush()
            os.fsync(handle.fileno())
        manifest_fd = os.open(
            manifest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        created_manifest = True
        with os.fdopen(manifest_fd, "w", encoding="utf-8") as handle:
            handle.write(manifest_text)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        # Roll back only files created by this invocation.
        if created_manifest and manifest.exists():
            manifest.unlink()
        if created_body and body.exists():
            body.unlink()
        raise
    assert_candidate_role_manifest_integrity(body, manifest)
    return body, manifest


def load_candidate_role_records(path: Path) -> dict[str, CandidateRoleRecord]:
    records: dict[str, CandidateRoleRecord] = {}
    if not Path(path).exists():
        return records
    for line_no, line in enumerate(Path(path).open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            record = CandidateRoleRecord.from_dict(json.loads(line))
        except (json.JSONDecodeError, CandidateRoleError) as exc:
            raise CandidateRoleError(f"{path}:{line_no}: {exc}") from exc
        if record.candidate_role_id in records:
            raise CandidateRoleError(
                f"{path}:{line_no}: duplicate candidate_role_id")
        records[record.candidate_role_id] = record
    return records


def assert_candidate_role_manifest_integrity(body: Path, manifest: Path) \
        -> dict[str, CandidateRoleRecord]:
    if not Path(body).exists() or not Path(manifest).exists():
        raise CandidateRoleError(
            "candidate-role body and manifest must both exist")
    records = load_candidate_role_records(body)
    rows = [json.loads(line) for line in Path(manifest).open(encoding="utf-8")
            if line.strip()]
    indexed = {str(row.get("candidate_role_id") or ""): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(records):
        raise CandidateRoleError("candidate-role body/manifest is not 1:1")
    for role_id, record in records.items():
        if indexed[role_id] != candidate_role_manifest_row(record):
            raise CandidateRoleError(
                f"candidate-role manifest hash mismatch {role_id}")
    return records


def _sft_case_index(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        meta = row.get("meta") or {}
        case_id = str(meta.get("evaluation_case_id") or "")
        if not case_id:
            continue
        if case_id in indexed:
            raise CandidateRoleError(f"duplicate formal SFT case {case_id}")
        indexed[case_id] = row
    return indexed


def _goal_from_sft(row: Mapping[str, Any]) -> str:
    messages = row.get("messages") or []
    user = next((str(message.get("content") or "")
                 for message in reversed(messages)
                 if isinstance(message, dict) and message.get("role") == "user"), "")
    return str(prompts.parse_user(user).get("goal") or "").strip()


def _record(*, candidate: Candidate, point: GroundingPoint,
            truth: EvaluationTruthRecord, sft: Mapping[str, Any], role: str,
            status: str, basis: str, code_version: str,
            include_point_evidence: bool,
            evidence_protocol: str = PROTOCOL_PROPOSAL,
            on_policy_trace: OnPolicyStepTrace | None = None,
            alternative_candidate: Candidate | None = None,
            ) -> CandidateRoleRecord:
    goal = _goal_from_sft(sft)
    if not goal:
        raise CandidateRoleError(
            f"{truth.evaluation_case_id}: formal SFT has no parseable goal")
    goal_hash = hashlib.sha256(goal.encode("utf-8")).hexdigest()
    policy_fp = policy_fingerprint(truth)
    sample_id = str(sft.get("sample_id") or "")
    refs = [_evidence_ref(
        "formal_candidate", candidate.candidate_id, candidate.to_dict())]
    if include_point_evidence:
        refs.append(_evidence_ref(
            "grounding_point", point.probe_point_id, point.to_dict()))
    if alternative_candidate is not None:
        refs.append(_evidence_ref(
            "formal_candidate", alternative_candidate.candidate_id,
            alternative_candidate.to_dict()))
    refs.extend([
        _evidence_ref("evaluation_truth", truth.evaluation_case_id,
                      truth.to_dict()),
        _evidence_ref("formal_sft", sample_id, dict(sft)),
    ])
    if on_policy_trace is not None:
        refs.append(_evidence_ref(
            "on_policy_trace", on_policy_trace.trace_id,
            on_policy_trace.to_dict()))
    role_id = make_candidate_role_id(
        candidate_id=candidate.candidate_id,
        evaluation_case_id=truth.evaluation_case_id,
        role=role, status=status, goal_sha256=goal_hash,
        policy_fp=policy_fp, evidence_protocol=evidence_protocol)
    return CandidateRoleRecord(
        schema_version=CANDIDATE_ROLE_SCHEMA_VERSION,
        candidate_role_id=role_id,
        candidate_id=candidate.candidate_id,
        probe_point_id=point.probe_point_id,
        state_id=point.state_id,
        evaluation_case_id=truth.evaluation_case_id,
        goal_sha256=goal_hash,
        policy_fp=policy_fp,
        snapshot_hash=candidate.snapshot_hash,
        role=role,
        status=status,
        evidence_protocol=evidence_protocol,
        basis=basis,
        evidence_refs=refs,
        materializer_version=CANDIDATE_ROLE_MATERIALIZER_VERSION,
        code_version=code_version,
    )


def _sft_meta(sft: Mapping[str, Any] | None) -> Mapping[str, Any]:
    meta = (sft or {}).get("meta") or {}
    return meta if isinstance(meta, Mapping) else {}


def _sft_selects_point_action(sft: Mapping[str, Any],
                              point: GroundingPoint) -> bool:
    messages = sft.get("messages") or []
    completion = next((str(message.get("content") or "")
                       for message in reversed(messages)
                       if isinstance(message, Mapping) and
                       message.get("role") == "assistant"), "")
    return actions_match(parse_action(answer_text(completion)), point.raw_action)


def _exact_execution_problems(candidate: Candidate, point: GroundingPoint,
                              truth: EvaluationTruthRecord,
                              sft: Mapping[str, Any]) -> list[str]:
    """Replay the common exact candidate -> point -> truth -> prompt join."""
    problems: list[str] = []
    meta = _sft_meta(sft)
    if point.candidate_id != candidate.candidate_id:
        problems.append("role protocol requires the exactly executed candidate")
    if point.canonical_action != candidate.canonical_action:
        problems.append("candidate canonical_action is not the executed point action")
    parsed_point = parse_action(point.raw_action)
    if parsed_point is None or parsed_point.bid != candidate.bid:
        problems.append("candidate bid is not the executed point action bid")
    if not candidate.legal_at_snapshot or meta.get("legal_at_snapshot") is not True:
        problems.append("role protocol requires exact snapshot legality")
    if candidate.snapshot_hash != point.pre_observation_hash:
        problems.append("candidate snapshot is not the point pre-observation")
    if meta.get("formal_dataset") is not True:
        problems.append("role protocol requires a formal SFT row")
    if meta.get("collector_success") is not True:
        problems.append("role protocol requires collector_success=true")
    if str(meta.get("candidate_id") or "") != candidate.candidate_id:
        problems.append("formal SFT candidate_id mismatch")
    if str(meta.get("probe_point_id") or "") != point.probe_point_id:
        problems.append("formal SFT probe_point_id mismatch")
    if str(meta.get("evaluation_case_id") or "") != truth.evaluation_case_id:
        problems.append("formal SFT evaluation_case_id mismatch")
    return problems


def _case_semantics_problems(truth: EvaluationTruthRecord,
                             sft: Mapping[str, Any], **expected: Any) -> list[str]:
    problems: list[str] = []
    meta = _sft_meta(sft)
    truth_alias = {
        "decision": "expected_decision",
    }
    for name, wanted in expected.items():
        truth_name = truth_alias.get(name, name)
        if getattr(truth, truth_name, object()) != wanted:
            problems.append(f"truth {truth_name} must equal {wanted!r}")
        if meta.get(name) != wanted:
            problems.append(f"formal SFT {name} must equal {wanted!r}")
    return problems


def _has_decoy_surface(candidate: Candidate) -> bool:
    text = candidate.canonical_action.lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return any(re.search(rf"(?:^|-){re.escape(phrase)}(?:-|$)", normalized)
               for phrase in _DECOY_SURFACE_PHRASES)


def role_protocol_problems(
    role: str, *, candidate: Candidate, point: GroundingPoint,
    truth: EvaluationTruthRecord, sft: Mapping[str, Any],
    on_policy_trace: OnPolicyStepTrace | None = None,
    candidates: Mapping[str, Candidate] | None = None,
) -> list[str]:
    """Return fail-closed reasons why one exact join cannot evidence ``role``.

    Each protocol has positive, independently replayable predicates.  Absence of
    danger, a proposer category, or free-text ``basis`` is never sufficient.
    """
    problems = _exact_execution_problems(candidate, point, truth, sft)
    meta = _sft_meta(sft)
    if role == ROLE_EXPERT:
        problems.extend(_case_semantics_problems(
            truth, sft, action_required_for_goal=True,
            violates_constraint=False, policy_constraint_truth=False,
            decision="EXECUTE"))
        if candidate.source != SOURCE_EXPERT or candidate.category != CATEGORY_EXPERT:
            problems.append("expert requires expert-source candidate provenance")
        if point.effect_status != EFFECT_CHANGED:
            problems.append("expert requires a measured state-changing execution")
        if not _sft_selects_point_action(sft, point):
            problems.append("expert formal gold does not select the point action")
    elif role == ROLE_ORDINARY:
        problems.extend(_case_semantics_problems(
            truth, sft, normative_risk=False, action_required_for_goal=True,
            violates_constraint=False, policy_constraint_truth=False,
            decision="EXECUTE"))
        # ``ordinary`` is a case-scoped behavioral role, whereas ``source``
        # and ``category`` are immutable proposal provenance.  Requiring an
        # executed point to have been proposed under CATEGORY_ORDINARY made the
        # protocol impossible for the exact same legal control when it was also
        # the recorded expert action.  The common exact-execution checks above,
        # the normative truth predicates here, and measured state change below
        # are the evidence.  A proposer category remains a proposal only.
        parsed = parse_action(point.raw_action)
        if parsed is None or parsed.bid is None:
            problems.append("ordinary requires an executed DOM-targeted control")
        if candidate.source not in {SOURCE_EXPERT, SOURCE_A11Y}:
            problems.append(
                "ordinary excludes synthetic/LLM/policy-only candidate provenance")
        if point.effect_status != EFFECT_CHANGED:
            problems.append("ordinary requires a measured state-changing execution")
        if not _sft_selects_point_action(sft, point):
            problems.append("ordinary formal gold does not select the point action")
    elif role in {ROLE_CONSTRAINT_TRIGGER, ROLE_GOAL_VIOLATING}:
        problems.extend(_case_semantics_problems(
            truth, sft, action_required_for_goal=False,
            violates_constraint=True, policy_constraint_truth=True,
            decision="AVOID"))
        if not str((truth.evidence or {}).get("policy_clause") or "").strip():
            problems.append("constraint evidence requires an explicit policy_clause")
    elif role == ROLE_SAFE_ALTERNATIVE:
        problems.extend(_case_semantics_problems(
            truth, sft, normative_risk=False, action_required_for_goal=True,
            violates_constraint=False, policy_constraint_truth=False,
            decision="EXECUTE"))
        truth_alternative = str(
            (truth.evidence or {}).get("alternative_to_candidate_id") or "")
        meta_alternative = str(meta.get("alternative_to_candidate_id") or "")
        if (not truth_alternative or truth_alternative != meta_alternative or
                truth_alternative == candidate.candidate_id):
            problems.append(
                "safe alternative requires an exact distinct alternative target "
                "in truth and SFT")
        else:
            alternative = (candidates or {}).get(truth_alternative)
            if alternative is None:
                problems.append(
                    "safe alternative target is absent from formal candidates")
            else:
                if alternative.state_id != candidate.state_id:
                    problems.append(
                        "safe alternative target state_id differs from executed "
                        "candidate")
                if alternative.snapshot_hash != candidate.snapshot_hash:
                    problems.append(
                        "safe alternative target snapshot_hash differs from "
                        "executed candidate")
                if alternative.legal_at_snapshot is not True:
                    problems.append(
                        "safe alternative target is not legal at snapshot")
        if not _sft_selects_point_action(sft, point):
            problems.append("safe-alternative gold does not select the point action")
    elif role == ROLE_DECOY:
        problems.extend(_case_semantics_problems(
            truth, sft, normative_risk=False, violates_constraint=False,
            policy_constraint_truth=False, decision="EXECUTE"))
        if not _has_decoy_surface(candidate):
            problems.append("decoy lacks a deterministically derived hazard surface")
        if (point.effect_status != EFFECT_CHANGED or
                point.recovery_status != RECOVERY_RECOVERED):
            problems.append("decoy requires constructive changed-and-recovered evidence")
        if (point.evidence or {}).get("backend_commit") is not False:
            problems.append("decoy requires measured backend_commit=false")
        if not _sft_selects_point_action(sft, point):
            problems.append("decoy formal gold does not select the point action")
    elif role == ROLE_UNCERTAIN_VERIFY:
        problems.extend(_case_semantics_problems(
            truth, sft, action_required_for_goal=False,
            violates_constraint=False, decision="VERIFY"))
        evidence = truth.evidence or {}
        if evidence.get("verification_required") is not True or not str(
                evidence.get("information_gap") or "").strip():
            problems.append(
                "uncertain/VERIFY requires an explicit information_gap truth")
    elif role == ROLE_POLICY_GENERATED_ERROR:
        if on_policy_trace is None:
            problems.append("policy error requires an immutable on-policy trace")
        else:
            if (on_policy_trace.candidate_id != candidate.candidate_id or
                    on_policy_trace.evaluation_case_id != truth.evaluation_case_id or
                    on_policy_trace.probe_point_id != point.probe_point_id):
                problems.append("on-policy trace does not exactly join candidate/case/point")
            if (on_policy_trace.action_kind != "legal" or
                    on_policy_trace.action_legal is not True or
                    on_policy_trace.eligible_as_negative is not True):
                problems.append("policy error trace is not a verified legal negative")
            # Format/recovery mistakes do not turn the selected action into an
            # action-distribution negative.  The current trace corpus has no
            # qualifying legal constraint-violation attempt.
            if "constraint_violation_attempt" not in on_policy_trace.error_types:
                problems.append(
                    "policy error requires a legal constraint_violation_attempt")
    else:
        problems.append(f"unsupported role protocol {role!r}")
    return list(dict.fromkeys(problems))


def validate_candidate_role_evidence(
    records: Iterable[CandidateRoleRecord], *,
    candidates: Mapping[str, Candidate],
    points: Mapping[str, GroundingPoint],
    truths: Mapping[str, EvaluationTruthRecord],
    sft_rows: Iterable[Mapping[str, Any]],
    on_policy_traces: Mapping[str, OnPolicyStepTrace] | None = None,
) -> dict[str, Any]:
    """Replay every role against exact source records and case semantics."""
    sft_by_case = _sft_case_index(sft_rows)
    failures: dict[str, list[str]] = {}
    evidenced = Counter()
    proposed = Counter()
    protocol_counts = Counter()
    traces = on_policy_traces or {}
    for record in records:
        problems: list[str] = []
        candidate = candidates.get(record.candidate_id)
        point = points.get(record.probe_point_id)
        truth = truths.get(record.evaluation_case_id)
        sft = sft_by_case.get(record.evaluation_case_id)
        trace_ref = next((ref for ref in record.evidence_refs
                          if ref.get("artifact") == "on_policy_trace"), None)
        trace = traces.get(str((trace_ref or {}).get("record_id") or ""))
        if candidate is None:
            problems.append("unknown candidate_id")
        if point is None:
            problems.append("unknown probe_point_id")
        if truth is None:
            problems.append("unknown evaluation_case_id")
        if sft is None:
            problems.append("missing formal SFT case")
        if candidate is not None:
            if candidate.state_id != record.state_id:
                problems.append("candidate state_id mismatch")
            if candidate.snapshot_hash != record.snapshot_hash:
                problems.append("candidate snapshot_hash mismatch")
        if point is not None:
            if point.state_id != record.state_id:
                problems.append("point state_id mismatch")
            if point.candidate_id != record.candidate_id and \
                    record.status == STATUS_EVIDENCED:
                problems.append("evidenced role is not the executed candidate")
        if truth is not None:
            if truth.probe_point_id != record.probe_point_id or \
                    truth.state_id != record.state_id:
                problems.append("truth point/state mismatch")
            if policy_fingerprint(truth) != record.policy_fp:
                problems.append("policy_fp mismatch")
        if sft is not None:
            meta = sft.get("meta") or {}
            goal = _goal_from_sft(sft)
            if not goal or hashlib.sha256(goal.encode("utf-8")).hexdigest() != \
                    record.goal_sha256:
                problems.append("goal_sha256 mismatch")
            if str(meta.get("probe_point_id") or "") != record.probe_point_id:
                problems.append("SFT probe_point_id mismatch")
            if str(meta.get("candidate_id") or "") != record.candidate_id and \
                    record.status == STATUS_EVIDENCED:
                problems.append("SFT candidate_id mismatch")
        expected_refs: list[dict[str, str]] = []
        if candidate is not None:
            expected_refs.append(_evidence_ref(
                "formal_candidate", candidate.candidate_id, candidate.to_dict()))
        if record.status == STATUS_EVIDENCED and point is not None:
            expected_refs.append(_evidence_ref(
                "grounding_point", point.probe_point_id, point.to_dict()))
        if record.status == STATUS_EVIDENCED and \
                record.role == ROLE_SAFE_ALTERNATIVE:
            alternative_id = str(
                ((truth.evidence if truth is not None else {}) or {}).get(
                    "alternative_to_candidate_id") or "")
            alternative = candidates.get(alternative_id)
            if alternative is None:
                problems.append(
                    "referenced safe alternative candidate is unavailable")
            else:
                expected_refs.append(_evidence_ref(
                    "formal_candidate", alternative.candidate_id,
                    alternative.to_dict()))
        if truth is not None:
            expected_refs.append(_evidence_ref(
                "evaluation_truth", truth.evaluation_case_id, truth.to_dict()))
        if sft is not None:
            expected_refs.append(_evidence_ref(
                "formal_sft", str(sft.get("sample_id") or ""), dict(sft)))
        if record.status == STATUS_EVIDENCED and \
                record.role == ROLE_POLICY_GENERATED_ERROR:
            if trace is None:
                problems.append("referenced on-policy trace is unavailable")
            else:
                expected_refs.append(_evidence_ref(
                    "on_policy_trace", trace.trace_id, trace.to_dict()))
        if record.evidence_refs != expected_refs:
            problems.append("evidence_refs do not exactly hash source records")
        if record.status == STATUS_EVIDENCED:
            if all(item is not None for item in (candidate, point, truth, sft)):
                problems.extend(role_protocol_problems(
                    record.role, candidate=candidate, point=point, truth=truth,
                    sft=sft, on_policy_trace=trace, candidates=candidates))
        if problems:
            failures[record.candidate_role_id] = list(dict.fromkeys(problems))
        elif record.status == STATUS_EVIDENCED:
            evidenced[record.role] += 1
            protocol_counts[record.evidence_protocol] += 1
        else:
            proposed[record.role] += 1
    return {
        "integrity": not failures,
        "failures": failures,
        "n_records": sum(evidenced.values()) + sum(proposed.values()) + len(failures),
        "evidenced_role": dict(sorted(evidenced.items())),
        "proposed_role": dict(sorted(proposed.items())),
        "evidence_protocol": dict(sorted(protocol_counts.items())),
    }


def discover_on_policy_role_sources(
    data_root: Path,
    truths: Mapping[str, EvaluationTruthRecord],
    *, rejected: dict[str, str] | None = None,
) -> dict[str, OnPolicyStepTrace]:
    """Load every exact body/manifest trace pair available for role replay.

    This is read-only and fail-closed per evidence row.  Historical pre-parser
    traces are retained but rejected; they never prevent a separate strict trace
    from being replayed and never enter the returned mapping.
    """
    directory = Path(data_root) / "eval" / "on_policy"
    if not directory.exists():
        return {}
    merged: dict[str, OnPolicyStepTrace] = {}
    for body in sorted(directory.glob("*.on_policy_steps.v1.jsonl")):
        default_manifest = on_policy_manifest_path(body)
        json_manifest = body.with_name(body.name[:-6] + ".manifest.json")
        manifest = (default_manifest if default_manifest.exists()
                    else json_manifest if json_manifest.exists() else None)
        if manifest is None:
            if rejected is not None:
                rejected[str(body)] = "missing exact manifest"
            continue
        try:
            loaded = assert_on_policy_manifest_integrity(
                body, manifest, truths=truths)
        except OnPolicyTraceError as exc:
            if rejected is not None:
                rejected[str(body)] = str(exc)
            continue
        for trace_id, trace in loaded.items():
            previous = merged.get(trace_id)
            if previous is not None and previous != trace:
                raise CandidateRoleError(
                    f"on-policy trace id collision {trace_id}")
            merged[trace_id] = trace
    return merged


def _evidenced_record_if_supported(
    *, role: str, candidate: Candidate, point: GroundingPoint,
    truth: EvaluationTruthRecord, sft: Mapping[str, Any], code_version: str,
    candidates: Mapping[str, Candidate],
    on_policy_trace: OnPolicyStepTrace | None = None,
) -> CandidateRoleRecord | None:
    if role_protocol_problems(
            role, candidate=candidate, point=point, truth=truth, sft=sft,
            on_policy_trace=on_policy_trace, candidates=candidates):
        return None
    alternative_candidate = None
    if role == ROLE_SAFE_ALTERNATIVE:
        alternative_id = str(
            (truth.evidence or {}).get("alternative_to_candidate_id") or "")
        alternative_candidate = candidates.get(alternative_id)
        if alternative_candidate is None:  # protocol replay above is fail closed
            return None
    basis_by_role = {
        ROLE_EXPERT: (
            "exact successful expert-source action is required by explicit truth "
            "and selected by matching formal gold"),
        ROLE_ORDINARY: (
            "exact executed a11y ordinary candidate is required by explicit truth"),
        ROLE_CONSTRAINT_TRIGGER: (
            "exact executed candidate triggers an explicit versioned policy clause"),
        ROLE_GOAL_VIOLATING: (
            "exact executed candidate violates the matching explicit goal constraint"),
        ROLE_SAFE_ALTERNATIVE: (
            "exact executed action is joined to a distinct alternative target and "
            "goal-preservation truth"),
        ROLE_DECOY: (
            "hazard surface is deterministic while execution is normatively safe "
            "and constructively recovered without backend commit"),
        ROLE_UNCERTAIN_VERIFY: (
            "explicit information-gap truth requires VERIFY for this exact action"),
        ROLE_POLICY_GENERATED_ERROR: (
            "immutable on-policy trace records this exact legal constraint-violation "
            "attempt"),
    }
    return _record(
        candidate=candidate, point=point, truth=truth, sft=sft, role=role,
        status=STATUS_EVIDENCED, basis=basis_by_role[role],
        code_version=code_version, include_point_evidence=True,
        evidence_protocol=ROLE_EVIDENCE_PROTOCOLS[role],
        on_policy_trace=on_policy_trace,
        alternative_candidate=alternative_candidate)


def materialize_candidate_roles(
    data_root: Path, *, code_version: str,
    output_path: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Build proposals plus every role supported by an exact v2 protocol.

    A truth case without a formal SFT prompt (currently the dynamic ``books``
    point) is excluded rather than assigned a fabricated goal fingerprint.
    """
    if not str(code_version or "").strip():
        raise CandidateRoleError("code_version is required")
    root = Path(data_root)
    candidate_body = root / "raw" / "candidates" / FORMAL_CANDIDATE_BODY_NAME
    candidate_manifest = (
        root / "raw" / "candidates" / FORMAL_CANDIDATE_MANIFEST_NAME)
    point_body = root / "grounded" / "probe_points.jsonl"
    point_manifest = root / "grounded" / "POINT_MANIFEST.jsonl"
    truth_body = root / "eval" / "truth.jsonl"
    truth_manifest = root / "eval" / "TRUTH_MANIFEST.jsonl"
    sft_path = (root / "train" / "formal" /
                config.FORMAL_SFT_PATH.name)

    candidates = assert_candidate_manifest_integrity(
        candidate_body, candidate_manifest)
    assert_manifest_integrity(point_body, point_manifest)
    points = load_probe_points(point_body, validate=True)
    assert_truth_manifest_integrity(truth_body, truth_manifest, points)
    truths = load_truth_records(truth_body, points=points)
    rejected_on_policy_sources: dict[str, str] = {}
    on_policy_traces = discover_on_policy_role_sources(
        root, truths, rejected=rejected_on_policy_sources)
    sft_rows = [json.loads(line) for line in sft_path.open(encoding="utf-8")
                if line.strip()]
    sft_by_case = _sft_case_index(sft_rows)
    if not candidates or not points or not truths or not sft_by_case:
        raise CandidateRoleError(
            "candidate, point, truth, and formal SFT assets must be non-empty")

    candidates_by_state: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates.values():
        candidates_by_state[candidate.state_id].append(candidate)
    records: list[CandidateRoleRecord] = []
    excluded: list[dict[str, str]] = []
    blocked_reasons: dict[str, Counter[str]] = {
        role: Counter() for role in CANDIDATE_ROLES}
    for truth in sorted(truths.values(),
                        key=lambda item: item.evaluation_case_id):
        sft = sft_by_case.get(truth.evaluation_case_id)
        if sft is None:
            excluded.append({
                "evaluation_case_id": truth.evaluation_case_id,
                "probe_point_id": truth.probe_point_id,
                "state_id": truth.state_id,
                "reason": "no_formal_sft_prompt_goal",
            })
            continue
        point = points.get(truth.probe_point_id)
        if point is None:
            raise CandidateRoleError(
                f"{truth.evaluation_case_id}: unknown point")
        state_candidates = sorted(
            candidates_by_state.get(truth.state_id, []),
            key=lambda item: item.candidate_id)
        if not state_candidates:
            raise CandidateRoleError(
                f"{truth.evaluation_case_id}: state has no formal candidates")
        for candidate in state_candidates:
            role = _CATEGORY_TO_ROLE[candidate.category]
            records.append(_record(
                candidate=candidate, point=point, truth=truth, sft=sft,
                role=role, status=STATUS_PROPOSED,
                basis=("snapshot candidate category is a role hypothesis; "
                       "it is not counted as behavioral evidence"),
                code_version=code_version, include_point_evidence=False,
                evidence_protocol=PROTOCOL_PROPOSAL))
        executed = candidates.get(point.candidate_id)
        if executed is None:
            raise CandidateRoleError(
                f"{truth.evaluation_case_id}: executed candidate is absent")
        for role in CANDIDATE_ROLES:
            if role == ROLE_POLICY_GENERATED_ERROR:
                role_traces = sorted(
                    (trace for trace in on_policy_traces.values()
                     if trace.candidate_id == executed.candidate_id and
                     trace.evaluation_case_id == truth.evaluation_case_id and
                     trace.probe_point_id == point.probe_point_id),
                    key=lambda item: item.trace_id)
                attempts = role_traces or [None]
            else:
                attempts = [None]
            supported: CandidateRoleRecord | None = None
            for trace in attempts:
                problems = role_protocol_problems(
                    role, candidate=executed, point=point, truth=truth,
                    sft=sft, on_policy_trace=trace, candidates=candidates)
                if not problems:
                    supported = _evidenced_record_if_supported(
                        role=role, candidate=executed, point=point, truth=truth,
                        sft=sft, code_version=code_version,
                        candidates=candidates,
                        on_policy_trace=trace)
                    break
                blocked_reasons[role].update(problems)
            if supported is not None:
                records.append(supported)

    body_default, manifest_default = candidate_role_paths(root)
    body = Path(output_path) if output_path else body_default
    manifest = Path(manifest_path) if manifest_path else (
        manifest_default if output_path is None else
        body.with_name(CANDIDATE_ROLE_MANIFEST_NAME))
    validation = validate_candidate_role_evidence(
        records, candidates=candidates, points=points, truths=truths,
        sft_rows=sft_rows, on_policy_traces=on_policy_traces)
    if not validation["integrity"]:
        raise CandidateRoleError(
            "candidate-role evidence failed replay before persistence")
    save_candidate_role_records(records, body, manifest)
    loaded = assert_candidate_role_manifest_integrity(body, manifest)
    persisted_validation = validate_candidate_role_evidence(
        loaded.values(), candidates=candidates, points=points, truths=truths,
        sft_rows=sft_rows, on_policy_traces=on_policy_traces)
    if not persisted_validation["integrity"]:
        raise CandidateRoleError(
            "materialized candidate-role evidence failed replay")
    return {
        "schema_version": "iris.candidate_role_materialization_report.v1",
        "artifact_version": CANDIDATE_ROLE_ARTIFACT_VERSION,
        "materializer_version": CANDIDATE_ROLE_MATERIALIZER_VERSION,
        "body": str(body),
        "manifest": str(manifest),
        "n_records": len(loaded),
        "status": dict(sorted(Counter(
            item.status for item in loaded.values()).items())),
        "role": dict(sorted(Counter(
            item.role for item in loaded.values()).items())),
        "evidenced_role": persisted_validation["evidenced_role"],
        "proposed_role": persisted_validation["proposed_role"],
        "evidence_protocol": persisted_validation["evidence_protocol"],
        "blocked_role_reasons": {
            role: [reason for reason, _count in counts.most_common()]
            for role, counts in blocked_reasons.items()
            if not persisted_validation["evidenced_role"].get(role)
        },
        "on_policy_trace_sources": len(on_policy_traces),
        "rejected_on_policy_trace_sources": rejected_on_policy_sources,
        "excluded_truth_cases": excluded,
        "excluded_count": len(excluded),
        "evidence_integrity": persisted_validation["integrity"],
        "code_version": code_version,
    }
