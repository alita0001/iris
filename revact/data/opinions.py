"""Point-keyed opinion labels kept outside the formal truth/grounding chain.

Opinion labels answer what a human or an LLM *believes* about an action.  They
are useful for measuring disagreement with execute--then--undo observations,
but they are not behavioral measurements and they are not policy truth.  This
module therefore uses a deliberately incompatible schema and vocabulary:

* ``perceived_effect`` is an opinion, never ``GroundingPoint.effect_status``;
* ``perceived_recoverability`` is an opinion, never
  ``GroundingPoint.recovery_status``;
* ``normative_risk_opinion`` is not ``EvaluationTruthRecord.normative_risk``.

The module only imports ratings that have already been collected.  It has no
HTTP or model-client dependency and cannot call an API.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from ..grounding.schema import (
    EFFECT_CHANGED,
    EFFECT_NO_EFFECT,
    EFFECT_STATUSES,
    EFFECT_UNKNOWN,
    GROUNDING_SCHEMA_VERSION,
    RECOVERY_NOT_WITHIN_BUDGET,
    RECOVERY_PARTIAL,
    RECOVERY_RECOVERED,
    RECOVERY_STATUSES,
    RECOVERY_UNKNOWN,
    GroundingPoint,
)

if TYPE_CHECKING:
    from ..eval.truth import EvaluationTruthRecord


LEGACY_OPINION_LABEL_SCHEMA_VERSION = "iris.opinion_label.v1"
OPINION_LABEL_SCHEMA_VERSION = "iris.opinion_label.v2"
OPINION_MANIFEST_SCHEMA_VERSION = "iris.opinion_label_manifest.v2"
OPINION_IMPORT_SCHEMA_VERSION = "iris.opinion_label_import.v2"
OPINION_MATRIX_SCHEMA_VERSION = "iris.opinion_label_matrix.v2"
OPINION_ARTIFACT_ROLE = "opinion_baseline_only_not_ground_truth"

RATER_HUMAN = "HUMAN"
RATER_LLM = "LLM"
RATER_TYPES = (RATER_HUMAN, RATER_LLM)

PERCEIVED_CHANGE = "CHANGE"
PERCEIVED_NO_CHANGE = "NO_CHANGE"
PERCEIVED_EFFECTS = (PERCEIVED_CHANGE, PERCEIVED_NO_CHANGE, "UNKNOWN")

PERCEIVED_RECOVERABLE = "RECOVERABLE"
PERCEIVED_PARTIAL = "PARTIALLY_RECOVERABLE"
PERCEIVED_NOT_RECOVERABLE = "NOT_RECOVERABLE"
PERCEIVED_RECOVERABILITIES = (
    PERCEIVED_RECOVERABLE,
    PERCEIVED_PARTIAL,
    PERCEIVED_NOT_RECOVERABLE,
    "UNKNOWN",
)

RISKY = "RISKY"
NOT_RISKY = "NOT_RISKY"
NORMATIVE_RISK_OPINIONS = (RISKY, NOT_RISKY, "UNKNOWN")

UNKNOWN_POLICIES = ("separate", "exclude", "error")

_RESERVED_FORMAL_FILENAMES = {
    "probe_points.jsonl",
    "point_manifest.jsonl",
    "evaluation_truth.jsonl",
    "truth_manifest.jsonl",
}
_CREDENTIAL_KEY = re.compile(
    r"(?:api[_-]?key|authorization|credential|password|secret|access[_-]?token|"
    r"refresh[_-]?token)",
    re.IGNORECASE,
)
_CREDENTIAL_VALUE = re.compile(
    r"(?:\bBearer\s+\S{8,}|\bsk-(?:or-v1-)?[A-Za-z0-9_-]{12,}|"
    r"(?:api[_-]?key|password|secret|token)\s*[:=]\s*\S{6,})",
    re.IGNORECASE,
)


class OpinionLabelError(ValueError):
    """An opinion artifact is malformed, ambiguous, or unsafe to persist."""


def _credential_material(value: Any, *, path: str = "record") -> str | None:
    """Return the first credential-like path, without returning the value."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if _CREDENTIAL_KEY.search(key_text):
                return f"{path}.{key_text}"
            found = _credential_material(item, path=f"{path}.{key_text}")
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _credential_material(item, path=f"{path}[{index}]")
            if found:
                return found
    elif isinstance(value, str) and _CREDENTIAL_VALUE.search(value):
        return path
    return None


def _canonical_bytes(row: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def make_opinion_label_id(
    probe_point_id: str,
    evaluation_case_id: str,
    goal_sha256: str,
    rater_id: str,
    instrument_id: str,
    instrument_version: str,
) -> str:
    """Create a stable ID without including a rating value in the identity."""
    payload = {
        "probe_point_id": probe_point_id,
        "evaluation_case_id": evaluation_case_id,
        "goal_sha256": goal_sha256,
        "rater_id": rater_id,
        "instrument_id": instrument_id,
        "instrument_version": instrument_version,
    }
    return "opinion-" + hashlib.sha256(_canonical_bytes(payload)).hexdigest()[:20]


@dataclass(frozen=True)
class OpinionLabelRecord:
    """One opinion for one point×goal evaluation case and one rater."""

    schema_version: str
    artifact_role: str
    opinion_label_id: str
    probe_point_id: str
    state_id: str
    evaluation_case_id: str
    variant: str
    goal_sha256: str
    opinion_input_sha256: str
    input_messages_sha256: str
    raw_response: str
    raw_response_sha256: str
    provider_response_id: str
    response_model: str
    finish_reason: str
    rater_id: str
    rater_type: str
    provider: str
    model: str | None
    prompt_generation_fp: str
    instrument_id: str
    instrument_version: str
    perceived_effect: str
    perceived_recoverability: str
    normative_risk_opinion: str
    confidence: float | None
    rationale: str
    source_record_id: str
    collection_timestamp: str
    import_batch_id: str
    code_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def context_rater_key(self) -> tuple[str, str, str]:
        return self.probe_point_id, self.evaluation_case_id, self.rater_id

    @property
    def point_rater_key(self) -> tuple[str, str, str]:
        """Compatibility name; the key is context-aware in schema v2."""
        return self.context_rater_key

    def validation_errors(
        self,
        point: GroundingPoint | None = None,
        truth: "EvaluationTruthRecord | None" = None,
    ) -> list[str]:
        errors: list[str] = []
        if self.schema_version != OPINION_LABEL_SCHEMA_VERSION:
            errors.append("bad schema_version")
        if self.artifact_role != OPINION_ARTIFACT_ROLE:
            errors.append("bad artifact_role")
        for name in (
            "opinion_label_id",
            "probe_point_id",
            "state_id",
            "evaluation_case_id",
            "variant",
            "goal_sha256",
            "opinion_input_sha256",
            "input_messages_sha256",
            "raw_response",
            "raw_response_sha256",
            "response_model",
            "rater_id",
            "provider",
            "prompt_generation_fp",
            "instrument_id",
            "instrument_version",
            "source_record_id",
            "collection_timestamp",
            "import_batch_id",
            "code_version",
        ):
            if not str(getattr(self, name) or "").strip():
                errors.append(f"missing {name}")
        for name in (
            "goal_sha256",
            "opinion_input_sha256",
            "input_messages_sha256",
            "raw_response_sha256",
        ):
            value = str(getattr(self, name) or "")
            if value and not re.fullmatch(r"[0-9a-f]{64}", value):
                errors.append(f"{name} must be SHA-256")
        if self.raw_response and hashlib.sha256(
                self.raw_response.encode("utf-8")).hexdigest() != \
                self.raw_response_sha256:
            errors.append("raw_response_sha256 mismatch")
        if self.rater_type not in RATER_TYPES:
            errors.append("invalid rater_type")
        if self.rater_type == RATER_LLM and not str(self.model or "").strip():
            errors.append("LLM rater requires model")
        if self.rater_type == RATER_HUMAN and self.model not in (None, ""):
            errors.append("HUMAN rater must not claim a model")
        if self.perceived_effect not in PERCEIVED_EFFECTS:
            errors.append("invalid perceived_effect")
        if self.perceived_recoverability not in PERCEIVED_RECOVERABILITIES:
            errors.append("invalid perceived_recoverability")
        if self.normative_risk_opinion not in NORMATIVE_RISK_OPINIONS:
            errors.append("invalid normative_risk_opinion")
        try:
            raw = json.loads(self.raw_response)
        except (TypeError, json.JSONDecodeError):
            raw = None
            errors.append("raw_response must be strict JSON")
        if isinstance(raw, dict):
            expected_keys = {
                "effect", "recovery", "normative_risk", "confidence", "rationale"
            }
            if set(raw) != expected_keys:
                errors.append("raw_response keys differ from opinion instrument")
            else:
                expected_raw = {
                    "effect": self.perceived_effect,
                    "recovery": self.perceived_recoverability,
                    "normative_risk": self.normative_risk_opinion,
                    "confidence": self.confidence,
                    "rationale": self.rationale,
                }
                if raw != expected_raw:
                    errors.append("raw_response does not reconstruct opinion axes")
        elif raw is not None:
            errors.append("raw_response must be one JSON object")
        if self.confidence is not None and (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            errors.append("confidence must be nullable and within [0,1]")
        expected_id = make_opinion_label_id(
            self.probe_point_id,
            self.evaluation_case_id,
            self.goal_sha256,
            self.rater_id,
            self.instrument_id,
            self.instrument_version,
        )
        if self.opinion_label_id and self.opinion_label_id != expected_id:
            errors.append(
                "opinion_label_id does not match point/case/goal/rater/instrument identity"
            )
        leaked = _credential_material(self.to_dict())
        if leaked:
            errors.append(f"credential-like material is forbidden at {leaked}")
        if point is not None:
            if point.schema_version != GROUNDING_SCHEMA_VERSION:
                errors.append("joined point is not canonical grounding v1")
            if self.probe_point_id != point.probe_point_id:
                errors.append("joined probe_point_id mismatch")
            if self.state_id != point.state_id:
                errors.append("joined state_id mismatch")
        if truth is not None:
            if self.evaluation_case_id != truth.evaluation_case_id:
                errors.append("joined evaluation_case_id mismatch")
            if self.probe_point_id != truth.probe_point_id:
                errors.append("joined truth probe_point_id mismatch")
            if self.state_id != truth.state_id:
                errors.append("joined truth state_id mismatch")
            if self.variant != truth.variant:
                errors.append("joined truth variant mismatch")
        return errors

    def validate(
        self,
        point: GroundingPoint | None = None,
        truth: "EvaluationTruthRecord | None" = None,
    ) -> None:
        errors = self.validation_errors(point, truth)
        if errors:
            raise OpinionLabelError(
                f"{self.opinion_label_id or '<missing opinion id>'}: "
                + "; ".join(errors)
            )

    @classmethod
    def from_dict(
        cls,
        row: Mapping[str, Any],
        *,
        point: GroundingPoint | None = None,
        truth: "EvaluationTruthRecord | None" = None,
    ) -> "OpinionLabelRecord":
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(row) - known)
        missing = sorted(known - set(row))
        if unknown:
            raise OpinionLabelError(f"unknown opinion fields: {unknown}")
        if missing:
            raise OpinionLabelError(f"missing serialized opinion fields: {missing}")
        leaked = _credential_material(row)
        if leaked:
            raise OpinionLabelError(
                f"credential-like material is forbidden at {leaked}"
            )
        try:
            record = cls(**dict(row))
        except TypeError as exc:
            raise OpinionLabelError(str(exc)) from exc
        record.validate(point, truth)
        return record


def opinion_manifest_row(record: OpinionLabelRecord) -> dict[str, Any]:
    return {
        "schema_version": OPINION_MANIFEST_SCHEMA_VERSION,
        "artifact_role": OPINION_ARTIFACT_ROLE,
        "opinion_schema_version": record.schema_version,
        "opinion_label_id": record.opinion_label_id,
        "probe_point_id": record.probe_point_id,
        "state_id": record.state_id,
        "evaluation_case_id": record.evaluation_case_id,
        "variant": record.variant,
        "goal_sha256": record.goal_sha256,
        "opinion_input_sha256": record.opinion_input_sha256,
        "input_messages_sha256": record.input_messages_sha256,
        "raw_response_sha256": record.raw_response_sha256,
        "provider_response_id": record.provider_response_id,
        "response_model": record.response_model,
        "finish_reason": record.finish_reason,
        "rater_id": record.rater_id,
        "rater_type": record.rater_type,
        "prompt_generation_fp": record.prompt_generation_fp,
        "record_sha256": hashlib.sha256(
            _canonical_bytes(record.to_dict())
        ).hexdigest(),
    }


def _jsonl_text(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )


def _stage_file(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return temporary


def _atomic_write_pair(
    body: Path,
    body_text: str,
    manifest: Path,
    manifest_text: str,
) -> None:
    """Stage both files before replacement and roll back ordinary failures.

    No filesystem offers a portable two-path atomic rename.  We therefore
    stage and fsync both new files, retain staged copies of an existing pair,
    roll both paths back if either rename raises, and use the 1:1 hashes to
    detect a process/power-loss tear that cannot be rolled back in-process.
    """
    had_body = body.exists()
    had_manifest = manifest.exists()
    if had_body != had_manifest:
        raise OpinionLabelError("opinion body/manifest must both exist or both be absent")
    body_tmp = ""
    manifest_tmp = ""
    body_backup = ""
    manifest_backup = ""
    body_replaced = False
    manifest_replaced = False
    try:
        body_tmp = _stage_file(body, body_text)
        manifest_tmp = _stage_file(manifest, manifest_text)
        if had_body:
            body_backup = _stage_file(body, body.read_text(encoding="utf-8"))
            manifest_backup = _stage_file(
                manifest, manifest.read_text(encoding="utf-8")
            )
        try:
            os.replace(body_tmp, body)
            body_tmp = ""
            body_replaced = True
            os.replace(manifest_tmp, manifest)
            manifest_tmp = ""
            manifest_replaced = True
        except BaseException:
            # Preserve the old valid pair on an ordinary rename failure.  A
            # crash between renames is instead detected by manifest integrity.
            if had_body and body_replaced:
                os.replace(body_backup, body)
                body_backup = ""
            elif not had_body and body_replaced and body.exists():
                body.unlink()
            if had_manifest and manifest_replaced:
                os.replace(manifest_backup, manifest)
                manifest_backup = ""
            elif not had_manifest and manifest_replaced and manifest.exists():
                manifest.unlink()
            raise
        for directory in {body.parent, manifest.parent}:
            try:
                fd = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except OSError:
                # Directory fsync is unavailable on some test filesystems; the
                # file contents were already fsynced before replacement.
                pass
    finally:
        for temporary in (body_tmp, manifest_tmp, body_backup, manifest_backup):
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)


def _validate_paths(body: Path, manifest: Path) -> None:
    if body.resolve() == manifest.resolve():
        raise OpinionLabelError("opinion body and manifest must be distinct")
    for path in (body, manifest):
        if path.name.lower() in _RESERVED_FORMAL_FILENAMES:
            raise OpinionLabelError(
                f"opinion artifacts cannot use formal truth/grounding filename {path.name}"
            )


def _validated_record_set(
    records: Iterable[OpinionLabelRecord],
    *,
    points: Mapping[str, GroundingPoint] | None = None,
    truths: Mapping[str, "EvaluationTruthRecord"] | None = None,
) -> dict[str, OpinionLabelRecord]:
    by_id: dict[str, OpinionLabelRecord] = {}
    by_context_rater: dict[tuple[str, str, str], str] = {}
    rater_provenance: dict[str, tuple[Any, ...]] = {}
    case_provenance: dict[str, tuple[Any, ...]] = {}
    for record in records:
        point = points.get(record.probe_point_id) if points is not None else None
        if points is not None and point is None:
            raise OpinionLabelError(
                f"unknown probe_point_id {record.probe_point_id!r}"
            )
        truth = (truths.get(record.evaluation_case_id)
                 if truths is not None else None)
        if truths is not None and truth is None:
            raise OpinionLabelError(
                f"unknown evaluation_case_id {record.evaluation_case_id!r}"
            )
        record.validate(point, truth)
        old = by_id.get(record.opinion_label_id)
        if old is not None and old != record:
            raise OpinionLabelError(
                f"immutable opinion collision {record.opinion_label_id}"
            )
        key = record.context_rater_key
        prior_id = by_context_rater.get(key)
        if prior_id is not None and prior_id != record.opinion_label_id:
            raise OpinionLabelError(f"duplicate point/case/rater opinion {key}")
        provenance = (
            record.rater_type,
            record.provider,
            record.model,
            record.prompt_generation_fp,
            record.instrument_id,
            record.instrument_version,
        )
        prior_provenance = rater_provenance.get(record.rater_id)
        if prior_provenance is not None and prior_provenance != provenance:
            raise OpinionLabelError(
                f"inconsistent provenance for rater {record.rater_id!r}"
            )
        context = (
            record.probe_point_id,
            record.state_id,
            record.variant,
            record.goal_sha256,
            record.opinion_input_sha256,
        )
        prior_context = case_provenance.get(record.evaluation_case_id)
        if prior_context is not None and prior_context != context:
            raise OpinionLabelError(
                f"inconsistent input provenance for evaluation case "
                f"{record.evaluation_case_id!r}"
            )
        by_id[record.opinion_label_id] = record
        by_context_rater[key] = record.opinion_label_id
        rater_provenance[record.rater_id] = provenance
        case_provenance[record.evaluation_case_id] = context
    return by_id


def load_opinion_records(
    path: Path,
    *,
    points: Mapping[str, GroundingPoint] | None = None,
    truths: Mapping[str, "EvaluationTruthRecord"] | None = None,
) -> dict[str, OpinionLabelRecord]:
    records: list[OpinionLabelRecord] = []
    if not Path(path).exists():
        return {}
    for line_no, line in enumerate(Path(path).open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            point_id = str(row.get("probe_point_id") or "")
            point = points.get(point_id) if points is not None else None
            case_id = str(row.get("evaluation_case_id") or "")
            truth = truths.get(case_id) if truths is not None else None
            records.append(OpinionLabelRecord.from_dict(
                row, point=point, truth=truth))
        except (json.JSONDecodeError, OpinionLabelError) as exc:
            raise OpinionLabelError(f"{path}:{line_no}: {exc}") from exc
    return _validated_record_set(records, points=points, truths=truths)


def save_opinion_records(
    records: Iterable[OpinionLabelRecord],
    body: Path,
    manifest: Path,
    *,
    append: bool = False,
    points: Mapping[str, GroundingPoint] | None = None,
    truths: Mapping[str, "EvaluationTruthRecord"] | None = None,
) -> tuple[Path, Path]:
    """Validate and atomically materialize an opinion-only body and manifest."""
    body, manifest = Path(body), Path(manifest)
    _validate_paths(body, manifest)
    if body.exists() != manifest.exists():
        raise OpinionLabelError("opinion body/manifest must both exist or both be absent")
    incoming = _validated_record_set(records, points=points, truths=truths)
    existing = (load_opinion_records(
        body, points=points, truths=truths) if body.exists() else {})
    if body.exists():
        assert_opinion_manifest_integrity(
            body, manifest, points=points, truths=truths)
        if not append:
            if existing == incoming:
                return body, manifest
            raise OpinionLabelError(
                "refusing to overwrite immutable opinion artifact; use append=True "
                "or a new versioned path"
            )
    merged = dict(existing)
    for opinion_id, record in incoming.items():
        old = merged.get(opinion_id)
        if old is not None and old != record:
            raise OpinionLabelError(f"immutable opinion collision {opinion_id}")
        merged[opinion_id] = record
    validated = _validated_record_set(
        merged.values(), points=points, truths=truths)
    if not validated:
        raise OpinionLabelError("refusing to materialize an empty opinion artifact")
    ordered = [validated[key] for key in sorted(validated)]
    _atomic_write_pair(
        body,
        _jsonl_text(record.to_dict() for record in ordered),
        manifest,
        _jsonl_text(opinion_manifest_row(record) for record in ordered),
    )
    assert_opinion_manifest_integrity(
        body, manifest, points=points, truths=truths)
    return body, manifest


def assert_opinion_manifest_integrity(
    body: Path,
    manifest: Path,
    *,
    points: Mapping[str, GroundingPoint] | None = None,
    truths: Mapping[str, "EvaluationTruthRecord"] | None = None,
) -> None:
    _validate_paths(Path(body), Path(manifest))
    if not Path(body).exists() or not Path(manifest).exists():
        raise OpinionLabelError("opinion body and manifest are both required")
    records = load_opinion_records(Path(body), points=points, truths=truths)
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(Path(manifest).open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OpinionLabelError(f"{manifest}:{line_no}: {exc}") from exc
        leaked = _credential_material(row)
        if leaked:
            raise OpinionLabelError(
                f"credential-like material is forbidden at {leaked}"
            )
        rows.append(row)
    indexed = {str(row.get("opinion_label_id") or ""): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(records):
        raise OpinionLabelError("opinion body/manifest is not 1:1")
    for opinion_id, record in records.items():
        if indexed[opinion_id] != opinion_manifest_row(record):
            raise OpinionLabelError(f"opinion manifest hash mismatch {opinion_id}")


def _read_json_source(source: Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    path = Path(source)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OpinionLabelError(f"{path}: external import must be one JSON object") from exc
    if not isinstance(payload, Mapping):
        raise OpinionLabelError("external opinion import must be a JSON object")
    return payload


def import_external_opinion_ratings(
    source: Path | Mapping[str, Any],
    *,
    points: Mapping[str, GroundingPoint] | None = None,
    truths: Mapping[str, "EvaluationTruthRecord"] | None = None,
) -> list[OpinionLabelRecord]:
    """Normalize an already-collected external rating batch; never call an API.

    Common instrument and import provenance live in the envelope.  Individual
    rows contain rater provenance and opinions.  Credential-named fields or
    credential-looking values are rejected before construction.
    """
    payload = _read_json_source(source)
    allowed = {
        "schema_version",
        "instrument_id",
        "instrument_version",
        "prompt_generation_fp",
        "collection_timestamp",
        "import_batch_id",
        "code_version",
        "ratings",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise OpinionLabelError(f"unknown external import fields: {unknown}")
    leaked = _credential_material(payload)
    if leaked:
        raise OpinionLabelError(f"credential-like material is forbidden at {leaked}")
    if payload.get("schema_version") != OPINION_IMPORT_SCHEMA_VERSION:
        raise OpinionLabelError("bad external opinion import schema_version")
    for name in (
        "instrument_id",
        "instrument_version",
        "prompt_generation_fp",
        "collection_timestamp",
        "import_batch_id",
        "code_version",
    ):
        if not str(payload.get(name) or "").strip():
            raise OpinionLabelError(f"missing external import {name}")
    ratings = payload.get("ratings")
    if not isinstance(ratings, list) or not ratings:
        raise OpinionLabelError("external import ratings must be a non-empty list")
    rating_fields = {
        "probe_point_id",
        "state_id",
        "evaluation_case_id",
        "variant",
        "goal_sha256",
        "opinion_input_sha256",
        "input_messages_sha256",
        "raw_response",
        "raw_response_sha256",
        "provider_response_id",
        "response_model",
        "finish_reason",
        "rater_id",
        "rater_type",
        "provider",
        "model",
        "perceived_effect",
        "perceived_recoverability",
        "normative_risk_opinion",
        "confidence",
        "rationale",
        "source_record_id",
    }
    records: list[OpinionLabelRecord] = []
    for index, rating in enumerate(ratings):
        if not isinstance(rating, Mapping):
            raise OpinionLabelError(f"rating[{index}] must be an object")
        unknown_rating = sorted(set(rating) - rating_fields)
        missing_rating = sorted(rating_fields - set(rating))
        if unknown_rating:
            raise OpinionLabelError(
                f"rating[{index}] unknown fields: {unknown_rating}"
            )
        if missing_rating:
            raise OpinionLabelError(
                f"rating[{index}] missing fields: {missing_rating}"
            )
        point_id = str(rating["probe_point_id"])
        case_id = str(rating["evaluation_case_id"])
        rater_id = str(rating["rater_id"])
        record = OpinionLabelRecord(
            schema_version=OPINION_LABEL_SCHEMA_VERSION,
            artifact_role=OPINION_ARTIFACT_ROLE,
            opinion_label_id=make_opinion_label_id(
                point_id,
                case_id,
                str(rating["goal_sha256"]),
                rater_id,
                str(payload["instrument_id"]),
                str(payload["instrument_version"]),
            ),
            probe_point_id=point_id,
            state_id=str(rating["state_id"]),
            evaluation_case_id=case_id,
            variant=str(rating["variant"]),
            goal_sha256=str(rating["goal_sha256"]),
            opinion_input_sha256=str(rating["opinion_input_sha256"]),
            input_messages_sha256=str(rating["input_messages_sha256"]),
            raw_response=str(rating["raw_response"]),
            raw_response_sha256=str(rating["raw_response_sha256"]),
            provider_response_id=str(rating["provider_response_id"]),
            response_model=str(rating["response_model"]),
            finish_reason=str(rating["finish_reason"]),
            rater_id=rater_id,
            rater_type=str(rating["rater_type"]),
            provider=str(rating["provider"]),
            model=(None if rating["model"] is None else str(rating["model"])),
            prompt_generation_fp=str(payload["prompt_generation_fp"]),
            instrument_id=str(payload["instrument_id"]),
            instrument_version=str(payload["instrument_version"]),
            perceived_effect=str(rating["perceived_effect"]),
            perceived_recoverability=str(rating["perceived_recoverability"]),
            normative_risk_opinion=str(rating["normative_risk_opinion"]),
            confidence=rating["confidence"],
            rationale=str(rating["rationale"]),
            source_record_id=str(rating["source_record_id"]),
            collection_timestamp=str(payload["collection_timestamp"]),
            import_batch_id=str(payload["import_batch_id"]),
            code_version=str(payload["code_version"]),
        )
        point = points.get(point_id) if points is not None else None
        truth = truths.get(case_id) if truths is not None else None
        if points is not None and point is None:
            raise OpinionLabelError(f"unknown probe_point_id {point_id!r}")
        if truths is not None and truth is None:
            raise OpinionLabelError(f"unknown evaluation_case_id {case_id!r}")
        record.validate(point, truth)
        records.append(record)
    return [
        _validated_record_set(records, points=points, truths=truths)[key]
        for key in sorted(_validated_record_set(
            records, points=points, truths=truths))
    ]


def import_opinion_records(
    source: Path | Sequence[Mapping[str, Any]],
    *,
    points: Mapping[str, GroundingPoint] | None = None,
    truths: Mapping[str, "EvaluationTruthRecord"] | None = None,
) -> list[OpinionLabelRecord]:
    """Import canonical records from JSONL or an in-memory sequence, offline."""
    rows: list[Mapping[str, Any]] = []
    if isinstance(source, (str, os.PathLike, Path)):
        path = Path(source)
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OpinionLabelError(f"{path}:{line_no}: {exc}") from exc
            if not isinstance(row, Mapping):
                raise OpinionLabelError(f"{path}:{line_no}: row must be an object")
            rows.append(row)
    else:
        rows = list(source)
    records: list[OpinionLabelRecord] = []
    for row in rows:
        point_id = str(row.get("probe_point_id") or "")
        point = points.get(point_id) if points is not None else None
        case_id = str(row.get("evaluation_case_id") or "")
        truth = truths.get(case_id) if truths is not None else None
        if points is not None and point is None:
            raise OpinionLabelError(f"unknown probe_point_id {point_id!r}")
        if truths is not None and truth is None:
            raise OpinionLabelError(f"unknown evaluation_case_id {case_id!r}")
        records.append(OpinionLabelRecord.from_dict(
            row, point=point, truth=truth))
    validated = _validated_record_set(
        records, points=points, truths=truths)
    return [validated[key] for key in sorted(validated)]


_LEGACY_V1_FIELDS = {
    "schema_version", "artifact_role", "opinion_label_id", "probe_point_id",
    "state_id", "rater_id", "rater_type", "provider", "model",
    "prompt_generation_fp", "instrument_id", "instrument_version",
    "perceived_effect", "perceived_recoverability", "normative_risk_opinion",
    "confidence", "rationale", "source_record_id", "collection_timestamp",
    "import_batch_id", "code_version",
}


def load_legacy_v1_opinion_rows(path: Path) -> list[dict[str, Any]]:
    """Read legacy point×rater rows for quarantine audit, never promotion.

    V1 cannot identify the goal/evaluation case behind normative risk.  The
    reader therefore returns isolated raw rows and canonical v2 writers never
    accept them.
    """
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(Path(path).open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OpinionLabelError(f"{path}:{line_no}: {exc}") from exc
        if not isinstance(row, dict) or set(row) != _LEGACY_V1_FIELDS:
            raise OpinionLabelError(
                f"{path}:{line_no}: malformed legacy v1 opinion row")
        if row.get("schema_version") != LEGACY_OPINION_LABEL_SCHEMA_VERSION:
            raise OpinionLabelError(
                f"{path}:{line_no}: not an opinion_label.v1 row")
        leaked = _credential_material(row)
        if leaked:
            raise OpinionLabelError(
                f"{path}:{line_no}: credential-like material at {leaked}")
        rows.append({**row, "legacy_audit_only": True,
                     "promotion_blocker": "missing_evaluation_case_and_goal_hash"})
    return rows


def _unknown_policy(value: str) -> str:
    if value not in UNKNOWN_POLICIES:
        raise OpinionLabelError(
            f"unknown_policy must be one of {UNKNOWN_POLICIES}, got {value!r}"
        )
    return value


def _axis_summary(
    values: Iterable[str],
    categories: Sequence[str],
    unknown_policy: str,
) -> dict[str, Any]:
    raw_counter = Counter(values)
    raw = {category: raw_counter.get(category, 0) for category in categories}
    unknown = raw.get("UNKNOWN", 0)
    if unknown_policy == "error" and unknown:
        raise OpinionLabelError("UNKNOWN encountered under unknown_policy=error")
    reported = dict(raw)
    if unknown_policy == "exclude":
        reported.pop("UNKNOWN", None)
    return {
        "raw_counts": raw,
        "reported_counts": reported,
        "unknown_count": unknown,
        "reported_denominator": sum(reported.values()),
    }


def build_point_rater_matrix(
    records: Iterable[OpinionLabelRecord],
    *,
    unknown_policy: str = "separate",
) -> dict[str, Any]:
    """Return exact evaluation-case×rater cells and raw opinion counts."""
    policy = _unknown_policy(unknown_policy)
    validated = _validated_record_set(records)
    if not validated:
        raise OpinionLabelError("evaluation-case/rater matrix requires records")
    ordered = sorted(validated.values(), key=lambda row: row.context_rater_key)
    point_ids = sorted({row.probe_point_id for row in ordered})
    case_ids = sorted({row.evaluation_case_id for row in ordered})
    rater_ids = sorted({row.rater_id for row in ordered})
    by_case = {row.evaluation_case_id: row for row in ordered}
    by_key = {row.context_rater_key: row for row in ordered}
    missing = [
        {
            "probe_point_id": by_case[case_id].probe_point_id,
            "evaluation_case_id": case_id,
            "variant": by_case[case_id].variant,
            "rater_id": rater_id,
        }
        for case_id in case_ids
        for rater_id in rater_ids
        if (by_case[case_id].probe_point_id, case_id, rater_id) not in by_key
    ]
    cells = [
        {
            "probe_point_id": row.probe_point_id,
            "state_id": row.state_id,
            "evaluation_case_id": row.evaluation_case_id,
            "variant": row.variant,
            "goal_sha256": row.goal_sha256,
            "rater_id": row.rater_id,
            "opinion_label_id": row.opinion_label_id,
            "perceived_effect": row.perceived_effect,
            "perceived_recoverability": row.perceived_recoverability,
            "normative_risk_opinion": row.normative_risk_opinion,
        }
        for row in ordered
    ]
    matrices = {
        axis: {
            case_id: {
                rater_id: (
                    getattr(by_key[(
                        by_case[case_id].probe_point_id, case_id, rater_id)], axis)
                    if (by_case[case_id].probe_point_id, case_id, rater_id) in by_key
                    else None
                )
                for rater_id in rater_ids
            }
            for case_id in case_ids
        }
        for axis in (
            "perceived_effect",
            "perceived_recoverability",
            "normative_risk_opinion",
        )
    }
    return {
        "schema_version": OPINION_MATRIX_SCHEMA_VERSION,
        "artifact_role": OPINION_ARTIFACT_ROLE,
        "matrix_kind": "evaluation_case_x_rater",
        "unknown_policy": policy,
        "point_ids": point_ids,
        "evaluation_case_ids": case_ids,
        "rater_ids": rater_ids,
        "cells": cells,
        "matrices": matrices,
        "missing_cells": missing,
        "raw_counts": {
            "n_records": len(ordered),
            "n_points": len(point_ids),
            "n_evaluation_cases": len(case_ids),
            "n_raters": len(rater_ids),
            "n_missing_cells": len(missing),
            "perceived_effect": _axis_summary(
                (row.perceived_effect for row in ordered), PERCEIVED_EFFECTS, policy
            ),
            "perceived_recoverability": _axis_summary(
                (row.perceived_recoverability for row in ordered),
                PERCEIVED_RECOVERABILITIES,
                policy,
            ),
            "normative_risk_opinion": _axis_summary(
                (row.normative_risk_opinion for row in ordered),
                NORMATIVE_RISK_OPINIONS,
                policy,
            ),
        },
    }


_EFFECT_OPINION_TO_GROUNDING = {
    PERCEIVED_CHANGE: EFFECT_CHANGED,
    PERCEIVED_NO_CHANGE: EFFECT_NO_EFFECT,
}
_RECOVERY_OPINION_TO_GROUNDING = {
    PERCEIVED_RECOVERABLE: RECOVERY_RECOVERED,
    PERCEIVED_PARTIAL: RECOVERY_PARTIAL,
    PERCEIVED_NOT_RECOVERABLE: RECOVERY_NOT_WITHIN_BUDGET,
}


def _formal_points(
    points: Mapping[str, GroundingPoint | Mapping[str, Any]],
) -> dict[str, GroundingPoint]:
    out: dict[str, GroundingPoint] = {}
    for point_id, raw in points.items():
        point = (
            raw
            if isinstance(raw, GroundingPoint)
            else GroundingPoint.from_dict(dict(raw), validate=True)
        )
        point.validate(formal=True)
        if point_id != point.probe_point_id:
            raise OpinionLabelError(
                f"grounding map key mismatch: {point_id!r} != {point.probe_point_id!r}"
            )
        out[point_id] = point
    return out


def _pair_axis(
    pairs: Iterable[tuple[str, str]],
    opinion_categories: Sequence[str],
    grounding_categories: Sequence[str],
    mapping: Mapping[str, str],
    unknown_policy: str,
) -> dict[str, Any]:
    pair_rows = list(pairs)
    raw_matrix = {
        opinion: {grounding: 0 for grounding in grounding_categories}
        for opinion in opinion_categories
    }
    comparable = 0
    disagreements = 0
    unknown_pairs = 0
    for opinion, grounding in pair_rows:
        raw_matrix[opinion][grounding] += 1
        if opinion == "UNKNOWN" or grounding == "UNKNOWN":
            unknown_pairs += 1
            continue
        comparable += 1
        if mapping[opinion] != grounding:
            disagreements += 1
    if unknown_policy == "error" and unknown_pairs:
        raise OpinionLabelError("UNKNOWN pair encountered under unknown_policy=error")
    if unknown_policy == "exclude":
        reported_matrix = {
            opinion: {
                grounding: count
                for grounding, count in columns.items()
                if grounding != "UNKNOWN"
            }
            for opinion, columns in raw_matrix.items()
            if opinion != "UNKNOWN"
        }
    else:
        reported_matrix = raw_matrix
    return {
        "raw_matrix": raw_matrix,
        "reported_matrix": reported_matrix,
        "n_total": len(pair_rows),
        "n_unknown_pairs": unknown_pairs,
        "n_comparable": comparable,
        "n_disagreements": disagreements,
        "disagreement_rate": (
            disagreements / comparable if comparable else None
        ),
    }


def build_behavior_grounding_disagreement_matrix(
    records: Iterable[OpinionLabelRecord],
    points: Mapping[str, GroundingPoint | Mapping[str, Any]],
    *,
    unknown_policy: str = "separate",
) -> dict[str, Any]:
    """Compare perceived behavior with formal grounding, never risk with behavior."""
    policy = _unknown_policy(unknown_policy)
    formal_points = _formal_points(points)
    validated = _validated_record_set(records, points=formal_points)
    if not validated:
        raise OpinionLabelError("behavior-grounding matrix requires records")
    ordered = sorted(validated.values(), key=lambda row: row.point_rater_key)
    rows: list[dict[str, Any]] = []
    for record in ordered:
        point = formal_points[record.probe_point_id]
        effect_disagreement = None
        if (
            record.perceived_effect != "UNKNOWN"
            and point.effect_status != EFFECT_UNKNOWN
        ):
            effect_disagreement = (
                _EFFECT_OPINION_TO_GROUNDING[record.perceived_effect]
                != point.effect_status
            )
        recovery_disagreement = None
        if (
            record.perceived_recoverability != "UNKNOWN"
            and point.recovery_status != RECOVERY_UNKNOWN
        ):
            recovery_disagreement = (
                _RECOVERY_OPINION_TO_GROUNDING[record.perceived_recoverability]
                != point.recovery_status
            )
        rows.append(
            {
                "probe_point_id": record.probe_point_id,
                "state_id": record.state_id,
                "rater_id": record.rater_id,
                "perceived_effect": record.perceived_effect,
                "behavior_effect_status": point.effect_status,
                "effect_disagreement": effect_disagreement,
                "perceived_recoverability": record.perceived_recoverability,
                "behavior_recovery_status": point.recovery_status,
                "recovery_disagreement": recovery_disagreement,
                "normative_risk_opinion": record.normative_risk_opinion,
            }
        )
    effect = _pair_axis(
        ((row["perceived_effect"], row["behavior_effect_status"]) for row in rows),
        PERCEIVED_EFFECTS,
        EFFECT_STATUSES,
        _EFFECT_OPINION_TO_GROUNDING,
        policy,
    )
    recovery = _pair_axis(
        (
            (row["perceived_recoverability"], row["behavior_recovery_status"])
            for row in rows
        ),
        PERCEIVED_RECOVERABILITIES,
        RECOVERY_STATUSES,
        _RECOVERY_OPINION_TO_GROUNDING,
        policy,
    )
    return {
        "schema_version": OPINION_MATRIX_SCHEMA_VERSION,
        "artifact_role": OPINION_ARTIFACT_ROLE,
        "matrix_kind": "behavior_opinion_x_formal_grounding",
        "unknown_policy": policy,
        "rows": rows,
        "effect": effect,
        "recovery": recovery,
        "normative_opinion": _axis_summary(
            (row["normative_risk_opinion"] for row in rows),
            NORMATIVE_RISK_OPINIONS,
            policy,
        ),
        "normative_vs_behavior_comparison_performed": False,
        "separation_note": (
            "normative risk opinions are reported separately; effect/recovery "
            "grounding cannot determine policy permission or safety"
        ),
    }


# Concise alias for callers that use the report name rather than the builder name.
behavior_grounding_disagreement_matrix = build_behavior_grounding_disagreement_matrix
