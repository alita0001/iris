"""Immutable API/DB signal evidence assets for point-level probes.

This module does not decide effect, recovery, or safety. It freezes three
read-only observer responses around an already reviewed probe execution and
returns a point-evidence patch. Formal readiness independently reloads every
raw snapshot and verifies all hashes and cross-phase semantics.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from .backend_observers import (BACKEND_ATTESTATION_SCHEMA_VERSION,
                                BACKEND_PROVIDER_SCHEMA_VERSION)

SIGNAL_EVIDENCE_SCHEMA_VERSION = "iris.signal_evidence.v2"
SIGNAL_EVIDENCE_REF_SCHEMA_VERSION = "iris.signal_evidence_ref.v2"
SIGNAL_SNAPSHOT_SCHEMA_VERSION = "iris.signal_snapshot.v2"
SIGNAL_SNAPSHOT_REF_SCHEMA_VERSION = "iris.signal_snapshot_ref.v2"
SIGNAL_PHASES = ("pre", "post", "final")
SIGNAL_CHANNELS = frozenset({"api", "backend_api", "db", "database"})
_PII_REVIEW_STATUSES = frozenset({"NOT_PRESENT", "REDACTED_AND_REVIEWED"})
_SECRET_KEY = re.compile(
    r"(?:authorization|cookie|password|passwd|secret|api[_-]?key|access[_-]?token|"
    r"refresh[_-]?token|session[_-]?token)",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"(?:\bBearer\s+[A-Za-z0-9._~+/=-]{12,}|"
    r"\bsk-(?:or-)?[A-Za-z0-9_-]{16,}|"
    r"\bBasic\s+[A-Za-z0-9+/=]{12,}|"
    r"(?:mysql|postgres(?:ql)?)://[^\s:@/]+:[^\s@/]+@)",
    re.IGNORECASE,
)
_PROVIDER_FIELDS = {
    "schema_version", "provider_id", "database_system", "transport",
    "query_id", "query_sha256", "source_instance_sha256",
    "read_only_enforcement", "projection_version", "redaction_strategy",
    "redaction_scope_sha256", "redaction_key_persisted",
    "container_image_ref", "container_id_sha256",
    "container_image_id_sha256",
}
_ATTESTATION_FIELDS = {
    "schema_version", "provider_id", "query_id", "transaction_read_only",
    "source_instance_sha256", "result_row_count",
}
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


class SignalEvidenceError(ValueError):
    """Signal evidence is incomplete, mutable, or likely contains credentials."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _secret_paths(value: Any, prefix: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            path = f"{prefix}.{name}"
            if _SECRET_KEY.search(name):
                found.append(path)
            found.extend(_secret_paths(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_secret_paths(item, f"{prefix}[{index}]"))
    elif isinstance(value, str) and _SECRET_VALUE.search(value):
        found.append(prefix)
    return found


def _validate_provider_metadata(provider: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(provider, Mapping) or set(provider) != _PROVIDER_FIELDS:
        actual = sorted(provider) if isinstance(provider, Mapping) else []
        raise SignalEvidenceError(
            "provider metadata fields differ: "
            f"missing={sorted(_PROVIDER_FIELDS - set(actual))}, "
            f"unknown={sorted(set(actual) - _PROVIDER_FIELDS)}")
    value = dict(provider)
    if value["schema_version"] != BACKEND_PROVIDER_SCHEMA_VERSION:
        raise SignalEvidenceError("bad backend provider schema_version")
    for field in (
            "provider_id", "database_system", "transport", "query_id",
            "read_only_enforcement", "projection_version",
            "redaction_strategy", "container_image_ref"):
        if not str(value.get(field) or "").strip():
            raise SignalEvidenceError(f"provider metadata missing {field}")
    for field in (
            "query_sha256", "source_instance_sha256", "redaction_scope_sha256",
            "container_id_sha256", "container_image_id_sha256"):
        if not _HEX_64.fullmatch(str(value.get(field) or "")):
            raise SignalEvidenceError(f"provider {field} is not lowercase sha256")
    if value["redaction_strategy"] != "ephemeral-hmac-sha256":
        raise SignalEvidenceError("unsupported signal redaction strategy")
    if value["redaction_key_persisted"] is not False:
        raise SignalEvidenceError("redaction key must not be persisted")
    if value["transport"] != "docker_exec":
        raise SignalEvidenceError("unsupported signal evidence transport")
    if value["database_system"] not in {"mariadb", "postgresql"}:
        raise SignalEvidenceError("unsupported signal evidence database_system")
    return value


def _validate_attestation(attestation: Any, provider: Mapping[str, Any],
                          raw_payload: Any) -> dict[str, Any]:
    if not isinstance(attestation, Mapping) or set(attestation) != \
            _ATTESTATION_FIELDS:
        raise SignalEvidenceError("read_only_attestation fields differ")
    value = dict(attestation)
    if value["schema_version"] != BACKEND_ATTESTATION_SCHEMA_VERSION:
        raise SignalEvidenceError("bad read_only_attestation schema_version")
    for field in ("provider_id", "query_id", "source_instance_sha256"):
        expected = provider[field]
        if value.get(field) != expected:
            raise SignalEvidenceError(
                f"read_only_attestation {field} contradicts provider")
    if value.get("transaction_read_only") is not True:
        raise SignalEvidenceError("database transaction was not read-only")
    if not isinstance(value.get("result_row_count"), int) or \
            value["result_row_count"] < 0:
        raise SignalEvidenceError("result_row_count must be non-negative")
    if not isinstance(raw_payload, Mapping):
        raise SignalEvidenceError("raw_payload must be an object")
    if raw_payload.get("payload_semantics") != \
            "minimized_redacted_projection":
        raise SignalEvidenceError("raw payload is not a minimized projection")
    if raw_payload.get("transaction_read_only") is not True:
        raise SignalEvidenceError("raw payload lacks read-only attestation")
    if raw_payload.get("query_id") != provider["query_id"]:
        raise SignalEvidenceError("raw payload query_id contradicts provider")
    if raw_payload.get("row_count") != value["result_row_count"]:
        raise SignalEvidenceError("raw payload row_count contradicts attestation")
    rows = raw_payload.get("rows")
    if not isinstance(rows, list) or len(rows) != value["result_row_count"]:
        raise SignalEvidenceError("raw payload rows contradict result_row_count")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"row_token", "state"}:
            raise SignalEvidenceError(
                f"raw payload row {index} is not a minimized token/state row")
        if not _HEX_64.fullmatch(str(row.get("row_token") or "")):
            raise SignalEvidenceError(
                f"raw payload row {index} has invalid row_token")
        canonical_sha256(row.get("state"))
    return value


def _immutable_json(path: Path, value: Mapping[str, Any]) -> Path:
    text = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, indent=2,
        allow_nan=False,
    ) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise SignalEvidenceError(
                f"refusing to overwrite non-identical signal evidence {path}")
        return path
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    return path


def materialize_signal_evidence(
    data_root: Path,
    *,
    probe_point_id: str,
    channel: str,
    environment_instance: str,
    collection_run_id: str,
    observer_version: str,
    endpoint_or_query_descriptor: str,
    provider_metadata: Mapping[str, Any],
    code_version: str,
    collection_timestamp: str,
    observations: Mapping[str, Mapping[str, Any]],
    collected_live: bool,
    is_fixture: bool,
    pii_review_status: str,
    redaction_applied: bool,
) -> dict[str, Any]:
    """Freeze one three-phase observer trace and return its point evidence patch.

    The endpoint/query descriptor is hashed but never persisted verbatim, so
    URLs/queries can be identified without leaking credentials. Raw payloads
    with credential-like field names are rejected before any file is written.
    """
    required_text = {
        "probe_point_id": probe_point_id,
        "channel": channel,
        "environment_instance": environment_instance,
        "collection_run_id": collection_run_id,
        "observer_version": observer_version,
        "endpoint_or_query_descriptor": endpoint_or_query_descriptor,
        "code_version": code_version,
        "collection_timestamp": collection_timestamp,
    }
    missing = sorted(name for name, value in required_text.items()
                     if not str(value or "").strip())
    if missing:
        raise SignalEvidenceError(
            "missing signal evidence fields: " + ",".join(missing))
    channel = channel.strip().lower()
    if channel not in SIGNAL_CHANNELS:
        raise SignalEvidenceError(f"unsupported signal channel {channel!r}")
    if not isinstance(collected_live, bool) or not isinstance(is_fixture, bool):
        raise SignalEvidenceError("collected_live/is_fixture must be booleans")
    if collected_live and is_fixture:
        raise SignalEvidenceError("live signal evidence cannot be a fixture")
    if pii_review_status not in _PII_REVIEW_STATUSES:
        raise SignalEvidenceError("invalid pii_review_status")
    if not isinstance(redaction_applied, bool):
        raise SignalEvidenceError("redaction_applied must be boolean")
    if pii_review_status == "NOT_PRESENT" and redaction_applied:
        raise SignalEvidenceError(
            "NOT_PRESENT cannot claim that PII redaction was applied")
    if pii_review_status == "REDACTED_AND_REVIEWED" and not redaction_applied:
        raise SignalEvidenceError(
            "REDACTED_AND_REVIEWED requires redaction_applied=true")
    if set(observations) != set(SIGNAL_PHASES):
        raise SignalEvidenceError("observations must contain exactly pre/post/final")
    provider = _validate_provider_metadata(provider_metadata)

    normalized: dict[str, Any] = {}
    snapshot_values: dict[str, dict[str, Any]] = {}
    for phase in SIGNAL_PHASES:
        item = observations[phase]
        if not isinstance(item, Mapping):
            raise SignalEvidenceError(f"{phase} observation must be an object")
        if set(item) != {"observed_at", "raw_payload", "normalized_state",
                         "read_only_attestation"}:
            raise SignalEvidenceError(
                f"{phase} observation keys must be observed_at/raw_payload/"
                "normalized_state/read_only_attestation")
        if not str(item["observed_at"] or "").strip():
            raise SignalEvidenceError(f"{phase} observed_at is required")
        secret_paths = _secret_paths(item["raw_payload"])
        if secret_paths:
            raise SignalEvidenceError(
                f"{phase} raw payload contains credential-like keys: "
                + ",".join(secret_paths))
        attestation = _validate_attestation(
            item["read_only_attestation"], provider, item["raw_payload"])
        canonical_sha256(item["raw_payload"])
        canonical_sha256(item["normalized_state"])
        normalized[phase] = item["normalized_state"]
        snapshot_values[phase] = {
            "schema_version": SIGNAL_SNAPSHOT_SCHEMA_VERSION,
            "probe_point_id": probe_point_id,
            "channel": channel,
            "phase": phase,
            "environment_instance": environment_instance,
            "observed_at": str(item["observed_at"]),
            "raw_payload": item["raw_payload"],
            "normalized_state": item["normalized_state"],
            "read_only_attestation": attestation,
        }

    identity = {
        "probe_point_id": probe_point_id,
        "channel": channel,
        "environment_instance": environment_instance,
        "collection_run_id": collection_run_id,
        "observer_version": observer_version,
        "endpoint_or_query_sha256":
            hashlib.sha256(endpoint_or_query_descriptor.encode("utf-8")).hexdigest(),
        "normalized_state_sha256": {
            phase: canonical_sha256(normalized[phase]) for phase in SIGNAL_PHASES},
        "provider_sha256": canonical_sha256(provider),
    }
    evidence_id = "signal-" + canonical_sha256(identity)[:20]
    base = Path(data_root) / "evidence" / "signals" / probe_point_id / evidence_id

    snapshot_refs: list[dict[str, Any]] = []
    for phase in SIGNAL_PHASES:
        path = _immutable_json(base / f"{phase}.json", snapshot_values[phase])
        snapshot_refs.append({
            "schema_version": SIGNAL_SNAPSHOT_REF_SCHEMA_VERSION,
            "phase": phase,
            "path": str(path.relative_to(Path(data_root))),
            "sha256": file_sha256(path),
            "normalized_state_sha256": canonical_sha256(normalized[phase]),
        })

    asset = {
        "schema_version": SIGNAL_EVIDENCE_SCHEMA_VERSION,
        "evidence_id": evidence_id,
        "probe_point_id": probe_point_id,
        "channel": channel,
        "environment_instance": environment_instance,
        "collection_timestamp": collection_timestamp,
        "code_version": code_version,
        "observer_version": observer_version,
        "collection_run_id": collection_run_id,
        "endpoint_or_query_sha256": identity["endpoint_or_query_sha256"],
        "provider": provider,
        "collected_live": collected_live,
        "is_fixture": is_fixture,
        "read_only_observer": True,
        "credential_value_stored": False,
        "pii_review_status": pii_review_status,
        "redaction_applied": redaction_applied,
        "payload_semantics": "minimized_redacted_projection",
        "redaction_key_persisted": False,
        "snapshots": snapshot_refs,
    }
    asset_path = _immutable_json(base / "evidence.json", asset)
    reference = {
        "schema_version": SIGNAL_EVIDENCE_REF_SCHEMA_VERSION,
        "probe_point_id": probe_point_id,
        "channel": channel,
        "path": str(asset_path.relative_to(Path(data_root))),
        "sha256": file_sha256(asset_path),
    }
    return {
        "asset": asset,
        "reference": reference,
        "point_evidence_patch": {
            "signal_channels": [channel],
            "api_db_signal_summaries": {channel: normalized},
            "signal_evidence_assets": [reference],
        },
    }
