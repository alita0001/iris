"""Strict backend observer, redaction and immutable evidence regression tests."""
from __future__ import annotations

import json
import subprocess
from collections import deque

import pytest

from revact.grounding.backend_observers import (
    BACKEND_PREFLIGHT_SCHEMA_VERSION, BACKEND_PROVIDER_SCHEMA_VERSION,
    BackendObserverError, BackendSignalObserver, PROVIDERS, QueryProjection,
    build_fixture_backend_observer)
from revact.grounding.signal_evidence import (
    SIGNAL_EVIDENCE_SCHEMA_VERSION, SIGNAL_SNAPSHOT_SCHEMA_VERSION,
    SignalEvidenceError, materialize_signal_evidence)


class _Runner:
    def __init__(self, rows_by_read):
        self.rows_by_read = deque(rows_by_read)
        self.commands = []

    def __call__(self, command, environment):
        self.commands.append(list(command))
        assert not any(name in environment for name in (
            "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"))
        if command[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(
                command, 0,
                stdout='"container-id"|"fixture/image:1"|"sha256:image-id"\n',
                stderr="")
        rows = self.rows_by_read.popleft()
        return subprocess.CompletedProcess(
            command, 0,
            stdout=json.dumps({
                "transaction_read_only": True,
                "rows": rows,
            }),
            stderr="")


def _cart_rows(item_id: str | None = None):
    if item_id is None:
        return []
    return [{
        "item_id": item_id,
        "quote_id": "7",
        "product_id": "99",
        "qty": "1.0000",
    }]


def _fixture_observer(runner: _Runner):
    return build_fixture_backend_observer(
        "shopping", "add_to_cart", "http://shop", runner=runner,
        redaction_key=b"stable-fixture-key")


def test_provider_registry_is_fixed_select_only():
    for provider in PROVIDERS.values():
        provider.validate()
        for projection in provider.projections.values():
            projection.validate()
            assert projection.sql.lstrip().upper().startswith("SELECT ")
            assert ";" not in projection.sql

    unsafe = QueryProjection(
        "unsafe", "SELECT id FROM rows; DELETE FROM rows", ("id",),
        ("id",), ())
    with pytest.raises(BackendObserverError, match="separators"):
        unsafe.validate()


def test_injected_executor_can_never_claim_live_evidence():
    runner = _Runner([])
    with pytest.raises(BackendObserverError, match="fixture-only"):
        BackendSignalObserver(
            PROVIDERS["shopping"], "add_to_cart", "http://shop",
            container="shopping", runner=runner, is_fixture=False)


def test_fixture_observer_hmac_minimizes_identifiers_across_three_phases():
    runner = _Runner([_cart_rows(), _cart_rows("42"), _cart_rows()])
    observer = _fixture_observer(runner)
    observations = {
        phase: observer.capture(phase) for phase in ("pre", "post", "final")}

    assert observer.collected_live is False
    assert observer.is_fixture is True
    assert observations["pre"]["normalized_state"] == \
        observations["final"]["normalized_state"]
    assert observations["pre"]["normalized_state"] != \
        observations["post"]["normalized_state"]
    post = observations["post"]["raw_payload"]
    assert set(post["rows"][0]) == {"row_token", "state"}
    assert len(post["rows"][0]["row_token"]) == 64
    assert post["rows"][0]["state"] == {"qty": "1.0000"}
    serialized = json.dumps(observations, sort_keys=True)
    assert "item_id" not in serialized
    assert "quote_id" not in serialized
    assert "product_id" not in serialized
    assert all(
        observation["read_only_attestation"]["transaction_read_only"]
        for observation in observations.values())


def test_materialized_fixture_is_hash_pinned_but_cannot_claim_live(tmp_path):
    runner = _Runner([_cart_rows(), _cart_rows("42"), _cart_rows()])
    observer = _fixture_observer(runner)
    observations = {
        phase: observer.capture(phase) for phase in ("pre", "post", "final")}
    bundle = materialize_signal_evidence(
        tmp_path, probe_point_id="point-1", channel="db",
        environment_instance="http://shop", collection_run_id="run-1",
        observer_version="fixture-observer.v1",
        endpoint_or_query_descriptor=observer.endpoint_or_query_descriptor,
        provider_metadata=observer.provider_metadata,
        code_version="deadbeef", collection_timestamp="2026-07-16T00:00:00Z",
        observations=observations, collected_live=False, is_fixture=True,
        pii_review_status="REDACTED_AND_REVIEWED", redaction_applied=True)

    asset = bundle["asset"]
    assert asset["schema_version"] == SIGNAL_EVIDENCE_SCHEMA_VERSION
    assert asset["collected_live"] is False
    assert asset["is_fixture"] is True
    assert asset["provider"]["schema_version"] == \
        BACKEND_PROVIDER_SCHEMA_VERSION
    assert asset["redaction_key_persisted"] is False
    assert bundle["point_evidence_patch"]["signal_channels"] == ["db"]
    for snapshot_ref in asset["snapshots"]:
        snapshot = json.loads((tmp_path / snapshot_ref["path"]).read_text())
        assert snapshot["schema_version"] == SIGNAL_SNAPSHOT_SCHEMA_VERSION
        assert snapshot["read_only_attestation"][
            "transaction_read_only"] is True


def test_materializer_rejects_secret_values_and_attestation_mismatch(tmp_path):
    runner = _Runner([_cart_rows(), _cart_rows("42"), _cart_rows()])
    observer = _fixture_observer(runner)
    observations = {
        phase: observer.capture(phase) for phase in ("pre", "post", "final")}
    observations["post"]["raw_payload"]["rows"][0]["state"]["note"] = \
        "Bearer definitely-a-secret-token"
    with pytest.raises(SignalEvidenceError, match="credential-like"):
        materialize_signal_evidence(
            tmp_path, probe_point_id="point-1", channel="db",
            environment_instance="http://shop", collection_run_id="run-1",
            observer_version="fixture-observer.v1",
            endpoint_or_query_descriptor=observer.endpoint_or_query_descriptor,
            provider_metadata=observer.provider_metadata,
            code_version="deadbeef",
            collection_timestamp="2026-07-16T00:00:00Z",
            observations=observations, collected_live=False, is_fixture=True,
            pii_review_status="REDACTED_AND_REVIEWED", redaction_applied=True)

    runner = _Runner([_cart_rows(), _cart_rows("42"), _cart_rows()])
    observer = _fixture_observer(runner)
    observations = {
        phase: observer.capture(phase) for phase in ("pre", "post", "final")}
    observations["post"]["read_only_attestation"][
        "transaction_read_only"] = False
    with pytest.raises(SignalEvidenceError, match="not read-only"):
        materialize_signal_evidence(
            tmp_path, probe_point_id="point-2", channel="db",
            environment_instance="http://shop", collection_run_id="run-2",
            observer_version="fixture-observer.v1",
            endpoint_or_query_descriptor=observer.endpoint_or_query_descriptor,
            provider_metadata=observer.provider_metadata,
            code_version="deadbeef",
            collection_timestamp="2026-07-16T00:00:00Z",
            observations=observations, collected_live=False, is_fixture=True,
            pii_review_status="REDACTED_AND_REVIEWED", redaction_applied=True)


def test_preflight_explicitly_cannot_count_as_point_evidence():
    runner = _Runner([_cart_rows()])
    report = _fixture_observer(runner).preflight()
    assert report["schema_version"] == BACKEND_PREFLIGHT_SCHEMA_VERSION
    assert report["transaction_read_only"] is True
    assert report["counts_as_point_signal_evidence"] is False
    assert report["is_fixture"] is True
    assert "normalized_state" not in report


def test_provider_failure_never_persists_stderr_value():
    secret = "password=must-never-escape"

    def runner(command, _environment):
        if command[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(
                command, 0,
                stdout='"container-id"|"fixture/image:1"|"sha256:image-id"\n',
                stderr="")
        return subprocess.CompletedProcess(command, 1, stdout="", stderr=secret)

    observer = build_fixture_backend_observer(
        "shopping", "add_to_cart", "http://shop", runner=runner)
    with pytest.raises(BackendObserverError) as captured:
        observer.capture("pre")
    assert secret not in str(captured.value)
    assert "stderr_sha256=" in str(captured.value)

