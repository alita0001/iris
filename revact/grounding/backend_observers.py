"""Read-only database observers for live point probes.

The browser-facing signal in :mod:`revact.grounding.signals` is useful but it
cannot reveal state that the rendered page omits.  This module provides an
independent, deliberately narrow upper-bound channel for the two local
WebArena services that IRIS currently uses:

* Magento: a fixed ``SELECT`` projection executed in a read-only MariaDB
  transaction inside the reviewed ``shopping`` container;
* Postmill: a fixed ``SELECT`` projection executed with PostgreSQL
  ``default_transaction_read_only=on`` inside the reviewed ``forum``
  container.

No caller can supply SQL.  Query text, expected columns, identifier fields and
container defaults are frozen in :data:`PROVIDERS`.  Database credentials are
loaded *inside* the Magento container and never cross the process boundary.
Rows are minimized immediately: internal identifiers become HMAC tokens under
an ephemeral per-probe key, while only mechanism-relevant state fields remain
in clear text.  The key is never persisted, so tokens can be compared across
pre/post/final phases but cannot be joined to another run.

An observer only reads.  It does not make a historical UI-only point eligible
for API/DB evidence: formal admission still requires three observations taken
around the same newly executed point transition.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence


BACKEND_PROVIDER_SCHEMA_VERSION = "iris.backend_signal_provider.v1"
BACKEND_ATTESTATION_SCHEMA_VERSION = "iris.backend_read_only_attestation.v1"
BACKEND_PREFLIGHT_SCHEMA_VERSION = "iris.signal_observer_preflight.v1"
OBSERVER_VERSION = "backend-db-observer.v1"

_CONTAINER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_MAX_ROWS = 10_000
_FORBIDDEN_SQL = re.compile(
    r"\b(?:insert|update|delete|replace|alter|drop|truncate|create|grant|"
    r"revoke|call|load|outfile|dumpfile|lock|unlock|set|do|handler)\b",
    re.IGNORECASE,
)


class BackendObserverError(RuntimeError):
    """A provider was unavailable, unsafe, or returned an invalid projection."""


@dataclass(frozen=True)
class QueryProjection:
    query_id: str
    sql: str
    columns: tuple[str, ...]
    identifier_fields: tuple[str, ...]
    state_fields: tuple[str, ...]

    def validate(self) -> None:
        errors: list[str] = []
        sql = self.sql.strip()
        if not self.query_id.strip():
            errors.append("query_id is required")
        if not re.match(r"^SELECT\b", sql, re.IGNORECASE):
            errors.append("observer query must start with SELECT")
        if ";" in sql or "--" in sql or "/*" in sql:
            errors.append("observer query cannot contain separators/comments")
        if _FORBIDDEN_SQL.search(sql):
            errors.append("observer query contains a forbidden SQL token")
        if not self.columns or len(set(self.columns)) != len(self.columns):
            errors.append("columns must be non-empty and unique")
        if set(self.identifier_fields) | set(self.state_fields) != set(self.columns):
            errors.append("identifier_fields + state_fields must partition columns")
        if set(self.identifier_fields) & set(self.state_fields):
            errors.append("identifier_fields and state_fields must be disjoint")
        if not self.identifier_fields:
            errors.append("at least one identifier field is required")
        if errors:
            raise BackendObserverError("; ".join(errors))


@dataclass(frozen=True)
class BackendProviderSpec:
    schema_version: str
    provider_id: str
    site: str
    channel: str
    database_system: str
    transport: str
    default_container: str
    database_name: str
    projections: Mapping[str, QueryProjection]
    read_only_enforcement: str
    projection_version: str = "minimized-hmac.v1"

    def validate(self) -> None:
        errors: list[str] = []
        if self.schema_version != BACKEND_PROVIDER_SCHEMA_VERSION:
            errors.append("bad provider schema_version")
        if not all(str(value).strip() for value in (
                self.provider_id, self.site, self.channel, self.database_system,
                self.transport, self.default_container, self.database_name,
                self.read_only_enforcement, self.projection_version)):
            errors.append("provider fields cannot be empty")
        if self.channel != "db":
            errors.append("local backend providers must use channel=db")
        if self.database_system not in {"mariadb", "postgresql"}:
            errors.append("unsupported database_system")
        if self.transport != "docker_exec":
            errors.append("unsupported observer transport")
        if not _CONTAINER_RE.fullmatch(self.default_container):
            errors.append("unsafe default container name")
        if not self.projections:
            errors.append("provider must register action projections")
        for action_type, projection in self.projections.items():
            if not str(action_type).strip():
                errors.append("empty action_type in provider projections")
            try:
                projection.validate()
            except BackendObserverError as exc:
                errors.append(f"{action_type}: {exc}")
        if errors:
            raise BackendObserverError("; ".join(errors))


_MAGENTO_PROJECTIONS: dict[str, QueryProjection] = {
    "add_to_cart": QueryProjection(
        "magento.active_quote_items.v1",
        "SELECT qi.item_id, qi.quote_id, qi.product_id, CAST(qi.qty AS CHAR) AS qty "
        "FROM quote_item qi JOIN quote q ON q.entity_id=qi.quote_id "
        "WHERE q.is_active=1 AND qi.parent_item_id IS NULL "
        "ORDER BY qi.quote_id, qi.item_id LIMIT 10001",
        ("item_id", "quote_id", "product_id", "qty"),
        ("item_id", "quote_id", "product_id"), ("qty",)),
    "wishlist_add": QueryProjection(
        "magento.wishlist_items.v1",
        "SELECT wi.wishlist_item_id, wi.wishlist_id, w.customer_id, wi.product_id, "
        "CAST(wi.qty AS CHAR) AS qty FROM wishlist_item wi "
        "JOIN wishlist w ON w.wishlist_id=wi.wishlist_id "
        "ORDER BY wi.wishlist_id, wi.wishlist_item_id LIMIT 10001",
        ("wishlist_item_id", "wishlist_id", "customer_id", "product_id", "qty"),
        ("wishlist_item_id", "wishlist_id", "customer_id", "product_id"), ("qty",)),
    "compare_add": QueryProjection(
        "magento.compare_items.v1",
        "SELECT catalog_compare_item_id, visitor_id, customer_id, product_id, "
        "store_id, list_id FROM catalog_compare_item "
        "ORDER BY catalog_compare_item_id LIMIT 10001",
        ("catalog_compare_item_id", "visitor_id", "customer_id", "product_id",
         "store_id", "list_id"),
        ("catalog_compare_item_id", "visitor_id", "customer_id", "product_id",
         "store_id", "list_id"), ()),
    "newsletter_subscribe": QueryProjection(
        "magento.newsletter_state.v1",
        "SELECT subscriber_id, store_id, customer_id, subscriber_status "
        "FROM newsletter_subscriber ORDER BY subscriber_id LIMIT 10001",
        ("subscriber_id", "store_id", "customer_id", "subscriber_status"),
        ("subscriber_id", "store_id", "customer_id"), ("subscriber_status",)),
    "address_add": QueryProjection(
        "magento.customer_address_rows.v1",
        "SELECT entity_id, parent_id FROM customer_address_entity "
        "ORDER BY entity_id LIMIT 10001",
        ("entity_id", "parent_id"), ("entity_id", "parent_id"), ()),
    "place_order": QueryProjection(
        "magento.sales_order_state.v1",
        "SELECT entity_id, customer_id, quote_id, increment_id, state, status "
        "FROM sales_order ORDER BY entity_id LIMIT 10001",
        ("entity_id", "customer_id", "quote_id", "increment_id", "state", "status"),
        ("entity_id", "customer_id", "quote_id", "increment_id"),
        ("state", "status")),
}

_POSTMILL_PROJECTIONS: dict[str, QueryProjection] = {
    "reddit_vote": QueryProjection(
        "postmill.submission_votes.v1",
        "SELECT id, user_id, submission_id, upvote FROM submission_votes "
        "ORDER BY id LIMIT 10001",
        ("id", "user_id", "submission_id", "upvote"),
        ("id", "user_id", "submission_id"), ("upvote",)),
    "reddit_subscribe": QueryProjection(
        "postmill.forum_subscriptions.v1",
        "SELECT fs.id, fs.user_id, fs.forum_id, f.normalized_name AS forum "
        "FROM forum_subscriptions fs JOIN forums f ON f.id=fs.forum_id "
        "ORDER BY fs.id LIMIT 10001",
        ("id", "user_id", "forum_id", "forum"),
        ("id", "user_id", "forum_id"), ("forum",)),
}

PROVIDERS: dict[str, BackendProviderSpec] = {
    "shopping": BackendProviderSpec(
        BACKEND_PROVIDER_SCHEMA_VERSION, "webarena.magento.local-db.v1",
        "shopping", "db", "mariadb", "docker_exec", "shopping", "magento",
        _MAGENTO_PROJECTIONS,
        "SET SESSION TRANSACTION READ ONLY + START TRANSACTION"),
    "reddit": BackendProviderSpec(
        BACKEND_PROVIDER_SCHEMA_VERSION, "webarena.postmill.local-db.v1",
        "reddit", "db", "postgresql", "docker_exec", "forum", "postmill",
        _POSTMILL_PROJECTIONS,
        "PGOPTIONS default_transaction_read_only=on"),
}
for _provider in PROVIDERS.values():
    _provider.validate()


CommandRunner = Callable[[Sequence[str], Mapping[str, str]], subprocess.CompletedProcess[str]]


def _default_runner(command: Sequence[str], environment: Mapping[str, str]
                    ) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command), env=dict(environment), text=True, capture_output=True,
        check=False, timeout=30)


def _docker_environment() -> dict[str, str]:
    # The workspace commonly routes HTTP through a proxy.  Docker's remote API
    # is a local research service and the Go client otherwise sends it to that
    # proxy, yielding a misleading 502/Bad Gateway.
    environment = dict(os.environ)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        environment.pop(name, None)
    return environment


def _output_error(result: subprocess.CompletedProcess[str], context: str
                  ) -> BackendObserverError:
    # Never put stderr in an exception/report: a client may include DSNs or
    # credentials in its diagnostics.  A digest is enough to correlate local
    # failures without persisting the value.
    stderr_hash = hashlib.sha256((result.stderr or "").encode()).hexdigest()
    return BackendObserverError(
        f"{context} failed rc={result.returncode} stderr_sha256={stderr_hash}")


def _container_fingerprint(container: str, runner: CommandRunner,
                           environment: Mapping[str, str]) -> dict[str, str]:
    result = runner([
        "docker", "inspect", "--format",
        "{{json .Id}}|{{json .Config.Image}}|{{json .Image}}", container,
    ], environment)
    if result.returncode:
        raise _output_error(result, "docker inspect")
    parts = (result.stdout or "").strip().split("|")
    if len(parts) != 3:
        raise BackendObserverError("docker inspect returned an invalid fingerprint")
    try:
        container_id, image_ref, image_id = (json.loads(item) for item in parts)
    except json.JSONDecodeError as exc:
        raise BackendObserverError("docker inspect fingerprint is not JSON") from exc
    if not all(isinstance(item, str) and item for item in
               (container_id, image_ref, image_id)):
        raise BackendObserverError("docker inspect fingerprint is incomplete")
    return {
        "container_id_sha256": hashlib.sha256(container_id.encode()).hexdigest(),
        "image_ref": image_ref,
        "image_id_sha256": hashlib.sha256(image_id.encode()).hexdigest(),
    }


_MYSQL_READER = r"""
$query = base64_decode($argv[1], true);
if ($query === false || !preg_match('/^\s*SELECT\b/i', $query) ||
    strpos($query, ';') !== false || preg_match('/\b(?:insert|update|delete|replace|alter|drop|truncate|create|grant|revoke|call|load|outfile|dumpfile|lock|unlock|set|do|handler)\b/i', $query)) {
    fwrite(STDERR, "unsafe query\n"); exit(64);
}
$config = include '/var/www/magento2/app/etc/env.php';
$db = $config['db']['connection']['default'];
$pdo = new PDO('mysql:host='.$db['host'].';dbname='.$db['dbname'],
               $db['username'], $db['password'],
               array(PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION));
$pdo->exec('SET SESSION TRANSACTION READ ONLY');
$pdo->beginTransaction();
$readOnly = (int)$pdo->query('SELECT @@session.tx_read_only')->fetchColumn();
$rows = $pdo->query($query)->fetchAll(PDO::FETCH_ASSOC);
$pdo->rollBack();
echo json_encode(array('transaction_read_only' => $readOnly === 1,
                       'rows' => $rows), JSON_THROW_ON_ERROR);
""".strip()


def _mysql_read(container: str, projection: QueryProjection,
                runner: CommandRunner, environment: Mapping[str, str]) -> dict[str, Any]:
    encoded = base64.b64encode(projection.sql.encode()).decode()
    result = runner(
        ["docker", "exec", container, "php", "-r", _MYSQL_READER, encoded],
        environment)
    if result.returncode:
        raise _output_error(result, "MariaDB read-only observer")
    return _decode_provider_output(result.stdout, projection)


def _postgres_read(container: str, database: str, projection: QueryProjection,
                   runner: CommandRunner,
                   environment: Mapping[str, str]) -> dict[str, Any]:
    wrapper = (
        "SELECT json_build_object(" 
        "'transaction_read_only', current_setting('transaction_read_only')='on',"
        "'rows', COALESCE(json_agg(row_to_json(q)), '[]'::json)) "
        f"FROM ({projection.sql}) q")
    result = runner([
        "docker", "exec", "-e", "PGOPTIONS=-c default_transaction_read_only=on",
        "--user", "postgres", container, "psql", "-X", "-A", "-t",
        "-v", "ON_ERROR_STOP=1", "-d", database, "-c", wrapper,
    ], environment)
    if result.returncode:
        raise _output_error(result, "PostgreSQL read-only observer")
    return _decode_provider_output(result.stdout, projection)


def _decode_provider_output(stdout: str, projection: QueryProjection
                            ) -> dict[str, Any]:
    try:
        payload = json.loads((stdout or "").strip())
    except json.JSONDecodeError as exc:
        digest = hashlib.sha256((stdout or "").encode()).hexdigest()
        raise BackendObserverError(
            f"observer output is not JSON stdout_sha256={digest}") from exc
    if not isinstance(payload, dict) or set(payload) != {
            "transaction_read_only", "rows"}:
        raise BackendObserverError("observer output fields are not exact")
    if payload["transaction_read_only"] is not True:
        raise BackendObserverError("database did not attest a read-only transaction")
    rows = payload["rows"]
    if not isinstance(rows, list):
        raise BackendObserverError("observer rows must be a list")
    if len(rows) > _MAX_ROWS:
        raise BackendObserverError(
            f"observer result exceeded {_MAX_ROWS} rows; projection is truncated")
    expected = set(projection.columns)
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != expected:
            actual = sorted(row) if isinstance(row, dict) else type(row).__name__
            raise BackendObserverError(
                f"observer row {index} fields differ expected={sorted(expected)} "
                f"actual={actual}")
        # Ensure the projected result remains canonical JSON before any point
        # can be mutated.  NaN/Infinity and driver-specific objects fail here.
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return payload


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


class BackendSignalObserver:
    """One per-point observer; its redaction scope spans exactly three phases."""

    def __init__(self, provider: BackendProviderSpec, action_type: str,
                 environment_instance: str, *, container: str,
                 runner: CommandRunner | None = None,
                 redaction_key: bytes | None = None,
                 is_fixture: bool = False) -> None:
        provider.validate()
        if action_type not in provider.projections:
            raise BackendObserverError(
                f"provider {provider.provider_id} does not support {action_type!r}")
        if not str(environment_instance).strip():
            raise BackendObserverError("environment_instance is required")
        if not _CONTAINER_RE.fullmatch(container):
            raise BackendObserverError("unsafe container name")
        if runner is not None and not is_fixture:
            raise BackendObserverError(
                "an injected runner is fixture-only and cannot claim live evidence")
        if redaction_key is not None and not is_fixture:
            raise BackendObserverError(
                "an injected redaction key is fixture-only and cannot claim live evidence")
        self.provider = provider
        self.projection = provider.projections[action_type]
        self.action_type = action_type
        self.environment_instance = str(environment_instance)
        self.container = container
        self.runner = runner or _default_runner
        self.is_fixture = bool(is_fixture)
        self.collected_live = not self.is_fixture
        self._redaction_key = redaction_key or secrets.token_bytes(32)
        self._environment = _docker_environment()
        self._source = _container_fingerprint(
            container, self.runner, self._environment)

    @property
    def endpoint_or_query_descriptor(self) -> str:
        # Contains no host credentials or SQL, but is still hashed by the
        # evidence materializer to avoid turning it into a connection hint.
        return (
            f"docker-exec://{self.container}/{self.provider.database_system}/"
            f"{self.provider.database_name}/{self.projection.query_id}")

    @property
    def provider_metadata(self) -> dict[str, Any]:
        source_sha = hashlib.sha256(_canonical(self._source).encode()).hexdigest()
        return {
            "schema_version": BACKEND_PROVIDER_SCHEMA_VERSION,
            "provider_id": self.provider.provider_id,
            "database_system": self.provider.database_system,
            "transport": self.provider.transport,
            "query_id": self.projection.query_id,
            "query_sha256": hashlib.sha256(
                self.projection.sql.encode()).hexdigest(),
            "source_instance_sha256": source_sha,
            "read_only_enforcement": self.provider.read_only_enforcement,
            "projection_version": self.provider.projection_version,
            "redaction_strategy": "ephemeral-hmac-sha256",
            "redaction_scope_sha256": hashlib.sha256(
                self._redaction_key).hexdigest(),
            "redaction_key_persisted": False,
            "container_image_ref": self._source["image_ref"],
            "container_id_sha256": self._source["container_id_sha256"],
            "container_image_id_sha256": self._source["image_id_sha256"],
        }

    def _query(self) -> dict[str, Any]:
        if self.provider.database_system == "mariadb":
            return _mysql_read(
                self.container, self.projection, self.runner, self._environment)
        if self.provider.database_system == "postgresql":
            return _postgres_read(
                self.container, self.provider.database_name, self.projection,
                self.runner, self._environment)
        raise BackendObserverError("unsupported database system")

    def _token(self, identities: Mapping[str, Any]) -> str:
        return hmac.new(
            self._redaction_key, _canonical(dict(identities)).encode(),
            hashlib.sha256).hexdigest()

    def capture(self, phase: str) -> dict[str, Any]:
        if phase not in {"pre", "post", "final"}:
            raise BackendObserverError(f"invalid point signal phase {phase!r}")
        payload = self._query()
        projected: list[dict[str, Any]] = []
        for row in payload["rows"]:
            identities = {field: row[field]
                          for field in self.projection.identifier_fields}
            state = {field: row[field] for field in self.projection.state_fields}
            projected.append({
                "row_token": self._token(identities),
                "state": state,
            })
        projected.sort(key=lambda item: (item["row_token"], _canonical(item["state"])))
        raw_payload = {
            "payload_semantics": "minimized_redacted_projection",
            "query_id": self.projection.query_id,
            "transaction_read_only": True,
            "row_count": len(projected),
            "rows": projected,
        }
        normalized_state = {
            "query_id": self.projection.query_id,
            "row_count": len(projected),
            "rows": projected,
        }
        attestation = {
            "schema_version": BACKEND_ATTESTATION_SCHEMA_VERSION,
            "provider_id": self.provider.provider_id,
            "query_id": self.projection.query_id,
            "transaction_read_only": True,
            "source_instance_sha256": self.provider_metadata[
                "source_instance_sha256"],
            "result_row_count": len(projected),
        }
        return {
            "observed_at": datetime.now(timezone.utc).isoformat(
                timespec="microseconds"),
            "raw_payload": raw_payload,
            "normalized_state": normalized_state,
            "read_only_attestation": attestation,
        }

    def preflight(self) -> dict[str, Any]:
        """Perform one real read without pretending it is point evidence."""
        observation = self.capture("pre")
        return {
            "schema_version": BACKEND_PREFLIGHT_SCHEMA_VERSION,
            "observer_version": OBSERVER_VERSION,
            "provider": self.provider_metadata,
            "environment_instance": self.environment_instance,
            "action_type": self.action_type,
            "channel": self.provider.channel,
            "collected_live": self.collected_live,
            "is_fixture": self.is_fixture,
            "read_only_observer": True,
            "counts_as_point_signal_evidence": False,
            "reason_not_point_evidence": (
                "single current-state read; not synchronized to one "
                "pre/action/post/undo/final transition"),
            "observed_at": observation["observed_at"],
            "transaction_read_only": observation[
                "read_only_attestation"]["transaction_read_only"],
            "result_row_count": observation[
                "read_only_attestation"]["result_row_count"],
            "raw_payload_sha256": hashlib.sha256(
                _canonical(observation["raw_payload"]).encode()).hexdigest(),
            "normalized_state_sha256": hashlib.sha256(
                _canonical(observation["normalized_state"]).encode()).hexdigest(),
            "credential_value_stored": False,
            "pii_review_status": "REDACTED_AND_REVIEWED",
            "redaction_applied": True,
        }


def build_live_backend_observer(site: str, action_type: str,
                                environment_instance: str, *,
                                container: str | None = None
                                ) -> BackendSignalObserver:
    """Resolve one immutable registry entry; caller cannot provide SQL."""
    provider = PROVIDERS.get(str(site))
    if provider is None:
        raise BackendObserverError(f"no backend observer provider for site {site!r}")
    selected = container or os.environ.get(
        f"REVACT_SIGNAL_{site.upper()}_CONTAINER", provider.default_container)
    return BackendSignalObserver(
        provider, action_type, environment_instance, container=selected)


def build_fixture_backend_observer(site: str, action_type: str,
                                   environment_instance: str, *,
                                   runner: CommandRunner,
                                   redaction_key: bytes = b"fixture-redaction-key",
                                   container: str | None = None
                                   ) -> BackendSignalObserver:
    """Test-only constructor; resulting evidence is permanently fixture-marked."""
    provider = PROVIDERS.get(str(site))
    if provider is None:
        raise BackendObserverError(f"no backend observer provider for site {site!r}")
    return BackendSignalObserver(
        provider, action_type, environment_instance,
        container=container or provider.default_container, runner=runner,
        redaction_key=redaction_key, is_fixture=True)

