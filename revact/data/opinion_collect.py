"""Provider-agnostic collection of *opinions* from label-blind pre-state inputs.

This module is intentionally separate from grounding.  Its model-facing input
contains only ``goal``, the exact pre-action observation, and ``action``.  A
caller cannot pass effect/recovery truth, post-state signals, or undo traces
through the strict input schema.  Results use :mod:`revact.data.opinions` and
therefore remain an opinion-only baseline, never formal ground truth.

The built-in HTTP client speaks the OpenAI-compatible chat-completions wire
format but has no provider default.  The base URL and model are explicit, and
the credential value is read only from a named environment variable at call
time.  Neither the value nor its environment-variable name is persisted.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .. import config, prompts
from ..eval.truth import EvaluationTruthRecord
from ..grounding.schema import GroundingPoint
from .opinions import (
    NOT_RISKY,
    OPINION_ARTIFACT_ROLE,
    OPINION_LABEL_SCHEMA_VERSION,
    PERCEIVED_EFFECTS,
    PERCEIVED_RECOVERABILITIES,
    RATER_LLM,
    RISKY,
    OpinionLabelError,
    OpinionLabelRecord,
    _atomic_write_pair,
    make_opinion_label_id,
)


OPINION_INPUT_SCHEMA_VERSION = "iris.opinion_rating_input.v2"
OPINION_INPUT_MANIFEST_SCHEMA_VERSION = "iris.opinion_rating_input_manifest.v2"
OPINION_INSTRUMENT_ID = "iris-preaction-opinion-rater"
OPINION_INSTRUMENT_VERSION = "v1"
_RISK_VALUES = (RISKY, NOT_RISKY, "UNKNOWN")
_OUTPUT_KEYS = {"effect", "recovery", "normative_risk", "confidence", "rationale"}
_INPUT_KEYS = {
    "schema_version",
    "probe_point_id",
    "state_id",
    "evaluation_case_id",
    "variant",
    "goal",
    "goal_sha256",
    "pre_observation",
    "action",
    "opinion_input_sha256",
}
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CREDENTIAL_VALUE = re.compile(
    r"(?:\bBearer\s+\S{8,}|\bsk-(?:or-v1-)?[A-Za-z0-9_-]{12,}|"
    r"(?:api[_-]?key|password|secret|token)\s*[:=]\s*\S{6,})",
    re.IGNORECASE,
)


class OpinionCollectionError(OpinionLabelError):
    """A rating input, provider response, or collection setting is unsafe."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    blob = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    return _sha256_text(blob)


def _canonical_value_sha256(value: Any) -> str:
    blob = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    return _sha256_text(blob)


def opinion_evidence_sha256(goal: str, pre_observation: str, action: str) -> str:
    return _canonical_sha256({
        "goal": goal,
        "pre_observation": pre_observation,
        "action": action,
    })


@dataclass(frozen=True)
class OpinionRatingInput:
    """One label-blind point input; IDs are join metadata, not model content."""

    schema_version: str
    probe_point_id: str
    state_id: str
    evaluation_case_id: str
    variant: str
    goal: str
    goal_sha256: str
    pre_observation: str
    action: str
    opinion_input_sha256: str

    @classmethod
    def from_dict(
        cls,
        row: Mapping[str, Any],
        *,
        point: GroundingPoint | None = None,
        truth: EvaluationTruthRecord | None = None,
    ) -> "OpinionRatingInput":
        unknown = sorted(set(row) - _INPUT_KEYS)
        missing = sorted(_INPUT_KEYS - set(row))
        if unknown:
            raise OpinionCollectionError(
                "opinion input contains forbidden/unknown fields: " + str(unknown)
            )
        if missing:
            raise OpinionCollectionError(f"opinion input missing fields: {missing}")
        bad_types = sorted(key for key in _INPUT_KEYS if not isinstance(row[key], str))
        if bad_types:
            raise OpinionCollectionError(
                f"opinion input fields must be strings: {bad_types}"
            )
        item = cls(**{key: row[key] for key in _INPUT_KEYS})
        item.validate(point, truth)
        return item

    def validate(
        self,
        point: GroundingPoint | None = None,
        truth: EvaluationTruthRecord | None = None,
    ) -> None:
        if self.schema_version != OPINION_INPUT_SCHEMA_VERSION:
            raise OpinionCollectionError("bad opinion input schema_version")
        for name in (
            "probe_point_id", "state_id", "evaluation_case_id", "variant",
            "goal", "goal_sha256", "pre_observation", "action",
            "opinion_input_sha256",
        ):
            if not getattr(self, name).strip():
                raise OpinionCollectionError(f"opinion input missing {name}")
        for name in ("goal", "pre_observation", "action"):
            if _CREDENTIAL_VALUE.search(getattr(self, name)):
                raise OpinionCollectionError(
                    f"credential-like material is forbidden in opinion input {name}"
                )
        if self.goal_sha256 != _sha256_text(self.goal):
            raise OpinionCollectionError("opinion input goal_sha256 mismatch")
        expected_input_sha = opinion_evidence_sha256(
            self.goal, self.pre_observation, self.action)
        if self.opinion_input_sha256 != expected_input_sha:
            raise OpinionCollectionError("opinion_input_sha256 mismatch")
        if point is not None:
            if self.probe_point_id != point.probe_point_id:
                raise OpinionCollectionError("opinion input probe_point_id mismatch")
            if self.state_id != point.state_id:
                raise OpinionCollectionError("opinion input state_id mismatch")
            if _sha256_text(self.pre_observation) != point.pre_observation_hash:
                raise OpinionCollectionError(
                    "opinion input is not the exact point pre_observation"
                )
            if self.action not in {point.raw_action, point.canonical_action}:
                raise OpinionCollectionError(
                    "opinion input action does not match the probed point action"
                )
        if truth is not None:
            expected = {
                "evaluation_case_id": truth.evaluation_case_id,
                "probe_point_id": truth.probe_point_id,
                "state_id": truth.state_id,
                "variant": truth.variant,
            }
            mismatched = [name for name, value in expected.items()
                          if getattr(self, name) != value]
            if mismatched:
                raise OpinionCollectionError(
                    "opinion input truth identity mismatch: " + ",".join(mismatched)
                )

    def to_dict(self) -> dict[str, str]:
        return {key: getattr(self, key) for key in _INPUT_KEYS}


def make_opinion_rating_input(
    *,
    point: GroundingPoint,
    truth: EvaluationTruthRecord,
    goal: str,
    pre_observation: str,
) -> OpinionRatingInput:
    item = OpinionRatingInput(
        schema_version=OPINION_INPUT_SCHEMA_VERSION,
        probe_point_id=point.probe_point_id,
        state_id=point.state_id,
        evaluation_case_id=truth.evaluation_case_id,
        variant=truth.variant,
        goal=goal,
        goal_sha256=_sha256_text(goal),
        pre_observation=pre_observation,
        action=point.raw_action,
        opinion_input_sha256=opinion_evidence_sha256(
            goal, pre_observation, point.raw_action),
    )
    item.validate(point, truth)
    return item


def load_opinion_rating_inputs(
    path: Path,
    *,
    points: Mapping[str, GroundingPoint],
    truths: Mapping[str, EvaluationTruthRecord],
) -> list[OpinionRatingInput]:
    """Load a strict JSONL batch and require a unique canonical point join."""
    inputs: list[OpinionRatingInput] = []
    seen: set[str] = set()
    for line_no, line in enumerate(Path(path).open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OpinionCollectionError(f"{path}:{line_no}: {exc}") from exc
        if not isinstance(row, Mapping):
            raise OpinionCollectionError(f"{path}:{line_no}: row must be an object")
        point_id = str(row.get("probe_point_id") or "")
        case_id = str(row.get("evaluation_case_id") or "")
        point = points.get(point_id)
        truth = truths.get(case_id)
        if point is None:
            raise OpinionCollectionError(
                f"{path}:{line_no}: unknown probe_point_id {point_id!r}"
            )
        if truth is None:
            raise OpinionCollectionError(
                f"{path}:{line_no}: unknown evaluation_case_id {case_id!r}"
            )
        try:
            item = OpinionRatingInput.from_dict(row, point=point, truth=truth)
        except OpinionCollectionError as exc:
            raise OpinionCollectionError(f"{path}:{line_no}: {exc}") from exc
        if item.evaluation_case_id in seen:
            raise OpinionCollectionError(
                f"{path}:{line_no}: duplicate evaluation_case_id "
                f"{item.evaluation_case_id!r}"
            )
        seen.add(item.evaluation_case_id)
        inputs.append(item)
    if not inputs:
        raise OpinionCollectionError("opinion input batch is empty")
    return inputs


def opinion_input_manifest_row(item: OpinionRatingInput) -> dict[str, Any]:
    return {
        "schema_version": OPINION_INPUT_MANIFEST_SCHEMA_VERSION,
        "input_schema_version": item.schema_version,
        "evaluation_case_id": item.evaluation_case_id,
        "probe_point_id": item.probe_point_id,
        "state_id": item.state_id,
        "variant": item.variant,
        "goal_sha256": item.goal_sha256,
        "opinion_input_sha256": item.opinion_input_sha256,
        "record_sha256": _canonical_sha256(item.to_dict()),
    }


def save_opinion_rating_inputs(
    inputs: Sequence[OpinionRatingInput],
    body: Path,
    manifest: Path,
) -> tuple[Path, Path]:
    if not inputs:
        raise OpinionCollectionError("refusing to materialize empty opinion inputs")
    if Path(body).resolve() == Path(manifest).resolve():
        raise OpinionCollectionError("opinion input body and manifest must differ")
    ordered = sorted(inputs, key=lambda item: item.evaluation_case_id)
    if len({item.evaluation_case_id for item in ordered}) != len(ordered):
        raise OpinionCollectionError("duplicate evaluation_case_id in opinion inputs")
    for item in ordered:
        item.validate()
    body_text = "".join(
        json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for item in ordered)
    manifest_text = "".join(
        json.dumps(opinion_input_manifest_row(item), ensure_ascii=False,
                   sort_keys=True) + "\n"
        for item in ordered)
    body, manifest = Path(body), Path(manifest)
    if body.exists() != manifest.exists():
        raise OpinionCollectionError(
            "opinion input body/manifest must both exist or both be absent")
    if body.exists():
        if (body.read_text(encoding="utf-8") == body_text and
                manifest.read_text(encoding="utf-8") == manifest_text):
            return body, manifest
        raise OpinionCollectionError("refusing to overwrite immutable opinion inputs")
    _atomic_write_pair(body, body_text, manifest, manifest_text)
    return body, manifest


def assert_opinion_input_manifest_integrity(
    body: Path,
    manifest: Path,
    *,
    points: Mapping[str, GroundingPoint],
    truths: Mapping[str, EvaluationTruthRecord],
) -> list[OpinionRatingInput]:
    inputs = load_opinion_rating_inputs(body, points=points, truths=truths)
    rows = [json.loads(line) for line in Path(manifest).open(encoding="utf-8")
            if line.strip()] if Path(manifest).exists() else []
    indexed = {str(row.get("evaluation_case_id") or ""): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != {
            item.evaluation_case_id for item in inputs}:
        raise OpinionCollectionError("opinion input body/manifest is not 1:1")
    for item in inputs:
        if indexed[item.evaluation_case_id] != opinion_input_manifest_row(item):
            raise OpinionCollectionError(
                f"opinion input manifest mismatch {item.evaluation_case_id}")
    return inputs


def prepare_formal_opinion_inputs(
    data_root: Path,
    *,
    variants: Sequence[str] = ("constraint", "request"),
    limit: int = 0,
) -> tuple[list[OpinionRatingInput], dict[str, Any]]:
    """Build label-blind v2 inputs from canonical point/truth/SFT/state joins.

    Evaluation truth supplies identity only (case, point, state, variant).  No
    truth label is copied into an input or report.  Goals and byte-exact
    observations come from the formal single-step policy user message (and the
    observation SHA-256 must equal the point's pre-observation hash); the action
    comes from that point's executed raw action.
    """
    from .governance import formal_release_context

    root = Path(data_root)
    context = formal_release_context(root)
    if context.grounding_error or context.truth_error:
        raise OpinionCollectionError(
            "formal point/truth context is invalid: "
            + (context.grounding_error or context.truth_error)
        )
    wanted_variants = tuple(dict.fromkeys(str(item).strip() for item in variants
                                          if str(item).strip()))
    if not wanted_variants:
        raise OpinionCollectionError("at least one opinion variant is required")
    unknown_variants = sorted(
        set(wanted_variants) - {truth.variant for truth in context.truth.values()})
    if unknown_variants:
        raise OpinionCollectionError(
            f"requested variants absent from formal truth: {unknown_variants}")
    if limit < 0:
        raise OpinionCollectionError("opinion input limit must be non-negative")

    sft_path = root / "train" / "formal" / config.FORMAL_SFT_PATH.name
    sft_by_case: dict[str, dict[str, Any]] = {}
    if sft_path.exists():
        for line_no, line in enumerate(sft_path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OpinionCollectionError(f"{sft_path}:{line_no}: {exc}") from exc
            meta = row.get("meta") or {}
            case_id = str(meta.get("evaluation_case_id") or "")
            if not case_id:
                continue
            if case_id in sft_by_case:
                raise OpinionCollectionError(
                    f"{sft_path}:{line_no}: duplicate evaluation_case_id {case_id}")
            sft_by_case[case_id] = row

    eligible: list[OpinionRatingInput] = []
    excluded: list[dict[str, str]] = []
    relevant_truths = sorted(
        (truth for truth in context.truth.values()
         if truth.variant in wanted_variants),
        key=lambda truth: truth.evaluation_case_id,
    )
    for truth in relevant_truths:
        point = context.points.get(truth.probe_point_id)
        sft = sft_by_case.get(truth.evaluation_case_id)
        reasons: list[str] = []
        if point is None:
            reasons.append("missing_canonical_point")
        if sft is None:
            reasons.append("missing_formal_single_sft_goal")
        goal = ""
        snapshot = ""
        if sft is not None:
            meta = sft.get("meta") or {}
            identity = {
                "probe_point_id": truth.probe_point_id,
                "state_id": truth.state_id,
                "variant": truth.variant,
            }
            if any(str(meta.get(name) or "") != value
                   for name, value in identity.items()):
                reasons.append("formal_sft_truth_identity_mismatch")
            messages = sft.get("messages") or []
            user_messages = [message for message in messages
                             if message.get("role") == "user"]
            if len(user_messages) != 1:
                reasons.append("formal_sft_user_topology_not_stateless")
            else:
                user_content = str(user_messages[0].get("content") or "")
                goal = prompts.parse_user(user_content)["goal"]
                if not goal:
                    reasons.append("formal_sft_goal_missing")
                marker = "<observation>\n"
                if marker not in user_content:
                    reasons.append("formal_sft_observation_missing")
                else:
                    # render_user adds exactly one final newline after the
                    # observation.  Remove that delimiter only; parse_user's
                    # generic .strip() would corrupt hashes when a 6000-char
                    # snapshot legitimately ends in a space/newline.
                    snapshot = user_content.split(marker, 1)[1]
                    if not snapshot.endswith("\n"):
                        reasons.append("formal_sft_observation_delimiter_missing")
                    else:
                        snapshot = snapshot[:-1]
        if point is not None and snapshot and _sha256_text(snapshot) != \
                point.pre_observation_hash:
            reasons.append("formal_sft_exact_observation_hash_mismatch")
        if reasons:
            excluded.append({
                "evaluation_case_id": truth.evaluation_case_id,
                "probe_point_id": truth.probe_point_id,
                "state_id": truth.state_id,
                "variant": truth.variant,
                "reason": ",".join(sorted(set(reasons))),
            })
            continue
        assert point is not None
        eligible.append(make_opinion_rating_input(
            point=point,
            truth=truth,
            goal=goal,
            pre_observation=snapshot,
        ))

    selected = eligible[:limit] if limit else eligible
    deferred = eligible[limit:] if limit else []
    return selected, {
        "schema_version": OPINION_INPUT_SCHEMA_VERSION,
        "requested_variants": list(wanted_variants),
        "n_formal_truth_cases": len(relevant_truths),
        "n_eligible_exact_input_cases": len(eligible),
        "eligible_evaluation_case_ids": [
            item.evaluation_case_id for item in eligible],
        "n_excluded_cases": len(excluded),
        "excluded_cases": excluded,
        "limit": limit,
        "n_selected": len(selected),
        "selected_evaluation_case_ids": [
            item.evaluation_case_id for item in selected],
        "n_deferred_by_limit": len(deferred),
        "deferred_evaluation_case_ids": [
            item.evaluation_case_id for item in deferred],
        "model_facing_fields": ["goal", "pre_observation", "action"],
        "truth_labels_copied": False,
    }


def build_opinion_messages(item: OpinionRatingInput) -> list[dict[str, str]]:
    """Render only the three pre-action evidence fields seen by the rater."""
    item.validate()
    user = prompts.get("opinion_rater_user").format(
        goal=item.goal,
        observation=item.pre_observation,
        action=item.action,
    )
    return [
        {"role": "system", "content": prompts.get("opinion_rater_system")},
        {"role": "user", "content": user},
    ]


@dataclass(frozen=True)
class OpinionCompletion:
    content: str
    response_id: str = ""
    response_model: str = ""
    finish_reason: str = ""


class OpinionCompletionClient(Protocol):
    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int | None,
    ) -> OpinionCompletion: ...


class OpenAICompatibleOpinionClient:
    """Minimal stdlib client; routing is supplied by the caller, never fixed."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key_env: str,
        timeout: int = 90,
    ) -> None:
        parsed = urllib.parse.urlsplit(base_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise OpinionCollectionError(
                "base_url must be an explicit HTTP(S) OpenAI-compatible API root"
            )
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise OpinionCollectionError(
                "base_url must not contain credentials, query parameters, or fragments"
            )
        if not _ENV_NAME.fullmatch(api_key_env or ""):
            raise OpinionCollectionError("api_key_env must be a valid environment name")
        if timeout <= 0:
            raise OpinionCollectionError("timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = timeout

    def _key(self) -> str:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise OpinionCollectionError(
                f"API credential missing; set environment variable {self.api_key_env}"
            )
        return key

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int | None,
    ) -> OpinionCompletion:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [dict(message) for message in messages],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if seed is not None:
            payload["seed"] = seed
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key()}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            raise OpinionCollectionError(f"opinion provider request failed: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OpinionCollectionError("opinion provider returned invalid JSON") from exc
        try:
            choice = data["choices"][0]
            message = choice["message"]
            content = message.get("content") or message.get("reasoning_content") or ""
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise OpinionCollectionError(
                "opinion provider response is not OpenAI-compatible"
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise OpinionCollectionError("opinion provider returned empty content")
        return OpinionCompletion(
            content=content.strip(),
            response_id=str(data.get("id") or ""),
            response_model=str(data.get("model") or ""),
            finish_reason=str(choice.get("finish_reason") or ""),
        )


def parse_opinion_json(text: str) -> dict[str, Any]:
    """Parse one exact object and reject Markdown, extra keys, or weak types."""
    if not isinstance(text, str) or not text.strip():
        raise OpinionCollectionError("opinion response is empty")
    stripped = text.strip()
    if stripped.startswith("```") or stripped.endswith("```"):
        raise OpinionCollectionError("opinion response must not use a Markdown fence")
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise OpinionCollectionError("opinion response is not strict JSON") from exc
    if not isinstance(value, dict):
        raise OpinionCollectionError("opinion response must be one JSON object")
    unknown = sorted(set(value) - _OUTPUT_KEYS)
    missing = sorted(_OUTPUT_KEYS - set(value))
    if unknown or missing:
        raise OpinionCollectionError(
            f"opinion response keys differ: missing={missing}, unknown={unknown}"
        )
    if value["effect"] not in PERCEIVED_EFFECTS:
        raise OpinionCollectionError("invalid opinion effect")
    if value["recovery"] not in PERCEIVED_RECOVERABILITIES:
        raise OpinionCollectionError("invalid opinion recovery")
    if value["normative_risk"] not in _RISK_VALUES:
        raise OpinionCollectionError("invalid opinion normative_risk")
    confidence = value["confidence"]
    if confidence is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise OpinionCollectionError("opinion confidence must be null or within [0,1]")
    if not isinstance(value["rationale"], str) or not value["rationale"].strip():
        raise OpinionCollectionError("opinion rationale must be a non-empty string")
    if _CREDENTIAL_VALUE.search(value["rationale"]):
        raise OpinionCollectionError("credential-like material in opinion rationale")
    return {
        "effect": value["effect"],
        "recovery": value["recovery"],
        "normative_risk": value["normative_risk"],
        "confidence": None if confidence is None else float(confidence),
        "rationale": value["rationale"].strip(),
    }


def validate_collection_settings(
    *,
    provider: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int | None,
) -> dict[str, Any]:
    if not provider.strip():
        raise OpinionCollectionError("provider provenance label is required")
    if not model.strip():
        raise OpinionCollectionError("model is required")
    if isinstance(temperature, bool) or not 0.0 <= float(temperature) <= 2.0:
        raise OpinionCollectionError("temperature must be within [0,2]")
    if isinstance(top_p, bool) or not 0.0 < float(top_p) <= 1.0:
        raise OpinionCollectionError("top_p must be within (0,1]")
    if isinstance(max_tokens, bool) or not 1 <= int(max_tokens) <= 4096:
        raise OpinionCollectionError("max_tokens must be within [1,4096]")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise OpinionCollectionError("seed must be an integer or null")
    return {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "seed": seed,
        "response_format": {"type": "json_object"},
    }


def collect_opinion_records(
    inputs: Sequence[OpinionRatingInput],
    *,
    client: OpinionCompletionClient,
    provider: str,
    model: str,
    rater_id: str,
    collection_timestamp: str,
    import_batch_id: str,
    code_version: str,
    provenance_root: Path,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 300,
    seed: int | None = 0,
    instrument_id: str = OPINION_INSTRUMENT_ID,
    instrument_version: str = OPINION_INSTRUMENT_VERSION,
    points: Mapping[str, GroundingPoint] | None = None,
    truths: Mapping[str, EvaluationTruthRecord] | None = None,
) -> tuple[list[OpinionLabelRecord], dict[str, Any]]:
    """Call an injected client and return canonical opinion-only records."""
    if not inputs:
        raise OpinionCollectionError("opinion collection requires inputs")
    for name, value in (
        ("rater_id", rater_id),
        ("collection_timestamp", collection_timestamp),
        ("import_batch_id", import_batch_id),
        ("code_version", code_version),
        ("instrument_id", instrument_id),
        ("instrument_version", instrument_version),
    ):
        if not str(value).strip():
            raise OpinionCollectionError(f"{name} is required")
    decode = validate_collection_settings(
        provider=provider,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
    )
    generation = prompts.snapshot_generation(
        root=Path(provenance_root),
        producer="revact.data.opinion_collect",
        model={
            "role": "opinion",
            "provider": provider,
            "name": model,
            "base_url": str(getattr(client, "base_url", "injected-client")),
        },
        decode_config=decode,
        author="opinion-collection",
    )
    records: list[OpinionLabelRecord] = []
    response_models: set[str] = set()
    for item in inputs:
        point = points.get(item.probe_point_id) if points is not None else None
        truth = (truths.get(item.evaluation_case_id)
                 if truths is not None else None)
        if points is not None and point is None:
            raise OpinionCollectionError(
                f"unknown collection probe_point_id {item.probe_point_id!r}")
        if truths is not None and truth is None:
            raise OpinionCollectionError(
                f"unknown collection evaluation_case_id "
                f"{item.evaluation_case_id!r}")
        item.validate(point, truth)
        messages = build_opinion_messages(item)
        completion = client.complete(
            messages,
            model=model,
            temperature=decode["temperature"],
            top_p=decode["top_p"],
            max_tokens=decode["max_tokens"],
            seed=decode["seed"],
        )
        parsed = parse_opinion_json(completion.content)
        actual_model = completion.response_model.strip() or model
        response_models.add(actual_model)
        input_messages_sha256 = _canonical_value_sha256(messages)
        raw_response_sha256 = _sha256_text(completion.content)
        source_payload = {
            "input_messages_sha256": input_messages_sha256,
            "raw_response_sha256": raw_response_sha256,
            "response_id": completion.response_id,
            "response_model": actual_model,
            "finish_reason": completion.finish_reason,
        }
        source_record_id = "opinion-call-" + _canonical_sha256(source_payload)[:24]
        record = OpinionLabelRecord(
            schema_version=OPINION_LABEL_SCHEMA_VERSION,
            artifact_role=OPINION_ARTIFACT_ROLE,
            opinion_label_id=make_opinion_label_id(
                item.probe_point_id,
                item.evaluation_case_id,
                item.goal_sha256,
                rater_id,
                instrument_id,
                instrument_version,
            ),
            probe_point_id=item.probe_point_id,
            state_id=item.state_id,
            evaluation_case_id=item.evaluation_case_id,
            variant=item.variant,
            goal_sha256=item.goal_sha256,
            opinion_input_sha256=item.opinion_input_sha256,
            input_messages_sha256=input_messages_sha256,
            raw_response=completion.content,
            raw_response_sha256=raw_response_sha256,
            provider_response_id=completion.response_id,
            response_model=actual_model,
            finish_reason=completion.finish_reason,
            rater_id=rater_id,
            rater_type=RATER_LLM,
            provider=provider,
            model=model,
            prompt_generation_fp=generation["prompt_generation_fp"],
            instrument_id=instrument_id,
            instrument_version=instrument_version,
            perceived_effect=parsed["effect"],
            perceived_recoverability=parsed["recovery"],
            normative_risk_opinion=parsed["normative_risk"],
            confidence=parsed["confidence"],
            rationale=parsed["rationale"],
            source_record_id=source_record_id,
            collection_timestamp=collection_timestamp,
            import_batch_id=import_batch_id,
            code_version=code_version,
        )
        record.validate(point, truth)
        records.append(record)
    return records, {
        "n_records": len(records),
        "provider": provider,
        "requested_model": model,
        "base_url": str(getattr(client, "base_url", "injected-client")),
        "response_models": sorted(response_models),
        "prompts_fp": generation["prompts_fp"],
        "prompt_generation_fp": generation["prompt_generation_fp"],
        "decode": decode,
        "instrument_id": instrument_id,
        "instrument_version": instrument_version,
        "evaluation_case_ids": [
            record.evaluation_case_id for record in records],
        "credential_value_stored": False,
    }
