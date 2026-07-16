"""Immutable stateless episode traces for history/supervision evidence.

IRIS is deployed as one independent ``system + user`` policy call per browser
step.  A multi-step episode therefore must not be represented as one synthetic
alternating chat.  This module stores a sequence of exact stateless calls,
their observed action deltas, and optional links to the formal SFT views that
supervise those calls.

The artifact is deliberately evidence-only.  A trace without an exact formal
sample join is still useful for debugging, but it cannot satisfy publication
readiness.  Likewise, declaring ``turn_type=state_changing`` is insufficient:
the action, pre/post snapshot hashes, and state must join a canonical CHANGED
grounding point.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping

from .. import config, prompts
from ..envs.obs_utils import bid_is_visible, history_entry, prune_axtree_txt
from ..eval.truth import (EvaluationTruthError, EvaluationTruthRecord,
                          assert_truth_manifest_integrity, load_truth_records)
from ..grounding.schema import EFFECT_CHANGED, GroundingPoint
from ..train.validators import (actions_match, answer_text, iris_tag_errors,
                                parse_action)


EPISODE_TRACE_SCHEMA_VERSION = "iris.stateless_episode_trace.v1"
EPISODE_TRACE_MANIFEST_SCHEMA_VERSION = \
    "iris.stateless_episode_trace_manifest.v1"
EPISODE_TRACE_ARTIFACT_VERSION = "stateless_episode_traces.v1"
EPISODE_TRACE_BODY_NAME = f"{EPISODE_TRACE_ARTIFACT_VERSION}.jsonl"
EPISODE_TRACE_MANIFEST_NAME = "STATELESS_EPISODE_TRACE_MANIFEST.v1.jsonl"
POLICY_BUILDER_ID = "revact.prompts.build_policy_messages"
MESSAGE_TOPOLOGY = "stateless"
TURN_TYPES = frozenset({"routine", "state_changing", "decision"})


class EpisodeTraceError(ValueError):
    """A trace is incomplete, contradictory, or not immutably pinned."""


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def observation_sha256(axtree_txt: str) -> str:
    return hashlib.sha256(str(axtree_txt or "").encode("utf-8")).hexdigest()


def _strict_dataclass(cls, row: Mapping[str, Any], *, what: str):
    if not isinstance(row, Mapping):
        raise EpisodeTraceError(f"{what} must be an object")
    known = {field.name for field in fields(cls)}
    unknown = sorted(set(row) - known)
    missing = sorted(known - set(row))
    if unknown:
        raise EpisodeTraceError(f"unknown {what} fields: {unknown}")
    if missing:
        raise EpisodeTraceError(f"missing {what} fields: {missing}")
    try:
        return cls(**dict(row))
    except TypeError as exc:
        raise EpisodeTraceError(f"invalid {what}: {exc}") from exc


@dataclass(frozen=True)
class EpisodeObservation:
    """Exact browser observation used before/after one policy action."""

    url: str
    title: str
    axtree_txt: str
    axtree_sha256: str
    persistent_signal: Any

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_view(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "axtree_txt": self.axtree_txt,
            "persistent_signal": self.persistent_signal,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "EpisodeObservation":
        value = _strict_dataclass(cls, row, what="episode observation")
        value.validate()
        return value

    def validate(self) -> None:
        if not isinstance(self.url, str) or not isinstance(self.title, str) or \
                not isinstance(self.axtree_txt, str):
            raise EpisodeTraceError("observation text fields must be strings")
        if not self.axtree_txt.strip():
            raise EpisodeTraceError("episode observation has an empty AXTree")
        expected = observation_sha256(self.axtree_txt)
        if self.axtree_sha256 != expected:
            raise EpisodeTraceError("episode observation AXTree hash mismatch")
        try:
            canonical_json_sha256(self.persistent_signal)
        except (TypeError, ValueError) as exc:
            raise EpisodeTraceError(
                f"persistent_signal is not portable JSON: {exc}") from exc


@dataclass(frozen=True)
class EpisodeStep:
    """One exact stateless policy call and its observed environment result."""

    step_index: int
    pre_observation: dict[str, Any]
    input_messages: list[dict[str, Any]]
    input_messages_sha256: str
    assistant_completion: str
    raw_action: str
    turn_type: str
    action_legal: bool
    executed: bool
    guarded: bool
    post_observation: dict[str, Any]
    history_entry: dict[str, Any]
    history_steps_total: int
    history_steps_kept: int
    probe_point_id: str
    state_id: str
    supervised_sample_id: str
    evaluation_case_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "EpisodeStep":
        value = _strict_dataclass(cls, row, what="episode step")
        value.validate_basic()
        return value

    def observations(self) -> tuple[EpisodeObservation, EpisodeObservation]:
        return (EpisodeObservation.from_dict(self.pre_observation),
                EpisodeObservation.from_dict(self.post_observation))

    def validate_basic(self) -> None:
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int) \
                or self.step_index < 0:
            raise EpisodeTraceError("step_index must be a non-negative integer")
        if self.turn_type not in TURN_TYPES:
            raise EpisodeTraceError(f"invalid episode turn_type {self.turn_type!r}")
        for name in ("action_legal", "executed", "guarded"):
            if not isinstance(getattr(self, name), bool):
                raise EpisodeTraceError(f"{name} must be boolean")
        for name in ("probe_point_id", "state_id", "supervised_sample_id",
                     "evaluation_case_id"):
            if not isinstance(getattr(self, name), str):
                raise EpisodeTraceError(f"{name} must be a string")
        if not isinstance(self.input_messages, list) or [
                message.get("role") if isinstance(message, dict) else None
                for message in self.input_messages] != ["system", "user"]:
            raise EpisodeTraceError(
                "each episode step input must be exactly system + user")
        if self.input_messages_sha256 != canonical_json_sha256(
                self.input_messages):
            raise EpisodeTraceError("episode input_messages hash mismatch")
        if not isinstance(self.assistant_completion, str) or \
                not self.assistant_completion.strip():
            raise EpisodeTraceError("assistant_completion must be non-empty")
        parsed_answer = parse_action(answer_text(self.assistant_completion))
        parsed_raw = parse_action(self.raw_action)
        if parsed_answer is None or parsed_raw is None or \
                not actions_match(parsed_answer, parsed_raw):
            raise EpisodeTraceError(
                "assistant answer does not match the recorded raw_action")
        if self.turn_type in {"state_changing", "decision"}:
            problems = iris_tag_errors(self.assistant_completion)
            if problems:
                raise EpisodeTraceError(
                    "full IRIS block required for state-changing/decision turn: "
                    + "; ".join(problems))
        if isinstance(self.history_steps_total, bool) or \
                not isinstance(self.history_steps_total, int) or \
                self.history_steps_total < 0:
            raise EpisodeTraceError(
                "history_steps_total must be a non-negative integer")
        if isinstance(self.history_steps_kept, bool) or \
                not isinstance(self.history_steps_kept, int) or \
                self.history_steps_kept < 0:
            raise EpisodeTraceError(
                "history_steps_kept must be a non-negative integer")
        if not isinstance(self.history_entry, dict):
            raise EpisodeTraceError("history_entry must be an object")
        self.observations()


@dataclass(frozen=True)
class StatelessEpisodeTrace:
    """A real episode represented as repeated deployment-shaped calls."""

    schema_version: str
    episode_trace_id: str
    rollout_run_id: str
    episode_id: str
    task_id: str
    trajectory_id: str
    goal: str
    system_prompt: str
    prompts_fp: str
    prompt_generation_fp: str
    policy_builder: str
    message_topology: str
    max_history_steps: int
    max_axtree_chars: int
    environment_family: str
    environment_instance: str
    environment_origin: str
    is_mock: bool
    collector_success: bool
    seed: int
    steps: list[dict[str, Any]]
    timestamp: str
    code_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "StatelessEpisodeTrace":
        value = _strict_dataclass(cls, row, what="stateless episode trace")
        value.validate()
        return value

    def parsed_steps(self) -> list[EpisodeStep]:
        if not isinstance(self.steps, list):
            raise EpisodeTraceError("episode steps must be a list")
        return [EpisodeStep.from_dict(step) for step in self.steps]

    def validate(self) -> None:
        if self.schema_version != EPISODE_TRACE_SCHEMA_VERSION:
            raise EpisodeTraceError(
                f"unsupported episode schema {self.schema_version!r}")
        for name in ("rollout_run_id", "episode_id", "task_id", "trajectory_id",
                     "goal", "system_prompt", "prompts_fp",
                     "prompt_generation_fp", "environment_family",
                     "environment_instance", "environment_origin", "timestamp",
                     "code_version"):
            if not isinstance(getattr(self, name), str) or \
                    not getattr(self, name).strip():
                raise EpisodeTraceError(f"episode trace missing {name}")
        if self.policy_builder != POLICY_BUILDER_ID:
            raise EpisodeTraceError("episode trace uses a non-canonical policy builder")
        if self.message_topology != MESSAGE_TOPOLOGY:
            raise EpisodeTraceError("episode trace must use stateless topology")
        if isinstance(self.max_history_steps, bool) or \
                not isinstance(self.max_history_steps, int) or \
                self.max_history_steps < 0:
            raise EpisodeTraceError("max_history_steps must be non-negative")
        if isinstance(self.max_axtree_chars, bool) or \
                not isinstance(self.max_axtree_chars, int) or \
                self.max_axtree_chars <= 0:
            raise EpisodeTraceError("max_axtree_chars must be positive")
        if not isinstance(self.is_mock, bool) or \
                not isinstance(self.collector_success, bool):
            raise EpisodeTraceError("episode origin/success flags must be boolean")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise EpisodeTraceError("episode seed must be an integer")

        steps = self.parsed_steps()
        if not steps:
            raise EpisodeTraceError("episode trace contains zero steps")
        if [step.step_index for step in steps] != list(range(len(steps))):
            raise EpisodeTraceError("episode step indexes must be contiguous from zero")
        decision_indexes = [step.step_index for step in steps
                            if step.turn_type == "decision"]
        if decision_indexes != [len(steps) - 1]:
            raise EpisodeTraceError(
                "episode must have exactly one final decision turn")

        observed_history: list[dict[str, Any]] = []
        previous_post: EpisodeObservation | None = None
        for step in steps:
            pre, post = step.observations()
            # A canonical episode is a continuous environment execution, not a
            # bag of individually plausible calls.  Without this join a caller
            # could stitch unrelated point snapshots into artificial long
            # history while still satisfying every per-step hash check.
            if previous_post is not None and pre != previous_post:
                raise EpisodeTraceError(
                    f"step {step.step_index} pre observation is not the exact "
                    "previous post observation")
            if step.executed and not step.action_legal:
                raise EpisodeTraceError(
                    f"step {step.step_index} executes an illegal action")
            if step.guarded and (step.executed or not step.action_legal):
                raise EpisodeTraceError(
                    f"step {step.step_index} has contradictory guard/execution "
                    "metadata")
            if not step.executed and pre != post:
                raise EpisodeTraceError(
                    f"step {step.step_index} changes observation without an "
                    "executed action")
            expected_messages = prompts.build_policy_messages(
                self.goal,
                pre.axtree_txt,
                observed_history,
                system_prompt=self.system_prompt,
                max_history=self.max_history_steps,
                max_axtree_chars=self.max_axtree_chars,
            )
            if step.input_messages != expected_messages:
                raise EpisodeTraceError(
                    f"step {step.step_index} is not byte-equivalent to the "
                    "canonical stateless policy builder")
            expected_total = step.step_index
            expected_kept = min(expected_total, self.max_history_steps)
            if step.history_steps_total != expected_total or \
                    step.history_steps_kept != expected_kept:
                raise EpisodeTraceError(
                    f"step {step.step_index} history length metadata mismatch")
            expected_entry = history_entry(
                step.raw_action, pre.to_view(), post.to_view())
            if step.history_entry != expected_entry:
                raise EpisodeTraceError(
                    f"step {step.step_index} history entry is not observed-delta derived")
            action = parse_action(step.raw_action)
            user_obs = prompts.parse_observation_message(
                step.input_messages[1]["content"])
            if action is not None and action.bid and not bid_is_visible(
                    user_obs, action.bid):
                raise EpisodeTraceError(
                    f"step {step.step_index} action bid [{action.bid}] is absent "
                    "from the exact policy input")
            if step.turn_type == "state_changing" and \
                    step.history_entry.get("flag") != "state-change":
                raise EpisodeTraceError(
                    f"step {step.step_index} claims state change without a "
                    "persistent observed delta")
            if step.step_index < len(steps) - 1 and \
                    step.history_entry.get("flag") == "state-change" and \
                    step.turn_type != "state_changing":
                raise EpisodeTraceError(
                    f"step {step.step_index} has a persistent state change but "
                    "is not typed state_changing")
            observed_history.append(step.history_entry)
            previous_post = post

        expected_id = episode_trace_id_for(self)
        if self.episode_trace_id != expected_id:
            raise EpisodeTraceError("episode_trace_id is not content-derived")


def _identity_payload(trace: StatelessEpisodeTrace) -> dict[str, Any]:
    row = trace.to_dict()
    row.pop("episode_trace_id", None)
    return row


def episode_trace_id_for(trace: StatelessEpisodeTrace) -> str:
    return "episode-trace-" + canonical_json_sha256(_identity_payload(trace))[:24]


def episode_trace_paths(data_root: Path) -> tuple[Path, Path]:
    directory = Path(data_root) / "raw" / "episodes"
    return (directory / EPISODE_TRACE_BODY_NAME,
            directory / EPISODE_TRACE_MANIFEST_NAME)


def _body_text(records: Iterable[StatelessEpisodeTrace]) -> str:
    return "".join(
        json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for record in records)


def _manifest_row(trace: StatelessEpisodeTrace, *, body_sha256: str) -> dict:
    steps = trace.parsed_steps()
    return {
        "schema_version": EPISODE_TRACE_MANIFEST_SCHEMA_VERSION,
        "artifact_version": EPISODE_TRACE_ARTIFACT_VERSION,
        "episode_trace_id": trace.episode_trace_id,
        "episode_id": trace.episode_id,
        "rollout_run_id": trace.rollout_run_id,
        "n_steps": len(steps),
        "input_messages_sha256": canonical_json_sha256(
            [step.input_messages_sha256 for step in steps]),
        "record_sha256": canonical_json_sha256(trace.to_dict()),
        "body_sha256": body_sha256,
    }


def _atomic_write_pair(body: Path, body_text: str, manifest: Path,
                       manifest_text: str) -> None:
    body, manifest = Path(body), Path(manifest)
    if body.parent.resolve() != manifest.parent.resolve():
        raise EpisodeTraceError("episode body and manifest must share a directory")
    body.parent.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    try:
        for target, text in ((body, body_text), (manifest, manifest_text)):
            fd, temporary = tempfile.mkstemp(
                prefix=f".{target.name}.", dir=target.parent)
            staged.append(temporary)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(staged[0], body)
        staged[0] = ""
        os.replace(staged[1], manifest)
        staged[1] = ""
    finally:
        for temporary in staged:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)


def load_episode_traces(body: Path) -> dict[str, StatelessEpisodeTrace]:
    traces: dict[str, StatelessEpisodeTrace] = {}
    episode_keys: set[tuple[str, str]] = set()
    body = Path(body)
    if not body.exists():
        return traces
    for line_no, line in enumerate(body.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            trace = StatelessEpisodeTrace.from_dict(raw)
        except (json.JSONDecodeError, EpisodeTraceError) as exc:
            raise EpisodeTraceError(f"{body}:{line_no}: {exc}") from exc
        if trace.episode_trace_id in traces:
            raise EpisodeTraceError(
                f"{body}:{line_no}: duplicate episode_trace_id")
        episode_key = (trace.rollout_run_id, trace.episode_id)
        if episode_key in episode_keys:
            raise EpisodeTraceError(
                f"{body}:{line_no}: duplicate rollout_run_id + episode_id")
        episode_keys.add(episode_key)
        traces[trace.episode_trace_id] = trace
    return traces


def assert_episode_manifest_integrity(
    body: Path, manifest: Path | None = None,
) -> dict[str, StatelessEpisodeTrace]:
    body = Path(body)
    manifest = (Path(manifest) if manifest else
                body.with_name(EPISODE_TRACE_MANIFEST_NAME))
    if not body.exists() or not manifest.exists():
        raise EpisodeTraceError(
            "episode trace body and manifest must both exist")
    traces = load_episode_traces(body)
    if not traces:
        raise EpisodeTraceError("episode trace artifact contains zero rows")
    try:
        rows = [json.loads(line) for line in manifest.open(encoding="utf-8")
                if line.strip()]
    except (json.JSONDecodeError, OSError) as exc:
        raise EpisodeTraceError(f"invalid episode trace manifest: {exc}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise EpisodeTraceError("episode manifest rows must be objects")
    indexed = {str(row.get("episode_trace_id") or ""): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(traces):
        raise EpisodeTraceError("episode trace body/manifest is not 1:1")
    body_sha = hashlib.sha256(body.read_bytes()).hexdigest()
    for trace_id, trace in traces.items():
        if indexed[trace_id] != _manifest_row(trace, body_sha256=body_sha):
            raise EpisodeTraceError(
                f"episode trace manifest/hash mismatch {trace_id}")
    return traces


def save_episode_traces(
    records: Iterable[StatelessEpisodeTrace],
    body: Path,
    manifest: Path | None = None,
    *,
    append: bool = False,
) -> tuple[Path, Path]:
    """Validate and immutably store a body/manifest pair."""
    body = Path(body)
    manifest = (Path(manifest) if manifest else
                body.with_name(EPISODE_TRACE_MANIFEST_NAME))
    if body.exists() != manifest.exists():
        raise EpisodeTraceError(
            "cannot update a partial episode trace body/manifest pair")
    if body.exists() and not append:
        raise EpisodeTraceError(
            "refusing to overwrite a versioned episode trace artifact; use append")
    existing = (assert_episode_manifest_integrity(body, manifest)
                if append and body.exists() else {})
    merged = dict(existing)
    materialized = list(records)
    if not materialized and not merged:
        raise EpisodeTraceError("refusing to save zero episode traces")
    for trace in materialized:
        trace.validate()
        previous = merged.get(trace.episode_trace_id)
        if previous is not None and previous != trace:
            raise EpisodeTraceError(
                f"immutable episode_trace_id collision {trace.episode_trace_id}")
        merged[trace.episode_trace_id] = trace
    ordered = [merged[trace_id] for trace_id in sorted(merged)]
    episode_keys = [
        (trace.rollout_run_id, trace.episode_id) for trace in ordered]
    if len(episode_keys) != len(set(episode_keys)):
        raise EpisodeTraceError(
            "duplicate rollout_run_id + episode_id in episode artifact")
    # A truth case is one immutable evaluation unit.  Reusing it in two episode
    # rows would make history coverage and downstream metrics count the same
    # normative case twice even though both trace identities are distinct.
    final_case_ids = [
        step.evaluation_case_id
        for trace in ordered
        for step in trace.parsed_steps()
        if step.turn_type == "decision" and step.evaluation_case_id
    ]
    if len(final_case_ids) != len(set(final_case_ids)):
        raise EpisodeTraceError(
            "duplicate final evaluation_case_id in episode artifact")
    body_text = _body_text(ordered)
    body_sha = hashlib.sha256(body_text.encode("utf-8")).hexdigest()
    manifest_text = "".join(
        json.dumps(_manifest_row(trace, body_sha256=body_sha),
                   ensure_ascii=False, sort_keys=True) + "\n"
        for trace in ordered)
    _atomic_write_pair(body, body_text, manifest, manifest_text)
    return body, manifest


def import_episode_trace_file(
    input_path: Path,
    data_root: Path,
    *,
    append: bool = False,
    points: Mapping[str, GroundingPoint] | None = None,
    formal_rows: list[dict] | None = None,
    truths: Mapping[str, EvaluationTruthRecord] | None = None,
) -> dict[str, Any]:
    """Strictly import authored/live-captured canonical episode JSONL.

    This is intentionally not a converter for the legacy ``StepRecord`` wire
    format.  Those records omit exact policy messages and per-step completions,
    so accepting them here would manufacture evidence the collector never
    recorded.  The publication gate independently checks every imported step
    against a formal SFT row and every intermediate state change against a
    canonical grounding point.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise EpisodeTraceError(
            f"episode trace input does not exist: {input_path}")
    records: list[StatelessEpisodeTrace] = []
    try:
        for line_no, line in enumerate(input_path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                records.append(StatelessEpisodeTrace.from_dict(row))
            except (json.JSONDecodeError, EpisodeTraceError) as exc:
                raise EpisodeTraceError(
                    f"{input_path}:{line_no}: {exc}") from exc
    except OSError as exc:
        raise EpisodeTraceError(
            f"cannot read episode trace input {input_path}: {exc}") from exc
    if not records:
        raise EpisodeTraceError("episode trace input contains zero rows")
    for trace in records:
        for step in trace.parsed_steps():
            ident = f"{trace.episode_trace_id}:step-{step.step_index}"
            if not step.supervised_sample_id:
                raise EpisodeTraceError(
                    f"{ident}: canonical import requires supervised_sample_id")
            if step.turn_type in {"state_changing", "decision"} and (
                    not step.probe_point_id or not step.state_id):
                raise EpisodeTraceError(
                    f"{ident}: {step.turn_type} turn requires point/state join")
            if step.turn_type == "decision" and not step.evaluation_case_id:
                raise EpisodeTraceError(
                    f"{ident}: decision turn requires evaluation_case_id")

    supplied = (points is not None, formal_rows is not None, truths is not None)
    if any(supplied) and not all(supplied):
        raise EpisodeTraceError(
            "points, formal_rows and truths must be supplied together")
    if points is None:
        from ..grounding.schema import (assert_manifest_integrity,
                                        load_probe_points)

        point_body = Path(data_root) / "grounded" / "probe_points.jsonl"
        point_manifest = Path(data_root) / "grounded" / "POINT_MANIFEST.jsonl"
        try:
            assert_manifest_integrity(point_body, point_manifest)
            points = load_probe_points(point_body, validate=True)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise EpisodeTraceError(
                f"canonical grounding is unavailable: {exc}") from exc
        truth_body = Path(data_root) / "eval" / "truth.jsonl"
        truth_manifest = Path(data_root) / "eval" / "TRUTH_MANIFEST.jsonl"
        try:
            if not truth_body.exists() or not truth_manifest.exists():
                raise EvaluationTruthError(
                    "evaluation truth body and manifest must both exist")
            assert_truth_manifest_integrity(
                truth_body, truth_manifest, points)
            truths = load_truth_records(truth_body, points=points)
        except (OSError, EvaluationTruthError, ValueError,
                json.JSONDecodeError) as exc:
            raise EpisodeTraceError(
                f"canonical evaluation truth is unavailable: {exc}") from exc
        formal_rows = []
        for name in (config.FORMAL_SFT_PATH.name,
                     config.FORMAL_MULTITURN_SFT_PATH.name):
            path = Path(data_root) / "train" / "formal" / name
            if not path.exists():
                continue
            try:
                formal_rows.extend(
                    json.loads(line) for line in path.open(encoding="utf-8")
                    if line.strip())
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                raise EpisodeTraceError(
                    f"invalid formal SFT body {path}: {exc}") from exc

    # Audit the exact joins before touching the canonical body.  A malformed
    # import must not leave an immutable-but-unusable episode identity that a
    # later corrected capture cannot append under the same rollout id.
    with tempfile.TemporaryDirectory(prefix="iris-episode-import-") as staging:
        stage_root = Path(staging)
        stage_body, stage_manifest = episode_trace_paths(stage_root)
        save_episode_traces(records, stage_body, stage_manifest)
        evidence = audit_episode_trace_evidence(
            stage_root, points or {}, formal_rows or [], truths or {})
    expected_steps = sum(len(trace.parsed_steps()) for trace in records)
    expected_intermediate = sum(
        step.turn_type == "state_changing" and
        step.step_index < len(trace.parsed_steps()) - 1
        for trace in records for step in trace.parsed_steps())
    rejected = evidence.get("supervision_rejections") or {}
    intermediate = evidence.get("intermediate_state_changing") or {}
    truth_join = evidence.get("final_decision_truth_join") or {}
    if (evidence.get("integrity") is not True or
            evidence.get("n_supervised_verified_steps") != expected_steps or
            rejected or evidence.get("configuration_mismatches") or
            truth_join.get("verified_count") != len(records) or
            truth_join.get("rejected_turns") or
            int(intermediate.get("verified_count") or 0) !=
            expected_intermediate or
            (intermediate.get("rejected_turns") or {})):
        raise EpisodeTraceError(
            "episode release join failed: " + json.dumps({
                "expected_steps": expected_steps,
                "verified_steps": evidence.get("n_supervised_verified_steps"),
                "supervision_rejections": rejected,
                "configuration_mismatches": evidence.get(
                    "configuration_mismatches"),
                "final_decision_truth_join": truth_join,
                "expected_intermediate_state_changes": expected_intermediate,
                "intermediate": intermediate,
            }, ensure_ascii=False, sort_keys=True))
    body, manifest = episode_trace_paths(Path(data_root))
    save_episode_traces(records, body, manifest, append=append)
    stored = assert_episode_manifest_integrity(body, manifest)
    return {
        "schema_version": "iris.stateless_episode_trace_import.v1",
        "input": str(input_path),
        "body": str(body),
        "manifest": str(manifest),
        "n_imported": len(records),
        "n_stored": len(stored),
        "n_exact_sft_joins": expected_steps,
        "n_intermediate_state_changes": expected_intermediate,
        "append": bool(append),
    }


def _point_snapshot_hash(observation: EpisodeObservation, raw_action: str) -> str:
    """Reproduce point_runner's action-anchored snapshot hashing contract."""
    action = parse_action(raw_action)
    anchors = [action.bid] if action is not None and action.bid else []
    snapshot = prune_axtree_txt(
        observation.axtree_txt,
        max_chars=config.MAX_AXTREE_CHARS_SNAPSHOT,
        anchor_bids=anchors,
    )
    return observation_sha256(snapshot)


def audit_episode_trace_evidence(
    data_root: Path,
    points: Mapping[str, GroundingPoint],
    formal_rows: list[dict],
    truths: Mapping[str, EvaluationTruthRecord] | None = None,
) -> dict[str, Any]:
    """Audit trace-backed long history and intermediate supervision.

    Only an exact formal SFT join contributes to a history bucket.  An
    intermediate state-changing turn additionally must join a canonical
    CHANGED point on action, state and action-anchored pre/post hashes.
    """
    body, manifest = episode_trace_paths(data_root)
    truth_body = Path(data_root) / "eval" / "truth.jsonl"
    truth_manifest = Path(data_root) / "eval" / "TRUTH_MANIFEST.jsonl"
    result: dict[str, Any] = {
        "body_path": str(body),
        "manifest_path": str(manifest),
        "body_exists": body.exists(),
        "manifest_exists": manifest.exists(),
        "integrity": False,
        "n_episodes": 0,
        "n_steps": 0,
        "n_supervised_verified_steps": 0,
        "truth_body_path": str(truth_body),
        "truth_manifest_path": str(truth_manifest),
        "truth_integrity": False,
        "n_truth_records": 0,
        "history_buckets": {"1_3": 0, "4_6": 0, "ge_7": 0},
        "history_lengths": [],
        "configured_history_budget": config.POLICY_HISTORY_STEPS,
        "configured_axtree_budget": config.MAX_AXTREE_CHARS_POLICY,
        "configuration_supports_ge_7": config.POLICY_HISTORY_STEPS >= 7,
        "configuration_mismatches": [],
        "supervision_rejections": {},
        "intermediate_state_changing": {
            "verified_count": 0,
            "verified_turn_ids": [],
            "rejected_turns": {},
            "passed": False,
        },
        "final_decision_truth_join": {
            "verified_count": 0,
            "verified_turn_ids": [],
            "rejected_turns": {},
            "passed": False,
        },
        "error": "",
    }
    if body.exists() != manifest.exists():
        result["error"] = "episode trace body/manifest must both exist or be absent"
        return result
    if not body.exists():
        result["error"] = "canonical stateless episode trace artifact is absent"
        return result
    try:
        traces = assert_episode_manifest_integrity(body, manifest)
    except (EpisodeTraceError, json.JSONDecodeError, OSError, TypeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    try:
        if truths is None:
            if not truth_body.exists() or not truth_manifest.exists():
                raise EvaluationTruthError(
                    "evaluation truth body and manifest must both exist")
            assert_truth_manifest_integrity(
                truth_body, truth_manifest, points)
            truth_index = load_truth_records(truth_body, points=points)
        else:
            truth_index = dict(truths)
            for case_id, truth in truth_index.items():
                if not isinstance(truth, EvaluationTruthRecord):
                    raise EvaluationTruthError(
                        f"truth mapping value {case_id!r} is not an "
                        "EvaluationTruthRecord")
                if case_id != truth.evaluation_case_id:
                    raise EvaluationTruthError(
                        f"truth mapping key mismatch {case_id!r}")
                point = points.get(truth.probe_point_id)
                if point is None:
                    raise EvaluationTruthError(
                        f"{case_id}: truth references unknown probe_point_id")
                truth.validate(point)
    except (OSError, EvaluationTruthError, ValueError,
            json.JSONDecodeError) as exc:
        result["error"] = f"canonical evaluation truth invalid: {exc}"
        return result
    result["truth_integrity"] = True
    result["n_truth_records"] = len(truth_index)
    result["integrity"] = True
    result["n_episodes"] = len(traces)

    sample_counts: dict[str, int] = {}
    samples: dict[str, dict] = {}
    for row in formal_rows:
        sample_id = str(row.get("sample_id") or "")
        sample_counts[sample_id] = sample_counts.get(sample_id, 0) + 1
        if sample_id:
            samples[sample_id] = row

    lengths: list[int] = []
    intermediate_verified: list[str] = []
    intermediate_rejected: dict[str, list[str]] = {}
    final_truth_verified: list[str] = []
    final_truth_rejected: dict[str, list[str]] = {}
    supervision_rejections: dict[str, list[str]] = {}
    config_mismatches: list[str] = []
    final_case_counts: dict[str, int] = {}
    for trace in traces.values():
        final = trace.parsed_steps()[-1]
        case_id = final.evaluation_case_id
        if case_id:
            final_case_counts[case_id] = final_case_counts.get(case_id, 0) + 1
    for trace in traces.values():
        steps = trace.parsed_steps()
        result["n_steps"] += len(steps)
        compatible = True
        if trace.max_history_steps != config.POLICY_HISTORY_STEPS:
            compatible = False
            config_mismatches.append(
                f"{trace.episode_trace_id}:history={trace.max_history_steps}")
        if trace.max_axtree_chars != config.MAX_AXTREE_CHARS_POLICY:
            compatible = False
            config_mismatches.append(
                f"{trace.episode_trace_id}:axtree={trace.max_axtree_chars}")
        if trace.is_mock is not False or trace.collector_success is not True:
            compatible = False
            config_mismatches.append(
                f"{trace.episode_trace_id}:nonformal_origin_or_failed_collector")

        for step in steps:
            ident = f"{trace.episode_trace_id}:step-{step.step_index}"
            problems: list[str] = []
            if not compatible:
                problems.append("trace is not current-deployment compatible")
            if not step.supervised_sample_id:
                problems.append("missing supervised_sample_id")
                sample = None
            elif sample_counts.get(step.supervised_sample_id, 0) != 1:
                problems.append("supervised_sample_id is absent or non-unique")
                sample = None
            else:
                sample = samples[step.supervised_sample_id]
                messages = sample.get("messages") or []
                expected_messages = step.input_messages + [{
                    "role": "assistant", "content": step.assistant_completion}]
                if messages != expected_messages:
                    problems.append("formal SFT messages differ from trace call/completion")
                meta = sample.get("meta") or {}
                if meta.get("formal_dataset") is not True:
                    problems.append("joined SFT row is not formal")
                if meta.get("message_topology") != MESSAGE_TOPOLOGY:
                    problems.append("joined SFT row is not stateless")
                if meta.get("history_source") != "trajectory":
                    problems.append("joined SFT row lacks trajectory history")
                if meta.get("history_steps_total") != step.history_steps_total:
                    problems.append("joined SFT history_steps_total mismatch")
                if meta.get("history_steps_kept") != step.history_steps_kept:
                    problems.append("joined SFT history_steps_kept mismatch")
                if step.probe_point_id and meta.get("probe_point_id") != \
                        step.probe_point_id:
                    problems.append("joined SFT probe_point_id mismatch")
                if step.state_id and meta.get("state_id") != step.state_id:
                    problems.append("joined SFT state_id mismatch")
                if step.evaluation_case_id and \
                        meta.get("evaluation_case_id") != \
                        step.evaluation_case_id:
                    problems.append("joined SFT evaluation_case_id mismatch")
                if step.probe_point_id:
                    joined_point = points.get(step.probe_point_id)
                    if joined_point is None:
                        problems.append("joined SFT references unknown probe_point_id")
                    elif step.state_id != joined_point.state_id:
                        problems.append("joined SFT point/state mismatch")
                turn_types = meta.get("assistant_turn_types") or []
                if turn_types != [step.turn_type]:
                    problems.append("joined SFT turn_type mismatch")
                if meta.get("prompts_fp") != trace.prompts_fp or \
                        meta.get("prompt_generation_fp") != \
                        trace.prompt_generation_fp:
                    problems.append("joined SFT prompt provenance mismatch")
            if step.turn_type == "decision":
                case_id = step.evaluation_case_id
                if not case_id:
                    problems.append(
                        "final decision missing evaluation_case_id")
                    truth = None
                elif final_case_counts.get(case_id) != 1:
                    problems.append(
                        "final evaluation_case_id is not unique across episodes")
                    truth = truth_index.get(case_id)
                else:
                    truth = truth_index.get(case_id)
                if not step.probe_point_id or not step.state_id:
                    problems.append(
                        "final decision missing probe_point_id/state_id")
                if truth is None and case_id:
                    problems.append(
                        "final decision references unknown evaluation_case_id")
                elif truth is not None:
                    if truth.probe_point_id != step.probe_point_id:
                        problems.append(
                            "final truth/trace probe_point_id mismatch")
                    if truth.state_id != step.state_id:
                        problems.append("final truth/trace state_id mismatch")
                    joined_point = points.get(step.probe_point_id)
                    if joined_point is None:
                        problems.append(
                            "final decision references unknown probe_point_id")
                    elif joined_point.state_id != step.state_id:
                        problems.append("final point/trace state_id mismatch")
                    if sample is not None:
                        meta = sample.get("meta") or {}
                        expected_meta = {
                            "evaluation_case_id": case_id,
                            "probe_point_id": step.probe_point_id,
                            "state_id": step.state_id,
                        }
                        mismatched = [
                            name for name, expected in expected_meta.items()
                            if meta.get(name) != expected
                        ]
                        if mismatched:
                            problems.append(
                                "final truth/trace/SFT meta mismatch: " +
                                ",".join(mismatched))
            if problems:
                supervision_rejections[ident] = list(dict.fromkeys(problems))
            else:
                lengths.append(step.history_steps_kept)
            if step.turn_type == "decision":
                if problems:
                    final_truth_rejected[ident] = list(
                        dict.fromkeys(problems))
                else:
                    final_truth_verified.append(ident)

            if step.turn_type != "state_changing" or \
                    step.step_index == len(steps) - 1:
                continue
            state_problems = list(problems)
            if step.action_legal is not True or step.executed is not True or \
                    step.guarded is not False:
                state_problems.append(
                    "state-changing evidence must be legal, executed and unguarded")
            point = points.get(step.probe_point_id)
            if point is None:
                state_problems.append("unknown canonical probe_point_id")
            else:
                if point.effect_status != EFFECT_CHANGED:
                    state_problems.append("joined point is not CHANGED")
                if point.state_id != step.state_id:
                    state_problems.append("joined point state_id mismatch")
                if point.task_id != trace.task_id:
                    state_problems.append("joined point task mismatch")
                # ``GroundingPoint.trajectory_id`` identifies the earlier
                # collection run that supplied the probed state.  A fresh live
                # replay necessarily has a new trajectory id; requiring them
                # to match would make legitimate episode evidence impossible.
                # State/action/hash/signal/environment joins below are the
                # replay identity contract.
                if point.environment_family != trace.environment_family or \
                        point.environment_instance != trace.environment_instance or \
                        point.environment_origin != trace.environment_origin or \
                        point.is_mock != trace.is_mock:
                    state_problems.append("joined point environment identity mismatch")
                if not actions_match(parse_action(step.raw_action), point.raw_action):
                    state_problems.append("assistant action does not match joined point")
                pre, post = step.observations()
                if pre.persistent_signal != point.pre_signal or \
                        post.persistent_signal != point.post_signal:
                    state_problems.append("joined point persistent signal mismatch")
                if _point_snapshot_hash(pre, step.raw_action) != \
                        point.pre_observation_hash:
                    state_problems.append("joined point pre hash mismatch")
                if _point_snapshot_hash(post, step.raw_action) != \
                        point.post_observation_hash:
                    state_problems.append("joined point post hash mismatch")
            if state_problems:
                intermediate_rejected[ident] = list(dict.fromkeys(state_problems))
            else:
                intermediate_verified.append(ident)

    buckets = {
        "1_3": sum(1 <= length <= 3 for length in lengths),
        "4_6": sum(4 <= length <= 6 for length in lengths),
        "ge_7": sum(length >= 7 for length in lengths),
    }
    result.update({
        "n_supervised_verified_steps": len(lengths),
        "history_buckets": buckets,
        "history_lengths": lengths,
        "configuration_mismatches": sorted(config_mismatches),
        "supervision_rejections": supervision_rejections,
        "intermediate_state_changing": {
            "verified_count": len(intermediate_verified),
            "verified_turn_ids": intermediate_verified,
            "rejected_turns": intermediate_rejected,
            "passed": bool(intermediate_verified) and not intermediate_rejected,
        },
        "final_decision_truth_join": {
            "verified_count": len(final_truth_verified),
            "verified_turn_ids": final_truth_verified,
            "rejected_turns": final_truth_rejected,
            "passed": (
                len(final_truth_verified) == len(traces) and
                not final_truth_rejected),
        },
        "error": "",
    })
    return result
