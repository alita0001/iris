"""Fail-closed preflight for raw collector episode sources.

This module deliberately stops one stage before canonical episode authoring.
It verifies that a completed collection-run manifest and its raw trajectory
form one continuous sequence of byte-exact stateless policy calls, then writes
an immutable review sheet whose formal joins and turn types remain empty.

Passing this preflight is evidence that the *source capture* is internally
consistent.  It is never evidence of a grounding label, normative truth, or a
formal supervised sample.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import fields
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .. import config, prompts
from ..envs.harness import StepRecord
from ..envs.obs_utils import (bid_is_visible, history_entry,
                              prune_axtree_txt)
from ..policies import parse_action as extract_policy_action
from ..train.validators import actions_match, parse_action
from .episode_traces import (EpisodeObservation, EpisodeTraceError,
                             canonical_json_sha256)


REVIEW_SCHEMA_VERSION = "iris.raw_episode_source_review.v1"
RAW_SOURCE_SCHEMA_VERSION = "iris.raw_episode_source.v1"
COLLECTION_RUN_SCHEMA_VERSION = "iris.collection-run.v2"
POLICY_BUILDER_ID = "revact.prompts.build_policy_messages"
MESSAGE_TOPOLOGY = "stateless"
MISSING_FORMAL_JOINS = (
    "supervised_sample_id", "probe_point_id", "evaluation_case_id",
)


class EpisodeSourcePreflightError(ValueError):
    """The raw source is incomplete, mutable, stitched, or contradictory."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EpisodeSourcePreflightError(message)


def _nonempty_text(value: Any, name: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()),
             f"missing or invalid {name}")
    return value


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    _require(not isinstance(value, bool) and isinstance(value, int) and
             value >= minimum,
             f"{name} must be an integer >= {minimum}")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json_object(path: Path, name: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise EpisodeSourcePreflightError(
            f"cannot load {name} {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{name} must be one JSON object")
    return value, raw


def _load_raw_rows(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EpisodeSourcePreflightError(
            f"cannot read raw trajectory {path}: {exc}") from exc
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EpisodeSourcePreflightError(
            f"raw trajectory is not UTF-8: {exc}") from exc
    rows: list[dict[str, Any]] = []
    expected_fields = {field.name for field in fields(StepRecord)}
    for lineno, line in enumerate(decoded.splitlines(), start=1):
        _require(bool(line.strip()),
                 f"raw trajectory contains blank line {lineno}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EpisodeSourcePreflightError(
                f"invalid raw trajectory JSON at line {lineno}: {exc}") from exc
        _require(isinstance(row, dict),
                 f"raw trajectory line {lineno} must be an object")
        actual_fields = set(row)
        missing = sorted(expected_fields - actual_fields)
        unknown = sorted(actual_fields - expected_fields)
        _require(not missing and not unknown,
                 f"raw trajectory line {lineno} StepRecord schema mismatch: "
                 f"missing={missing} unknown={unknown}")
        rows.append(row)
    _require(bool(rows), "raw trajectory contains zero StepRecord rows")
    return rows, raw


def _safe_artifact_path(data_root: Path, artifact: Any,
                        trajectory_id: str) -> tuple[Path, str]:
    text = _nonempty_text(artifact, "trajectory raw_artifact")
    _require("\\" not in text, "raw_artifact must use POSIX separators")
    relative = PurePosixPath(text)
    _require(not relative.is_absolute() and ".." not in relative.parts,
             "raw_artifact must be a relative path without traversal")
    _require(relative.name == f"{trajectory_id}.jsonl",
             "raw_artifact filename does not match physical trajectory_id")
    target = (data_root / Path(*relative.parts)).resolve()
    try:
        target.relative_to(data_root)
    except ValueError as exc:
        raise EpisodeSourcePreflightError(
            "raw_artifact resolves outside data_root") from exc
    _require(target.is_file(), f"raw_artifact does not exist: {target}")
    return target, relative.as_posix()


def _history_views(observation: EpisodeObservation,
                   *, signal_mode: str) -> dict[str, Any]:
    view = observation.to_view()
    if signal_mode == "backend":
        # Mock/source collectors expose this channel as backend_state.  The raw
        # portable observation calls it persistent_signal; the value is exact.
        view["backend_state"] = observation.persistent_signal
        view.pop("persistent_signal", None)
    return view


def _verify_history_delta(raw_action: str, pre: EpisodeObservation,
                          post: EpisodeObservation, actual: Any,
                          step_id: int) -> tuple[dict[str, Any], str]:
    _require(isinstance(actual, dict),
             f"step {step_id} observed_history_entry must be an object")
    candidates = {
        mode: history_entry(
            raw_action,
            _history_views(pre, signal_mode=mode),
            _history_views(post, signal_mode=mode),
        )
        for mode in ("persistent", "backend")
    }
    matches = [mode for mode, expected in candidates.items()
               if actual == expected]
    _require(bool(matches),
             f"step {step_id} observed_history_entry is not derived from the "
             "captured pre/post observations")
    return dict(actual), matches[0]


def _completion_action(completion: str):
    parsed = parse_action(completion)
    if parsed is not None:
        return parsed
    extracted = extract_policy_action(completion)
    return parse_action(extracted) if extracted else None


def _review_identity(payload: Mapping[str, Any]) -> str:
    return "episode-source-review-" + canonical_json_sha256(payload)[:24]


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Install one JSON object without ever replacing an existing review."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise EpisodeSourcePreflightError(
            f"review output already exists and is immutable: {path}")
    text = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2,
        allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise EpisodeSourcePreflightError(
                f"review output already exists and is immutable: {path}") from exc
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def preflight_episode_source(
    run_manifest: Path,
    trajectory_id: str,
    output: Path,
    data_root: Path,
) -> dict[str, Any]:
    """Validate one raw source and freeze a non-formal review sheet.

    This function never calls :mod:`revact.data.episode_traces` import/save
    functions and never writes beneath the canonical ``raw/episodes`` path.
    """
    root = Path(data_root).resolve()
    manifest_path = Path(run_manifest).resolve()
    output_path = Path(output).resolve()
    expected_manifest_dir = (root / "manifests" / "collection_runs").resolve()
    _require(manifest_path.parent == expected_manifest_dir,
             "run manifest must be directly under "
             "data_root/manifests/collection_runs")
    _require(output_path != manifest_path,
             "review output cannot replace the run manifest")
    canonical_episode_dir = (root / "raw" / "episodes").resolve()
    try:
        output_path.relative_to(canonical_episode_dir)
    except ValueError:
        pass
    else:
        raise EpisodeSourcePreflightError(
            "review output cannot be written in the canonical episode directory")

    manifest, manifest_bytes = _load_json_object(
        manifest_path, "collection run manifest")
    _require(manifest.get("schema_version") == COLLECTION_RUN_SCHEMA_VERSION,
             "unsupported collection run manifest schema")
    run_id = _nonempty_text(manifest.get("run_id"), "manifest run_id")
    _require(manifest_path.name == f"{run_id}.json",
             "manifest filename does not match run_id")
    _require(manifest.get("status") == "COMPLETE",
             "collection run must be COMPLETE")
    code_version = _nonempty_text(
        manifest.get("code_version"), "manifest code_version")
    summaries = manifest.get("trajectories")
    _require(isinstance(summaries, list) and bool(summaries),
             "collection run has no trajectory summaries")
    summary_ids: list[str] = []
    for index, item in enumerate(summaries):
        _require(isinstance(item, dict),
                 f"trajectory summary {index} must be an object")
        summary_id = _nonempty_text(
            item.get("trajectory_id"),
            f"trajectory summary {index} trajectory_id")
        summary_ids.append(summary_id)
        _require(item.get("run_id") == run_id,
                 f"trajectory summary {summary_id} run_id mismatch")
    _require(len(summary_ids) == len(set(summary_ids)),
             "collection run contains duplicate physical trajectory_id values")
    requested_id = _nonempty_text(trajectory_id, "requested trajectory_id")
    selected = [item for item in summaries
                if item.get("trajectory_id") == requested_id]
    _require(len(selected) == 1,
             "requested physical trajectory_id must match exactly one summary")
    summary = selected[0]

    task_id = _nonempty_text(summary.get("task_id"), "summary task_id")
    seed = _strict_int(summary.get("seed"), "summary seed")
    logical_id = _nonempty_text(
        summary.get("logical_trajectory_id"),
        "summary logical_trajectory_id")
    _require(logical_id == f"{task_id}_seed{seed}",
             "logical_trajectory_id does not match task_id and seed")
    _require(requested_id == f"{logical_id}__run_{run_id}",
             "physical trajectory_id does not match logical id and run_id")
    _require(summary.get("code_version") == code_version,
             "trajectory and run code_version mismatch")
    goal = _nonempty_text(summary.get("goal"), "trajectory goal")
    environment_origin = _nonempty_text(
        summary.get("environment_origin"), "trajectory environment_origin")
    environment_family = _nonempty_text(
        summary.get("environment_family"), "trajectory environment_family")
    _require(isinstance(summary.get("is_mock"), bool),
             "trajectory is_mock must be boolean")
    _require(isinstance(summary.get("collector_success"), bool),
             "trajectory collector_success must be boolean")
    prompts_fp = _nonempty_text(
        summary.get("prompts_fp"), "trajectory prompts_fp")
    generation_fp = _nonempty_text(
        summary.get("prompt_generation_fp"),
        "trajectory prompt_generation_fp")
    _require(summary.get("policy_builder") == POLICY_BUILDER_ID,
             "trajectory uses a non-canonical policy builder")
    _require(summary.get("message_topology") == MESSAGE_TOPOLOGY,
             "trajectory message topology is not stateless")
    provenance = summary.get("policy_provenance")
    _require(isinstance(provenance, dict),
             "trajectory policy_provenance must be an object")
    decode = provenance.get("decode")
    _require(isinstance(decode, dict),
             "trajectory policy_provenance.decode must be an object")
    max_history = _strict_int(
        decode.get("max_history"), "policy decode max_history")
    max_axtree_chars = _strict_int(
        config.MAX_AXTREE_CHARS_POLICY,
        "configured max_axtree_chars_policy", minimum=1)
    capture = summary.get("episode_source_capture")
    _require(isinstance(capture, dict),
             "trajectory episode_source_capture must be an object")
    _require(capture.get("schema_version") == RAW_SOURCE_SCHEMA_VERSION,
             "unsupported raw episode source schema")
    _require(capture.get("counts_as_formal_supervision") is False,
             "raw episode source must explicitly be non-formal")

    raw_path, raw_relative = _safe_artifact_path(
        root, summary.get("raw_artifact"), requested_id)
    _require(output_path != raw_path,
             "review output cannot replace the raw trajectory")
    rows, raw_bytes = _load_raw_rows(raw_path)
    action_count = len(rows) - 1
    _require(action_count > 0,
             "raw trajectory contains no policy action steps")
    _require(_strict_int(summary.get("n_steps"), "summary n_steps") ==
             action_count, "summary n_steps does not match raw trajectory")
    _require(_strict_int(capture.get("n_action_steps"),
                         "source n_action_steps") == action_count,
             "episode source action count mismatch")
    _require(_strict_int(capture.get("n_exact_policy_calls"),
                         "source n_exact_policy_calls") == action_count,
             "episode source exact-call count mismatch")

    expected_raw_fields = {field.name for field in fields(StepRecord)}
    _require(set(rows[0]) == expected_raw_fields,
             "initial raw StepRecord schema mismatch")
    _require(rows[0].get("step_id") == 0 and rows[0].get("action") is None,
             "raw trajectory must start with action-free step 0")
    _require(rows[0].get("replay_prefix") == [],
             "initial raw replay_prefix must be empty")
    _require(rows[0].get("policy_call_captured") is False and
             rows[0].get("policy_input_messages") == [] and
             rows[0].get("assistant_completion") == "",
             "initial observation must not contain a fabricated policy call")

    for row in rows:
        step_id = row.get("step_id")
        _require(not isinstance(step_id, bool) and isinstance(step_id, int),
                 "raw step_id must be an integer")
        _require(row.get("run_id") == run_id,
                 f"raw step {step_id} run_id mismatch")
        _require(row.get("trajectory_id") == requested_id,
                 f"raw step {step_id} trajectory_id mismatch")
        _require(row.get("task_id") == task_id,
                 f"raw step {step_id} task_id mismatch")
        _require(row.get("code_version") == code_version,
                 f"raw step {step_id} code_version mismatch")
    _require([row["step_id"] for row in rows] == list(range(len(rows))),
             "raw step_ids must be contiguous from zero")

    try:
        first_pre = EpisodeObservation.from_dict(rows[1]["pre_observation"])
    except (KeyError, EpisodeTraceError) as exc:
        raise EpisodeSourcePreflightError(
            f"step 1 invalid captured pre-observation: {exc}") from exc
    _require(rows[0].get("url_after") == first_pre.url,
             "initial observation URL does not match first pre-observation")
    _require(rows[0].get("obs_after_axtree") ==
             prune_axtree_txt(first_pre.axtree_txt),
             "initial AXTree does not match first pre-observation")
    _require(rows[0].get("backend_after") == first_pre.persistent_signal,
             "initial persistent signal does not match first pre-observation")

    system_prompt: str | None = None
    observed_history: list[dict[str, Any]] = []
    prior_post: EpisodeObservation | None = None
    executed_actions: list[str] = []
    review_steps: list[dict[str, Any]] = []
    for row in rows[1:]:
        step_id = row["step_id"]
        raw_action = _nonempty_text(row.get("action"),
                                    f"step {step_id} action")
        _require(row.get("policy_call_captured") is True,
                 f"step {step_id} lacks an exact policy call")
        _require(isinstance(row.get("policy_guarded"), bool),
                 f"step {step_id} policy_guarded must be boolean")
        messages = row.get("policy_input_messages")
        _require(isinstance(messages, list) and
                 [message.get("role") if isinstance(message, dict) else None
                  for message in messages] == ["system", "user"] and
                 all(set(message) == {"role", "content"}
                     and isinstance(message.get("content"), str)
                     for message in messages),
                 f"step {step_id} policy input must be exact system + user")
        input_hash = _nonempty_text(
            row.get("policy_input_messages_sha256"),
            f"step {step_id} policy input hash")
        _require(input_hash == canonical_json_sha256(messages),
                 f"step {step_id} policy input hash mismatch")
        completion = _nonempty_text(
            row.get("assistant_completion"),
            f"step {step_id} assistant completion")
        completion_hash = _nonempty_text(
            row.get("assistant_completion_sha256"),
            f"step {step_id} assistant completion hash")
        _require(completion_hash ==
                 hashlib.sha256(completion.encode("utf-8")).hexdigest(),
                 f"step {step_id} assistant completion hash mismatch")
        _require(actions_match(_completion_action(completion), raw_action),
                 f"step {step_id} completion action does not match executed action")

        try:
            pre = EpisodeObservation.from_dict(row["pre_observation"])
            post = EpisodeObservation.from_dict(row["post_observation"])
        except (KeyError, EpisodeTraceError) as exc:
            raise EpisodeSourcePreflightError(
                f"step {step_id} invalid captured observation: {exc}") from exc
        _require(pre.url == row.get("url_before"),
                 f"step {step_id} pre-observation URL mismatch")
        _require(post.url == row.get("url_after"),
                 f"step {step_id} post-observation URL mismatch")
        _require(row.get("obs_after_axtree") ==
                 prune_axtree_txt(post.axtree_txt),
                 f"step {step_id} post AXTree mismatch")
        _require(row.get("backend_after") == post.persistent_signal,
                 f"step {step_id} post persistent signal mismatch")
        if prior_post is not None:
            _require(pre == prior_post,
                     f"step {step_id} pre-observation is not the exact previous "
                     "post-observation")

        entry, signal_mode = _verify_history_delta(
            raw_action, pre, post, row.get("observed_history_entry"), step_id)
        if system_prompt is None:
            system_prompt = messages[0]["content"]
            _nonempty_text(system_prompt, "policy system prompt")
        _require(messages[0]["content"] == system_prompt,
                 f"step {step_id} policy system prompt changed within trajectory")
        expected_messages = prompts.build_policy_messages(
            goal,
            pre.axtree_txt,
            observed_history,
            system_prompt=system_prompt,
            max_history=max_history,
            max_axtree_chars=max_axtree_chars,
        )
        _require(messages == expected_messages,
                 f"step {step_id} is not byte-equivalent to canonical policy builder")
        parsed = parse_action(raw_action)
        _require(parsed is not None,
                 f"step {step_id} executed action is not a literal action")
        if parsed.bid is not None:
            current_observation = prompts.parse_observation_message(
                messages[1]["content"])
            _require(bid_is_visible(current_observation, parsed.bid),
                     f"step {step_id} action bid [{parsed.bid}] is absent from "
                     "the exact policy input")
        executed_actions.append(raw_action)
        _require(row.get("replay_prefix") == executed_actions,
                 f"step {step_id} replay_prefix mismatch")

        review_steps.append({
            "source_step_id": step_id,
            "raw_action": raw_action,
            "policy_input_messages": messages,
            "policy_input_messages_sha256": input_hash,
            "assistant_completion": completion,
            "assistant_completion_sha256": completion_hash,
            "pre_observation": pre.to_dict(),
            "post_observation": post.to_dict(),
            "observed_history_entry": entry,
            "history_signal_mode": signal_mode,
            "policy_guarded": bool(row.get("policy_guarded")),
            "source_validated": True,
            "turn_type": None,
            "turn_type_status": "UNAUTHORED",
            "normative_truth": None,
            "truth_status": "UNAUTHORED",
            "supervised_sample_id": None,
            "probe_point_id": None,
            "evaluation_case_id": None,
            "missing_join_fields": list(MISSING_FORMAL_JOINS),
            "counts_as_formal": False,
        })
        observed_history.append(entry)
        prior_post = post

    base_review: dict[str, Any] = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "counts_as_formal": False,
        "counts_as_formal_supervision": False,
        "canonical_import_permitted": False,
        "source": {
            "run_manifest": str(manifest_path.relative_to(root)),
            "run_manifest_sha256": _sha256_bytes(manifest_bytes),
            "raw_artifact": raw_relative,
            "raw_artifact_sha256": _sha256_bytes(raw_bytes),
            "run_id": run_id,
            "trajectory_id": requested_id,
            "logical_trajectory_id": logical_id,
            "task_id": task_id,
            "seed": seed,
            "goal": goal,
            "code_version": code_version,
            "prompts_fp": prompts_fp,
            "prompt_generation_fp": generation_fp,
            "policy_builder": POLICY_BUILDER_ID,
            "message_topology": MESSAGE_TOPOLOGY,
            "max_history_steps": max_history,
            "max_axtree_chars": max_axtree_chars,
            "environment_origin": environment_origin,
            "environment_family": environment_family,
            "is_mock": summary.get("is_mock"),
            "collector_success": summary.get("collector_success"),
        },
        "source_validation": {
            "status": "PASSED",
            "n_raw_records": len(rows),
            "n_action_steps": action_count,
            "n_exact_policy_calls": action_count,
            "pre_post_continuity": True,
            "history_delta_exact": True,
            "policy_builder_byte_equivalent": True,
        },
        "steps": review_steps,
        "missing_join_counts": {
            field: action_count for field in MISSING_FORMAL_JOINS
        },
        "authoring_requirements": [
            "A reviewer must explicitly assign each turn_type.",
            "State-changing turns require a unique point-level probe join.",
            "Formal supervision requires an exact supervised_sample_id join.",
            "Decision truth requires an explicitly authored evaluation_case_id.",
            "This review sheet cannot be imported as a canonical episode.",
        ],
    }
    review = {
        **base_review,
        "review_sheet_id": _review_identity(base_review),
    }
    _exclusive_json(output_path, review)
    return {
        "review_sheet_id": review["review_sheet_id"],
        "output": str(output_path),
        "run_id": run_id,
        "trajectory_id": requested_id,
        "n_action_steps": action_count,
        "n_exact_policy_calls": action_count,
        "counts_as_formal": False,
        "canonical_import_permitted": False,
        "missing_join_counts": review["missing_join_counts"],
    }
