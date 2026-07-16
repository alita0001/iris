"""Raw collection episode source preflight stays review-only and fail-closed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from revact import config, prompts
from revact.cli import build_parser
from revact.data.collect import run_collection
from revact.data.episode_authoring import (
    REVIEW_SCHEMA_VERSION,
    EpisodeSourcePreflightError,
    preflight_episode_source,
)
from revact.data.episode_traces import canonical_json_sha256
from revact.envs.mock_env import MockShoppingEnv


def _full_iris(action: str) -> str:
    return (
        "<think>\n"
        "<observation>the selected control is visible</observation>\n"
        "<reasoning>take the next fixture step</reasoning>\n"
        "<prediction>the browser observation may update</prediction>\n"
        "<rev_check>no recovery label is authored here</rev_check>\n"
        "<reversibility>UNKNOWN</reversibility>\n"
        "<undo>VERIFY</undo>\n"
        "<decision>EXECUTE risk=0.1</decision>\n"
        "</think>\n"
        f"<answer> {action}"
    )


class _ExactStatelessPolicy:
    provider = "fixture"
    model = "fixture-policy"
    base_url = "http://fixture.invalid/v1"
    api_key_env = "FIXTURE_API_KEY"
    temperature = 0.0
    top_p = 1.0
    max_tokens = 512
    max_history = config.POLICY_HISTORY_STEPS
    system_prompt = "fixture episode source system"

    def __init__(self):
        self._index = 0
        self.last_request_messages = []
        self.last_raw_response = ""
        self.last_finish_reason = ""

    def reset(self):
        self._index = 0
        self.last_request_messages = []
        self.last_raw_response = ""
        self.last_finish_reason = ""

    def act(self, obs_view, goal="", history=None):
        plan = (
            "click('11')",
            "click('20')",
            "send_msg_to_user('fixture complete')",
        )
        if self._index >= len(plan):
            return None
        action = plan[self._index]
        self._index += 1
        self.last_request_messages = prompts.build_policy_messages(
            goal,
            obs_view["axtree_txt"],
            history,
            system_prompt=self.system_prompt,
            max_history=self.max_history,
            max_axtree_chars=config.MAX_AXTREE_CHARS_POLICY,
        )
        self.last_raw_response = _full_iris(action)
        self.last_finish_reason = "stop"
        return action


@pytest.fixture
def raw_episode_source(tmp_path):
    result = run_collection(
        lambda _task_id: MockShoppingEnv(goal="Inspect one laptop."),
        lambda _task_id, _seed: _ExactStatelessPolicy(),
        ["mock.episode"],
        [0],
        out_dir=tmp_path,
        max_steps=3,
        code_version="worktree:episode-authoring-test",
    )
    summary = result["summaries"][0]
    return {
        "root": tmp_path,
        "manifest": Path(result["collection_manifest"]),
        "trajectory_id": summary["trajectory_id"],
        "raw": tmp_path / summary["raw_artifact"],
    }


def _rewrite_raw(path: Path, mutator) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    mutator(rows)
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows), encoding="utf-8")


def test_preflight_freezes_only_a_nonformal_review_sheet(raw_episode_source):
    source = raw_episode_source
    output = source["root"] / "authoring" / "episode-review.json"
    report = preflight_episode_source(
        source["manifest"], source["trajectory_id"], output, source["root"])

    review = json.loads(output.read_text(encoding="utf-8"))
    assert review["schema_version"] == REVIEW_SCHEMA_VERSION
    assert review["review_sheet_id"] == report["review_sheet_id"]
    assert review["counts_as_formal"] is False
    assert review["counts_as_formal_supervision"] is False
    assert review["canonical_import_permitted"] is False
    assert review["source_validation"] == {
        "status": "PASSED",
        "n_raw_records": 4,
        "n_action_steps": 3,
        "n_exact_policy_calls": 3,
        "pre_post_continuity": True,
        "history_delta_exact": True,
        "policy_builder_byte_equivalent": True,
    }
    assert review["missing_join_counts"] == {
        "supervised_sample_id": 3,
        "probe_point_id": 3,
        "evaluation_case_id": 3,
    }
    assert len(review["steps"]) == 3
    for step in review["steps"]:
        assert step["source_validated"] is True
        assert step["turn_type"] is None
        assert step["normative_truth"] is None
        assert step["supervised_sample_id"] is None
        assert step["probe_point_id"] is None
        assert step["evaluation_case_id"] is None
        assert step["counts_as_formal"] is False
    # Source preflight must not materialize or import canonical episode files.
    assert not (source["root"] / "raw" / "episodes").exists()


def test_review_output_is_exclusive_and_immutable(raw_episode_source):
    source = raw_episode_source
    output = source["root"] / "review.json"
    preflight_episode_source(
        source["manifest"], source["trajectory_id"], output, source["root"])
    before = output.read_bytes()
    with pytest.raises(EpisodeSourcePreflightError, match="already exists"):
        preflight_episode_source(
            source["manifest"], source["trajectory_id"], output,
            source["root"])
    assert output.read_bytes() == before


def test_preflight_rejects_completion_hash_tampering(raw_episode_source):
    source = raw_episode_source
    _rewrite_raw(source["raw"], lambda rows: rows[1].__setitem__(
        "assistant_completion", rows[1]["assistant_completion"] + " altered"))
    with pytest.raises(EpisodeSourcePreflightError,
                       match="assistant completion hash mismatch"):
        preflight_episode_source(
            source["manifest"], source["trajectory_id"],
            source["root"] / "review.json", source["root"])


def test_preflight_rejects_policy_input_hash_tampering(raw_episode_source):
    source = raw_episode_source
    _rewrite_raw(source["raw"], lambda rows: rows[1].__setitem__(
        "policy_input_messages_sha256", "0" * 64))
    with pytest.raises(EpisodeSourcePreflightError,
                       match="policy input hash mismatch"):
        preflight_episode_source(
            source["manifest"], source["trajectory_id"],
            source["root"] / "review.json", source["root"])


def test_preflight_rejects_completion_action_mismatch(raw_episode_source):
    source = raw_episode_source

    def mutate(rows):
        rows[1]["assistant_completion"] = _full_iris("click('99')")
        rows[1]["assistant_completion_sha256"] = hashlib.sha256(
            rows[1]["assistant_completion"].encode("utf-8")).hexdigest()

    _rewrite_raw(source["raw"], mutate)
    with pytest.raises(EpisodeSourcePreflightError,
                       match="completion action does not match"):
        preflight_episode_source(
            source["manifest"], source["trajectory_id"],
            source["root"] / "review.json", source["root"])


def test_preflight_rejects_stitched_pre_post_observations(raw_episode_source):
    source = raw_episode_source
    _rewrite_raw(source["raw"], lambda rows: rows[2]["pre_observation"].__setitem__(
        "title", "stitched unrelated observation"))
    with pytest.raises(EpisodeSourcePreflightError,
                       match="not the exact previous post-observation"):
        preflight_episode_source(
            source["manifest"], source["trajectory_id"],
            source["root"] / "review.json", source["root"])


def test_preflight_rejects_fabricated_history_delta(raw_episode_source):
    source = raw_episode_source
    _rewrite_raw(source["raw"], lambda rows: rows[1].__setitem__(
        "observed_history_entry",
        {"action": "click('11')", "flag": "state-change", "delta": "invented"}))
    with pytest.raises(EpisodeSourcePreflightError,
                       match="not derived from the captured pre/post"):
        preflight_episode_source(
            source["manifest"], source["trajectory_id"],
            source["root"] / "review.json", source["root"])


def test_preflight_rejects_noncanonical_builder_even_with_valid_hash(
        raw_episode_source):
    source = raw_episode_source

    def mutate(rows):
        rows[2]["policy_input_messages"][1]["content"] += "\nunsourced suffix"
        rows[2]["policy_input_messages_sha256"] = canonical_json_sha256(
            rows[2]["policy_input_messages"])

    _rewrite_raw(source["raw"], mutate)
    with pytest.raises(EpisodeSourcePreflightError,
                       match="not byte-equivalent to canonical policy builder"):
        preflight_episode_source(
            source["manifest"], source["trajectory_id"],
            source["root"] / "review.json", source["root"])


def test_preflight_rejects_manifest_path_traversal(raw_episode_source):
    source = raw_episode_source
    manifest = json.loads(source["manifest"].read_text(encoding="utf-8"))
    manifest["trajectories"][0]["raw_artifact"] = "../outside.jsonl"
    source["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8")
    with pytest.raises(EpisodeSourcePreflightError, match="without traversal"):
        preflight_episode_source(
            source["manifest"], source["trajectory_id"],
            source["root"] / "review.json", source["root"])


def test_preflight_requires_exact_manifest_location_and_physical_id(
        raw_episode_source):
    source = raw_episode_source
    copied = source["root"] / "copied-run.json"
    copied.write_bytes(source["manifest"].read_bytes())
    with pytest.raises(EpisodeSourcePreflightError,
                       match="directly under.*collection_runs"):
        preflight_episode_source(
            copied, source["trajectory_id"],
            source["root"] / "review.json", source["root"])
    with pytest.raises(EpisodeSourcePreflightError,
                       match="match exactly one summary"):
        preflight_episode_source(
            source["manifest"], "mock.episode_seed0__run_wrong",
            source["root"] / "review.json", source["root"])


def test_review_cannot_be_written_into_canonical_episode_directory(
        raw_episode_source):
    source = raw_episode_source
    output = source["root"] / "raw" / "episodes" / "review.json"
    with pytest.raises(EpisodeSourcePreflightError,
                       match="canonical episode directory"):
        preflight_episode_source(
            source["manifest"], source["trajectory_id"], output,
            source["root"])
    assert not output.exists()


def test_cli_help_states_review_only_boundary(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit) as stopped:
        parser.parse_args(["preflight-episode-source", "--help"])
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    assert "immutable" in help_text and "review-" in help_text
    assert "never a canonical episode" in help_text
    assert "counts_as_formal=false" in help_text


def test_cli_executes_the_same_review_only_preflight(
        raw_episode_source, capsys):
    source = raw_episode_source
    output = source["root"] / "cli-review.json"
    args = build_parser().parse_args([
        "preflight-episode-source",
        "--run-manifest", str(source["manifest"]),
        "--trajectory-id", source["trajectory_id"],
        "--output", str(output),
        "--data-root", str(source["root"]),
    ])
    assert args.fn(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["counts_as_formal"] is False
    assert report["canonical_import_permitted"] is False
    assert output.exists()
