"""Reddit (Postmill) probe protocol on the mock env: reversibility labels,
undo tracing, the [deleted]-tombstone PARTIALLY case, and destructive gating."""
import pytest

from revact.envs.harness import RevActEnv
from revact.envs.mock_env import MockRedditEnv
from revact.grounding import ProbeContext, run_probe
import revact.grounding.probes  # noqa: F401  (registers probes)

BASE = "http://mock-reddit"


@pytest.fixture()
def ctx():
    renv = RevActEnv(MockRedditEnv(goal="reddit probe test"), task_id="mock-reddit")
    renv.reset(seed=0)
    yield ProbeContext(renv=renv, base=BASE,
                       submission_url=f"{BASE}/f/AskReddit/1",
                       forum_url=f"{BASE}/f/books")
    renv.close()


def test_vote_is_reversible_with_undo_trace(ctx):
    r = run_probe("reddit.vote", ctx)
    assert r.label == "REVERSIBLE"
    assert r.evidence["baseline_score"] == 38
    assert r.evidence["baseline_voted"] is False
    assert r.evidence["after_action_score"] == 39   # upvote moved the score
    assert r.evidence["after_action_voted"] is True  # and set the active state
    assert r.evidence["after_undo_score"] == 38     # undo restored both
    assert r.evidence["after_undo_voted"] is False
    assert r.evidence["undo_steps"] == 1
    assert r.evidence["residual_diff"]["score_delta"] == 0
    assert r.evidence["residual_diff"]["vote_state_changed"] is False
    assert r.action_type == "reddit_vote"


def test_subscribe_is_reversible(ctx):
    r = run_probe("reddit.subscribe", ctx)
    assert r.label == "REVERSIBLE"
    assert r.evidence["residual_diff"]["subscribed_changed"] is False


def test_comment_submit_dry_run_without_commit(ctx):
    r = run_probe("reddit.comment_submit", ctx)
    assert r.label == "UNKNOWN"
    assert "dry-run" in r.evidence["reason"]
    assert r.commit_mode is False


def test_comment_submit_commit_forced_dry_without_env_gate(ctx, monkeypatch):
    monkeypatch.delenv("REVACT_ALLOW_DESTRUCTIVE", raising=False)
    ctx.commit = True
    r = run_probe("reddit.comment_submit", ctx)
    assert r.label == "UNKNOWN"
    assert "forced dry-run" in r.evidence.get("gate_note", "")


def test_comment_submit_partial_due_to_tombstone(ctx, monkeypatch):
    """The centerpiece cross-site anchor: an author CAN delete the comment
    (undo affordance exists, unlike shopping.place_order) but a [deleted]
    tombstone remains -> PARTIALLY_RECOVERABLE, not REVERSIBLE."""
    monkeypatch.setenv("REVACT_ALLOW_DESTRUCTIVE", "1")
    ctx.commit = True
    r = run_probe("reddit.comment_submit", ctx)
    assert r.label == "PARTIALLY_RECOVERABLE"
    assert r.commit_mode is True
    assert r.evidence["after_undo_count"] == 0          # comment removed
    assert r.evidence["tombstone_after_undo"] == 1      # residue remains
    assert r.evidence["residual_diff"]["tombstone_left"] is True


@pytest.mark.parametrize("name", [
    "reddit.comment_delete", "reddit.submission_delete",
    "reddit.edit_comment", "reddit.submission_create",
])
def test_destructive_skeletons_are_unknown(ctx, name):
    r = run_probe(name, ctx)
    assert r.label == "UNKNOWN"
    assert "skeleton" in r.evidence["reason"]
