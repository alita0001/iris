"""Probe protocol on the mock env: labels, undo tracing, gating, containment."""
import json

import pytest

from revact.envs.harness import RevActEnv
from revact.envs.mock_env import MockShoppingEnv
from revact.grounding import (ProbeContext, get_probe, load_reversibility,
                              run_probe)
import revact.grounding.probes  # noqa: F401  (registers probes)

BASE = "http://mock-shopping"
PRODUCT = "http://mock-shopping/product/20"


@pytest.fixture()
def ctx():
    renv = RevActEnv(MockShoppingEnv(goal="probe test"), task_id="mock")
    renv.reset(seed=0)
    yield ProbeContext(renv=renv, base=BASE, product_url=PRODUCT)
    renv.close()


def test_add_to_cart_reversible_with_undo_trace(ctx):
    r = run_probe("shopping.add_to_cart", ctx)
    assert r.label == "REVERSIBLE"
    assert r.evidence["undo_steps"] == 1
    assert r.evidence["undo_actions"]
    assert r.evidence["residual_diff"]["count_delta"] == 0
    assert r.probe_id and r.timestamp and r.probe_name == "shopping.add_to_cart"


def test_wishlist_and_newsletter_reversible(ctx):
    assert run_probe("shopping.wishlist_add", ctx).label == "REVERSIBLE"
    assert run_probe("shopping.newsletter_subscribe", ctx).label == "REVERSIBLE"


def test_place_order_dry_run_without_commit(ctx):
    r = run_probe("shopping.place_order", ctx)
    assert r.label == "UNKNOWN"
    assert "dry-run" in r.evidence["reason"]
    assert r.commit_mode is False


def test_commit_forced_dry_without_env_gate(ctx, monkeypatch):
    monkeypatch.delenv("REVACT_ALLOW_DESTRUCTIVE", raising=False)
    ctx.commit = True
    r = run_probe("shopping.place_order", ctx)
    assert r.label == "UNKNOWN"
    assert "forced dry-run" in r.evidence.get("gate_note", "")


def test_commit_with_env_gate_grounds_irreversible(ctx, monkeypatch):
    monkeypatch.setenv("REVACT_ALLOW_DESTRUCTIVE", "1")
    ctx.commit = True
    r = run_probe("shopping.place_order", ctx)
    assert r.label == "IRREVERSIBLE"
    assert r.commit_mode is True
    assert r.evidence["new_orders"] == ["000001001"]


def test_probe_error_is_contained(ctx):
    spec = get_probe("shopping.address_add")  # mock has no Save Address form
    r = run_probe(spec.name, ctx)
    assert r.label == "UNKNOWN"          # graceful, not raised


def test_loader_is_dry_run_safe(tmp_path):
    p = tmp_path / "rev.jsonl"
    rows = [
        {"action_type": "place_order", "label": "IRREVERSIBLE"},
        {"action_type": "place_order", "label": "UNKNOWN",
         "evidence": {"reason": "dry-run"}},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    assert load_reversibility(p)["place_order"] == "IRREVERSIBLE"
