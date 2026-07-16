"""Product-level split: no slug leakage between train and test."""
import json

import pytest

from revact.cli import main
from revact.data.splits import (audit_split_leakage, build_splits, parse_sid,
                                formal_dpo_supplement_manifest_path,
                                formal_validation_partition,
                                stable_group_is_val, state_group_of,
                                write_formal_dpo_supplement_manifest)


def _sample(action, slug, variant):
    sid = f"{action}__{slug}__{variant}"
    return {"sample_id": sid,
            "messages": [{"role": "system", "content": "s"},
                         {"role": "user", "content": "u"},
                         {"role": "assistant", "content": "a"}],
            "meta": {"action_type": action, "decision": "EXECUTE",
                     "variant": variant}}


def _formal_sample(index: int, variant: str = "request", **updates):
    row = _sample("add_to_cart", f"entity-{index}", variant)
    row["meta"].update({
        "formal_dataset": True,
        "state_id": f"state-{index}",
        "canonical_entity_id": f"entity-{index}",
        "goal_template": f"template-{index}",
        "page_template_id": f"page-template-{index}",
        "environment_instance": f"environment-{index}",
        **updates,
    })
    return row


def test_no_product_leakage(tmp_path):
    sft = tmp_path / "sft.jsonl"
    rows = [_sample("add_to_cart", f"prod{i}", v)
            for i in range(8) for v in ("constraint", "request")]
    sft.write_text("\n".join(json.dumps(r) for r in rows))
    dpo = tmp_path / "dpo.jsonl"
    dpo.write_text("\n".join(json.dumps(
        {"pair_id": r["sample_id"] + "__false_safe", "meta": r["meta"]}) for r in rows))

    rep = build_splits(sft, dpo, tmp_path / "out", holdout_frac=0.25)
    train = [json.loads(ln) for ln in open(tmp_path / "out" / "sft_train.jsonl")]
    test = [json.loads(ln) for ln in open(tmp_path / "out" / "sft_test.jsonl")]
    train_slugs = {parse_sid(r["sample_id"])[1] for r in train}
    test_slugs = {parse_sid(r["sample_id"])[1] for r in test}
    assert train_slugs.isdisjoint(test_slugs)
    assert rep["n_train"] + rep["n_test"] == len(rows)
    # DPO train must exclude held-out products too
    dpo_train = [json.loads(ln) for ln in open(tmp_path / "out" / "dpo_train.jsonl")]
    assert all(parse_sid(p["pair_id"])[1] in train_slugs for p in dpo_train)
    assert rep["leakage"]["state_group"]["n_overlap"] == 0
    assert rep["leakage"]["entity"]["n_overlap"] == 0


def test_variants_and_validation_are_group_atomic():
    rows = [_sample("add_to_cart", "same-product", v)
            for v in ("constraint", "request")]
    assert len({state_group_of(r) for r in rows}) == 1
    assert len({stable_group_is_val(r, 0.31) for r in rows}) == 1


def test_challenge_splits_and_multiturn_assets(tmp_path):
    sft = tmp_path / "revact_sft.jsonl"
    rows = []
    for i in range(8):
        r = _sample("add_to_cart" if i < 4 else "place_order",
                    f"prod{i}", "request")
        r["meta"].update({"site": "shopping" if i % 2 == 0 else "reddit",
                          "privilege": "user" if i % 2 == 0 else "admin",
                          "goal_template": f"template:{i % 2}"})
        rows.append(r)
    sft.write_text("\n".join(json.dumps(r) for r in rows))
    dpo = tmp_path / "revact_dpo.jsonl"
    dpo.write_text("\n".join(json.dumps({
        "pair_id": r["sample_id"] + "__wrong", "meta": r["meta"]}) for r in rows))

    mt = json.loads(json.dumps(rows[0]))
    mt["sample_id"] = "mt__t1__s3__add_to_cart__request"
    mt["meta"].update({"kind": "multiturn", "state_id": "t1_s3",
                       "canonical_entity_id": "entity-t1"})
    mt_sft = tmp_path / "revact_sft_multiturn.jsonl"
    mt_sft.write_text(json.dumps(mt))
    mt_dpo = tmp_path / "revact_dpo_multiturn.jsonl"
    mt_dpo.write_text(json.dumps({"pair_id": mt["sample_id"] + "__wrong",
                                  "meta": mt["meta"]}))

    rep = build_splits(sft, dpo, tmp_path / "out", holdout_frac=0.25,
                       multiturn_sft_path=mt_sft,
                       multiturn_dpo_path=mt_dpo)
    for axis in ("cross_action", "cross_privilege", "cross_template", "cross_site"):
        assert rep["challenges"][axis]["available"]
        key = {"cross_action": "action", "cross_privilege": "privilege",
               "cross_template": "template", "cross_site": "site"}[axis]
        assert rep["challenges"][axis]["leakage"][key]["n_overlap"] == 0
        assert rep["challenges"][axis]["leakage"]["state_group"]["n_overlap"] == 0
        assert rep["challenges"][axis]["leakage"]["entity"]["n_overlap"] == 0
    assert (tmp_path / "out" / "sft_train_multiturn.jsonl").exists()
    assert rep["multiturn"]["available"]


def test_leakage_audit_reports_shared_template_and_environment():
    a = _sample("add_to_cart", "a", "request")
    b = _sample("add_to_cart", "b", "request")
    for row in (a, b):
        row["meta"].update({"goal_template": "shared", "site": "shopping",
                            "environment_origin": "webarena"})
    audit = audit_split_leakage([a], [b])
    assert audit["state_group"]["n_overlap"] == 0
    assert audit["entity"]["n_overlap"] == 0
    assert audit["template"]["overlap"] == ["shared"]
    assert audit["environment"]["overlap"] == ["webarena"]
    assert audit["site"]["overlap"] == ["shopping"]


def test_cross_template_drops_sibling_variant_from_train(tmp_path):
    rows = []
    for slug in ("a", "b", "c"):
        for variant, template in (("constraint", "explicit:0"),
                                  ("request", "request:0")):
            row = _sample("add_to_cart", slug, variant)
            row["meta"]["goal_template"] = template
            rows.append(row)
    sft = tmp_path / "revact_sft.jsonl"
    sft.write_text("\n".join(json.dumps(r) for r in rows))
    dpo = tmp_path / "revact_dpo.jsonl"
    dpo.write_text("")
    rep = build_splits(sft, dpo, tmp_path / "out")
    challenge = rep["challenges"]["cross_template"]
    assert not challenge["available"]
    audit = challenge["leakage"]
    assert audit["template"]["n_overlap"] == 0
    assert audit["state_group"]["n_overlap"] == 0
    assert audit["entity"]["n_overlap"] == 0


def test_formal_split_uses_joint_leakage_components(tmp_path):
    rows = [_formal_sample(i, variant)
            for i in range(6) for variant in ("constraint", "request")]
    sft = tmp_path / "formal_sft.jsonl"
    dpo = tmp_path / "formal_dpo.jsonl"
    sft.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    dpo.write_text("")

    report = build_splits(
        sft, dpo, tmp_path / "formal" / "splits", holdout_frac=.25)
    assert report["formal"] is True and report["available"] is True
    assert report["n_train"] and report["n_test"]
    for axis in ("state_group", "canonical_entity", "goal_template",
                 "page_template", "environment"):
        assert report["leakage"][axis]["n_overlap"] == 0
    # Both variants of a state are one indivisible component.
    train_states = {state_group_of(row) for row in
                    [json.loads(line) for line in
                     (tmp_path / "formal" / "splits" / "sft_train.jsonl").open()]}
    test_states = {state_group_of(row) for row in
                   [json.loads(line) for line in
                    (tmp_path / "formal" / "splits" / "sft_test.jsonl").open()]}
    assert train_states.isdisjoint(test_states)


def test_formal_split_is_unavailable_instead_of_leaking_shared_environment(
        tmp_path):
    rows = [_formal_sample(i, environment_instance="shared-environment")
            for i in range(4)]
    sft = tmp_path / "formal_sft.jsonl"
    dpo = tmp_path / "formal_dpo.jsonl"
    out = tmp_path / "formal" / "splits"
    sft.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    dpo.write_text("")

    report = build_splits(sft, dpo, out)
    assert report["available"] is False
    assert report["n_train"] == report["n_test"] == 0
    assert "one connected component" in report["reason"]
    assert (out / "sft_train.jsonl").read_text() == ""
    assert (out / "sft_test.jsonl").read_text() == ""
    persisted = json.loads((out / "SPLIT_REPORT.json").read_text())
    assert persisted["available"] is False
    assert persisted["reason"] == report["reason"]
    assert persisted["schema_version"] == "iris.split_report.v2"
    graph = report["joint_graph"]
    assert persisted["joint_graph"] == graph
    assert graph["schema_version"] == \
        "iris.formal_joint_graph_diagnostics.v1"
    assert graph["partition_metadata_complete"] is True
    assert graph["n_rows"] == 4 and graph["n_components"] == 1
    component = graph["components"][0]
    assert component["n_rows"] == 4
    assert component["distributions"]["environment"] == {
        "shared-environment": 4}
    assert component["distributions"]["action"] == {"add_to_cart": 4}
    assert component["merge_evidence"]["shared_isolation_values"][
        "environment"] == [{"value": "shared-environment", "n_rows": 4}]


def test_formal_joint_graph_reports_cross_environment_template_bridges(
        tmp_path):
    rows = [
        _formal_sample(
            0, site="shopping", goal_template="shared-template",
            environment_instance="shop-env", page_template_id="shop-page"),
        _formal_sample(
            1, site="reddit", goal_template="shared-template",
            environment_instance="reddit-env", page_template_id="reddit-page"),
        _formal_sample(
            2, site="shopping_admin", goal_template="admin-template",
            environment_instance="admin-env", page_template_id="admin-page"),
        _formal_sample(
            3, site="third", goal_template="third-template",
            environment_instance="third-env", page_template_id="third-page"),
    ]
    sft = tmp_path / "formal_sft.jsonl"
    dpo = tmp_path / "formal_dpo.jsonl"
    out = tmp_path / "formal" / "splits"
    sft.write_text("".join(json.dumps(row) + "\n" for row in rows))
    dpo.write_text("")

    report = build_splits(sft, dpo, out, holdout_frac=.25)
    # The diagnostic code must not relax the graph: three genuine components
    # remain sufficient for strict train/dev/test, with zero isolation overlap.
    assert report["available"] is True
    assert report["n_train"] + report["n_dev"] + report["n_test"] == 4
    for axis in ("state_group", "canonical_entity", "goal_template",
                 "page_template", "environment"):
        assert report["leakage"][axis]["n_overlap"] == 0
        assert report["validation"]["leakage"][axis]["n_overlap"] == 0

    graph = report["joint_graph"]
    assert graph["n_rows"] == 4 and graph["n_components"] == 3
    bridges = graph["goal_template_bridges"]
    assert bridges["cross_environment_values"] == ["shared-template"]
    assert bridges["cross_site_values"] == ["shared-template"]
    assert bridges["details"] == [{
        "goal_template": "shared-template",
        "n_rows": 2,
        "environments": {"reddit-env": 1, "shop-env": 1},
        "sites": {"reddit": 1, "shopping": 1},
        "cross_environment": True,
        "cross_site": True,
        "component_ids": bridges["details"][0]["component_ids"],
    }]
    assert len(bridges["details"][0]["component_ids"]) == 1
    bridged_component = next(
        component for component in graph["components"]
        if component["distributions"]["goal_template"] == {
            "shared-template": 2})
    assert bridged_component["n_rows"] == 2
    assert bridged_component["distributions"]["environment"] == {
        "reddit-env": 1, "shop-env": 1}
    assert bridged_component["distributions"]["site"] == {
        "reddit": 1, "shopping": 1}
    assert bridged_component["merge_evidence"][
        "cross_environment_or_site_goal_templates"] == ["shared-template"]
    persisted = json.loads((out / "SPLIT_REPORT.json").read_text())
    assert persisted["joint_graph"] == graph


def test_cross_site_challenge_can_be_audited_when_base_split_is_blocked(
        tmp_path):
    # A shared goal-template connects the two environments, so the joint base
    # graph is intentionally unsplittable.  The independent site challenge may
    # still drop the leaking shopping state and retain another clean one.
    rows = [
        _formal_sample(
            0, site="shopping", goal_template="shared-template",
            environment_instance="shop-env", page_template_id="shop-page"),
        _formal_sample(
            1, site="shopping", goal_template="shopping-only-template",
            environment_instance="shop-env", page_template_id="shop-page"),
        _formal_sample(
            2, site="reddit", goal_template="shared-template",
            environment_instance="reddit-env", page_template_id="reddit-page"),
    ]
    sft = tmp_path / "formal_sft.jsonl"
    dpo = tmp_path / "formal_dpo.jsonl"
    sft.write_text("".join(json.dumps(row) + "\n" for row in rows))
    dpo.write_text("")

    report = build_splits(sft, dpo, tmp_path / "formal" / "splits")
    assert report["available"] is False
    challenge = report["challenges"]["cross_site"]
    assert challenge["available"] is True
    assert challenge["n_train"] == 1 and challenge["n_test"] == 1
    for axis in ("state_group", "canonical_entity", "goal_template",
                 "page_template", "environment"):
        assert challenge["leakage"][axis]["n_overlap"] == 0


def test_formal_split_requires_every_isolation_axis(tmp_path):
    rows = [_formal_sample(0), _formal_sample(1)]
    rows[1]["meta"].pop("goal_template")
    sft = tmp_path / "formal_sft.jsonl"
    dpo = tmp_path / "formal_dpo.jsonl"
    sft.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    dpo.write_text("")

    report = build_splits(sft, dpo, tmp_path / "formal" / "splits")
    assert report["available"] is False
    assert "template" in report["reason"]
    graph = report["joint_graph"]
    assert graph["partition_metadata_complete"] is False
    assert graph["missing_isolation_metadata"] == [{
        "sample_id": rows[1]["sample_id"],
        "missing_axes": ["goal_template"],
    }]


def test_formal_validation_never_splits_a_joint_component():
    independent = [_formal_sample(i) for i in range(4)]
    train, val, report = formal_validation_partition(independent, .25)
    assert report["available"] is True
    assert train and val
    for axis in ("state_group", "canonical_entity", "goal_template",
                 "page_template", "environment"):
        assert report["leakage"][axis]["n_overlap"] == 0

    connected = [_formal_sample(i, environment_instance="shared")
                 for i in range(4)]
    train, val, report = formal_validation_partition(connected, .25)
    assert report["available"] is False
    assert len(train) == len(connected) and val == []
    assert "one connected component" in report["reason"]


def test_formal_single_and_multiturn_share_one_component_assignment(tmp_path):
    single = [_formal_sample(i) for i in range(10)]
    multi = []
    for i in range(10):
        row = _formal_sample(i)
        row["sample_id"] = f"mt__trajectory-{i}__state-{i}__request"
        row["meta"]["kind"] = "multiturn"
        multi.append(row)
    sft = tmp_path / "formal_sft.jsonl"
    dpo = tmp_path / "formal_dpo.jsonl"
    mt_sft = tmp_path / "formal_sft_multiturn.jsonl"
    mt_dpo = tmp_path / "formal_dpo_multiturn.jsonl"
    sft.write_text("".join(json.dumps(row) + "\n" for row in single))
    dpo.write_text("")
    mt_sft.write_text("".join(json.dumps(row) + "\n" for row in multi))
    mt_dpo.write_text("")
    out = tmp_path / "formal" / "splits"
    report = build_splits(
        sft, dpo, out, holdout_frac=.25,
        multiturn_sft_path=mt_sft, multiturn_dpo_path=mt_dpo, formal=True)
    assert report["available"] is True

    def read(name):
        return [json.loads(line) for line in (out / name).open() if line.strip()]

    sides = {
        side: read(f"sft_{side}.jsonl") +
              read(f"sft_{side}_multiturn.jsonl")
        for side in ("train", "dev", "test")
    }
    for left, right in (("train", "dev"), ("train", "test"),
                        ("dev", "test")):
        audit = audit_split_leakage(sides[left], sides[right])
        for axis in ("state_group", "canonical_entity", "goal_template",
                     "page_template", "environment"):
            assert audit[axis]["n_overlap"] == 0
    membership = {}
    for side, rows in sides.items():
        for row in rows:
            entity = row["meta"]["canonical_entity_id"]
            assert membership.setdefault(entity, side) == side


def test_rebuild_with_empty_multiturn_source_clears_stale_shards(tmp_path):
    single = [_formal_sample(i) for i in range(8)]
    multi = [_formal_sample(20 + i) for i in range(4)]
    for i, row in enumerate(multi):
        row["sample_id"] = f"mt__trajectory-{i}__state-{i}__request"
    sft = tmp_path / "formal_sft.jsonl"
    dpo = tmp_path / "formal_dpo.jsonl"
    mt_sft = tmp_path / "formal_sft_multiturn.jsonl"
    mt_dpo = tmp_path / "formal_dpo_multiturn.jsonl"
    sft.write_text("".join(json.dumps(row) + "\n" for row in single))
    dpo.write_text("")
    mt_sft.write_text("".join(json.dumps(row) + "\n" for row in multi))
    mt_dpo.write_text("")
    out = tmp_path / "formal" / "splits"
    build_splits(sft, dpo, out, multiturn_sft_path=mt_sft,
                 multiturn_dpo_path=mt_dpo, formal=True)
    assert any(path.read_text() for path in out.glob("*_multiturn.jsonl"))

    mt_sft.write_text("")
    build_splits(sft, dpo, out, multiturn_sft_path=mt_sft,
                 multiturn_dpo_path=mt_dpo, formal=True)
    assert all(path.read_text() == "" for path in out.glob("*_multiturn.jsonl"))


def _supplement_pair(source: dict, pair_id: str, *, formal: bool = True) -> dict:
    return {
        "pair_id": pair_id,
        "prompt": source["messages"][:-1],
        "chosen": source["messages"][-1]["content"],
        "rejected": "rejected",
        "meta": {
            **source["meta"],
            "formal_dataset": formal,
            "source_sample_id": source["sample_id"],
            "negative_source": "on_policy",
        },
    }


def test_additional_dpo_supplement_requires_release_manifest_and_formal_join(
        tmp_path):
    rows = [_formal_sample(index) for index in range(8)]
    sft = tmp_path / "formal" / "formal_sft.jsonl"
    dpo = tmp_path / "formal" / "formal_dpo.jsonl"
    sft.parent.mkdir(parents=True)
    sft.write_text("".join(json.dumps(row) + "\n" for row in rows))
    dpo.write_text("")
    supplement = tmp_path / "formal" / "iris_dpo_on_policy_batch.jsonl"
    supplement.write_text(json.dumps(_supplement_pair(rows[0], "pair-supp-1")) + "\n")

    with pytest.raises(ValueError, match="body/manifest are both required"):
        build_splits(
            sft, dpo, tmp_path / "formal" / "splits",
            additional_dpo_paths=(supplement,), formal=True)

    manifest = write_formal_dpo_supplement_manifest(
        supplement, release_id="release-20260714")
    assert manifest == formal_dpo_supplement_manifest_path(supplement)
    report = build_splits(
        sft, dpo, tmp_path / "formal" / "splits",
        additional_dpo_paths=(supplement,), formal=True)
    assert report["n_additional_dpo"] == 1
    assert report["additional_dpo_releases"] == [{
        "path": str(supplement),
        "manifest_path": str(manifest),
        "release_id": "release-20260714",
        "body_sha256": json.loads(manifest.read_text())["body_sha256"],
        "n_rows": 1,
    }]

    supplement.write_text(json.dumps(
        _supplement_pair(rows[0], "pair-tampered")) + "\n")
    with pytest.raises(ValueError, match="does not exactly pin"):
        build_splits(
            sft, dpo, tmp_path / "formal" / "splits",
            additional_dpo_paths=(supplement,), formal=True)


def test_pin_dpo_supplement_cli_creates_immutable_release_manifest(
        tmp_path, capsys):
    source = _formal_sample(0)
    supplement = tmp_path / "iris_dpo_on_policy_batch.jsonl"
    supplement.write_text(json.dumps(
        _supplement_pair(source, "pair-cli-1")) + "\n")
    assert main([
        "pin-dpo-supplement",
        "--body", str(supplement),
        "--release-id", "release-cli-1",
    ]) == 0
    response = json.loads(capsys.readouterr().out)
    manifest = formal_dpo_supplement_manifest_path(supplement)
    assert response == {
        "body": str(supplement),
        "manifest": str(manifest),
        "release_id": "release-cli-1",
    }
    assert json.loads(manifest.read_text())["release_id"] == "release-cli-1"

    # Re-pinning an identical body/release is idempotent, never an overwrite.
    assert main([
        "pin-dpo-supplement",
        "--body", str(supplement),
        "--release-id", "release-cli-1",
    ]) == 0
    capsys.readouterr()
    assert main([
        "pin-dpo-supplement",
        "--body", str(supplement),
        "--release-id", "different-release",
    ]) == 1
    assert "refusing to overwrite" in capsys.readouterr().out


def test_additional_dpo_supplement_rejects_nonformal_unknown_and_duplicate_ids(
        tmp_path):
    rows = [_formal_sample(index) for index in range(8)]
    sft = tmp_path / "formal" / "formal_sft.jsonl"
    dpo = tmp_path / "formal" / "formal_dpo.jsonl"
    sft.parent.mkdir(parents=True)
    sft.write_text("".join(json.dumps(row) + "\n" for row in rows))

    nonformal = tmp_path / "formal" / "nonformal.jsonl"
    nonformal.write_text(json.dumps(
        _supplement_pair(rows[0], "nonformal-pair", formal=False)) + "\n")
    with pytest.raises(ValueError, match="non-formal row"):
        write_formal_dpo_supplement_manifest(nonformal, release_id="release-1")

    unknown_source = json.loads(json.dumps(rows[0]))
    unknown_source["sample_id"] = "unknown-source"
    unknown = tmp_path / "formal" / "unknown.jsonl"
    unknown.write_text(json.dumps(
        _supplement_pair(unknown_source, "unknown-pair")) + "\n")
    write_formal_dpo_supplement_manifest(unknown, release_id="release-1")
    dpo.write_text("")
    with pytest.raises(ValueError, match="unknown formal SFT sources"):
        build_splits(
            sft, dpo, tmp_path / "formal" / "splits",
            additional_dpo_paths=(unknown,), formal=True)

    duplicate = tmp_path / "formal" / "duplicate.jsonl"
    duplicate.write_text(json.dumps(
        _supplement_pair(rows[0], "duplicate-pair")) + "\n")
    write_formal_dpo_supplement_manifest(duplicate, release_id="release-1")
    dpo.write_text(json.dumps(_supplement_pair(rows[0], "duplicate-pair")) + "\n")
    with pytest.raises(ValueError, match="unique non-empty pair_id"):
        build_splits(
            sft, dpo, tmp_path / "formal" / "splits",
            additional_dpo_paths=(duplicate,), formal=True)

    with pytest.raises(ValueError, match="duplicate additional DPO supplement path"):
        build_splits(
            sft, dpo, tmp_path / "formal" / "splits",
            additional_dpo_paths=(duplicate, duplicate), formal=True)
