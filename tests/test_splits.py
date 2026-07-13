"""Product-level split: no slug leakage between train and test."""
import json

from revact.data.splits import (audit_split_leakage, build_splits, parse_sid,
                                formal_validation_partition,
                                stable_group_is_val, state_group_of)


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
