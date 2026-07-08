"""Product-level split: no slug leakage between train and test."""
import json

from revact.data.splits import build_splits, parse_sid


def _sample(action, slug, variant):
    sid = f"{action}__{slug}__{variant}"
    return {"sample_id": sid,
            "messages": [{"role": "system", "content": "s"},
                         {"role": "user", "content": "u"},
                         {"role": "assistant", "content": "a"}],
            "meta": {"action_type": action, "decision": "EXECUTE",
                     "variant": variant}}


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
