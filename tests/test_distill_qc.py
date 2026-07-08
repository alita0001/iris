"""Teacher distillation QC: pinned conclusions survive, contradictions rejected."""
from revact.train.distill import distill_sample, qc_check

GOOD = ("<observation> On the checkout page with an order summary.\n"
        "<reasoning> Committing here creates a record that this account cannot take back, "
        "so asking the user first is the right call.\n"
        "<prediction> The purchase would be finalized and an order record created.\n"
        "<rev_check> The order pages expose no cancel control to this account, "
        "so there is no in-site path back once committed.")
BAD = ("<observation> On the checkout page.\n"
       "<reasoning> This can be undone later, so it is fine.\n"
       "<prediction> The order would be placed.\n"
       "<rev_check> A cancel button surely exists somewhere.")


def _row():
    return {"sample_id": "s1",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "<goal>\nPlease place the order.\n\n<observation>\n[12] button 'Place Order'\n"},
                {"role": "assistant", "content":
                    "<think>\n<observation> old obs\n<reasoning> old reason\n"
                    "<prediction> Clicking 'Place Order' would submit the order.\n"
                    "<rev_check> old check\n"
                    "<reversibility> IRREVERSIBLE\n<undo> none available\n"
                    "<decision> CONFIRM risk=0.7\n</think>\n"
                    "<answer> send_msg_to_user('Confirm?')"},
            ],
            "meta": {"action_type": "place_order", "reversibility": "IRREVERSIBLE",
                     "decision": "CONFIRM"}}


def test_qc_rejects_contradiction():
    assert qc_check(BAD, "IRREVERSIBLE", "CONFIRM") is not None
    assert qc_check(GOOD, "IRREVERSIBLE", "CONFIRM") is None
    assert qc_check("<observation> x\n<reasoning> y", "REVERSIBLE", "EXECUTE") is not None  # missing tags


def test_qc_requires_rev_check_tag():
    three_lines = GOOD.rsplit("\n<rev_check>", 1)[0]
    assert qc_check(three_lines, "IRREVERSIBLE", "CONFIRM") == "missing tag <rev_check>"


def test_distill_keeps_pinned_labels():
    row = _row()
    rep = distill_sample(row, lambda prompt: GOOD)
    assert rep["ok"] and rep["attempts"] == 1
    asst = row["messages"][2]["content"]
    assert "<reversibility> IRREVERSIBLE" in asst       # pinned, untouched
    assert "<undo> none available" in asst              # pinned, untouched
    assert "<decision> CONFIRM risk=0.7" in asst
    assert "asking the user first" in asst              # prose replaced
    assert "no in-site path back" in asst               # rev_check prose replaced


def test_distill_prompt_carries_undo_fact():
    seen = {}

    def fake(prompt):
        seen["prompt"] = prompt
        return GOOD

    assert distill_sample(_row(), fake)["ok"]
    assert "none available" in seen["prompt"]           # measured undo path pinned


def test_distill_retries_then_succeeds():
    row = _row()
    seq = iter([BAD, GOOD])
    rep = distill_sample(row, lambda prompt: next(seq))
    assert rep["ok"] and rep["attempts"] == 2


def test_distill_drops_persistent_contradiction():
    row = _row()
    rep = distill_sample(row, lambda prompt: BAD, max_retries=2)
    assert not rep["ok"]
    assert "contradicts" in rep["last_reason"]
