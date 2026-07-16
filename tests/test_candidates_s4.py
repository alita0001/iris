"""S4 candidate legality, diversity, and source-lineage gates."""
from __future__ import annotations

import pytest

from revact.data.candidates import (
    CATEGORY_CONSTRAINT_TRIGGER,
    CATEGORY_DECOY,
    CATEGORY_EXPERT,
    CATEGORY_GOAL_VIOLATING,
    CATEGORY_ORDINARY,
    CATEGORY_SAFE_ALTERNATIVE,
    CATEGORY_POLICY_ERROR,
    CATEGORY_UNCERTAIN,
    SOURCE_A11Y,
    SOURCE_LLM,
    CandidateValidationError,
    _rule_category,
    audit_dpo_negative_sources,
    build_a11y_candidate_set,
    candidate_distribution,
    candidate_from_proposal,
    canonical_click_for_element,
    interactive_elements,
    validate_candidate_set,
)


AXTREE = """RootWebArea 'Fixture'
  [10] button 'Submit order'
  [11] button 'Review cart'
  [12] link 'Delete draft'
  [13] button 'Undo vote'
  [14] checkbox 'Subscribe'
  [15] link 'Continue'
"""


def _candidate(bid: str, category: str, *, source: str = SOURCE_A11Y):
    element = next((row for row in interactive_elements(AXTREE)
                    if row["bid"] == bid),
                   {"bid": bid, "role": "button", "name": "missing"})
    return candidate_from_proposal(
        {"bid": bid, "canonical_action": canonical_click_for_element(element),
         "category": category},
        state_id="state-1",
        axtree_txt=AXTREE,
        source=source,
        proposer_model="fixture-enumerator",
        proposer_version="1",
    )


def test_s4_accepts_four_to_six_legal_diverse_candidates():
    rows = [
        _candidate("10", CATEGORY_EXPERT),
        _candidate("11", CATEGORY_ORDINARY),
        _candidate("12", CATEGORY_CONSTRAINT_TRIGGER),
        _candidate("13", CATEGORY_ORDINARY),
        _candidate("14", CATEGORY_ORDINARY),
        _candidate("15", CATEGORY_UNCERTAIN),
    ]
    validate_candidate_set(rows, AXTREE)
    distribution = candidate_distribution(rows)
    assert distribution["category"] == {
        CATEGORY_CONSTRAINT_TRIGGER: 1,
        CATEGORY_EXPERT: 1,
        CATEGORY_ORDINARY: 3,
        CATEGORY_UNCERTAIN: 1,
    }
    assert distribution["source"] == {SOURCE_A11Y: 6}


@pytest.mark.parametrize(("name", "expected"), [
    ("Bunch of 3 Bananas", CATEGORY_ORDINARY),
    ("Stored Payment Methods", CATEGORY_ORDINARY),
    ("Ban user", CATEGORY_CONSTRAINT_TRIGGER),
    ("Pay now", CATEGORY_CONSTRAINT_TRIGGER),
    ("Place Order", CATEGORY_CONSTRAINT_TRIGGER),
])
def test_constraint_trigger_matching_respects_token_and_phrase_boundaries(
        name, expected):
    element = {"bid": "2", "role": "button", "name": name}
    assert _rule_category(element, expert_bid="1") == expected


def test_static_rule_never_fabricates_evidence_dependent_categories():
    forbidden = {
        CATEGORY_DECOY, CATEGORY_GOAL_VIOLATING, CATEGORY_POLICY_ERROR,
        CATEGORY_SAFE_ALTERNATIVE,
    }
    for index, name in enumerate((
            "Cancel", "Undo vote", "Restore draft", "Remove from cart"), 1):
        element = {"bid": str(index), "role": "button", "name": name}
        assert _rule_category(element, expert_bid="99") not in forbidden


def test_s4_rejects_absent_bid_and_snapshot_hash_mismatch():
    with pytest.raises(CandidateValidationError, match="not legal"):
        _candidate("404", CATEGORY_ORDINARY)
    row = _candidate("10", CATEGORY_EXPERT)
    with pytest.raises(CandidateValidationError, match="snapshot_hash"):
        row.validate(AXTREE + "  [99] button 'new'\n")


def test_llm_can_propose_but_cannot_supply_a_label():
    with pytest.raises(CandidateValidationError, match="forbidden labels"):
        candidate_from_proposal(
            {"bid": "10", "canonical_action": "click('10')",
             "category": CATEGORY_EXPERT, "recovery_status": "RECOVERED"},
            state_id="state-1", axtree_txt=AXTREE, source=SOURCE_LLM,
            proposer_model="offline-fixture", proposer_version="1")


def test_candidate_set_keeps_honest_low_diversity_but_rejects_wrong_size():
    rows = [_candidate(str(bid), CATEGORY_ORDINARY) for bid in range(10, 14)]
    validate_candidate_set(rows, AXTREE)
    with pytest.raises(CandidateValidationError, match="size"):
        validate_candidate_set(rows[:3], AXTREE)


def test_a11y_builder_filters_structure_and_builds_diverse_legal_set():
    tree = ("RootWebArea 'Checkout'\n"
            "[1] banner ''\n"
            "[10] button 'Place Order'\n"
            "[11] link 'Cancel'\n"
            "[12] button 'Continue'\n"
            "[13] link 'Help'\n"
            "[14] checkbox 'Gift receipt'\n"
            "[15] StaticText 'not a control'\n"
            "[16] button 'Delete draft'\n")
    rows = build_a11y_candidate_set(
        state_id="checkout-1", axtree_txt=tree, expert_bid="10")
    assert 4 <= len(rows) <= 6
    assert rows[0].category == CATEGORY_EXPERT
    assert {r.bid for r in rows}.isdisjoint({"1", "15"})
    assert all(r.legal_at_snapshot for r in rows)
    assert rows[0].source == "expert"
    assert all(r.source == SOURCE_A11Y for r in rows[1:])
    assert len({r.category for r in rows}) >= 4


def test_a11y_builder_prefers_task_local_siblings_and_caps_global_chrome():
    tree = ("RootWebArea 'Product'\n"
            "  [1] link 'My Account'\n"
            "  [2] link 'My Wish List'\n"
            "  [3] link 'Sign Out'\n"
            "  [4] link 'Skip to Content'\n"
            "  [5] link 'store logo'\n"
            "  [6] menuitem 'Electronics'\n"
            "  [20] main 'Product'\n"
            "    [21] spinbutton 'Qty'\n"
            "    [22] button 'Add to Cart'\n"
            "    [23] link 'Add to Wish List'\n"
            "    [24] link 'Add to Compare'\n"
            "    [25] tab 'Details'\n"
            "    [26] button 'Next'\n")
    rows = build_a11y_candidate_set(
        state_id="product-1", axtree_txt=tree, expert_bid="22")
    bids = {row.bid for row in rows}
    assert {"22", "23", "24", "25"} <= bids
    assert len(bids & {"1", "2", "3", "4", "5", "6"}) <= 1
    next_row = next(row for row in rows if row.bid == "26")
    assert next_row.category == CATEGORY_ORDINARY


def test_dpo_negative_source_gate_and_auditable_distribution():
    report = audit_dpo_negative_sources([
        {"negative_source": "legal_candidate"},
        {"negative_source": "on_policy"},
        {"negative_source": "synthetic_flip"},
        {"negative_source": "synthetic_flip"},
    ])
    assert report["legal_or_on_policy_share"] == 0.5
    assert report["source_counts"] == {
        "legal_candidate": 1, "on_policy": 1, "synthetic_flip": 2}
    with pytest.raises(CandidateValidationError, match="below"):
        audit_dpo_negative_sources([
            {"negative_source": "legal_candidate"},
            {"negative_source": "synthetic_flip"},
            {"negative_source": "synthetic_flip"},
        ])
