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
    SOURCE_A11Y,
    SOURCE_LLM,
    CandidateValidationError,
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
        _candidate("11", CATEGORY_SAFE_ALTERNATIVE),
        _candidate("12", CATEGORY_CONSTRAINT_TRIGGER),
        _candidate("13", CATEGORY_DECOY),
        _candidate("14", CATEGORY_ORDINARY),
        _candidate("15", CATEGORY_GOAL_VIOLATING),
    ]
    validate_candidate_set(rows, AXTREE)
    distribution = candidate_distribution(rows)
    assert distribution["category"][CATEGORY_DECOY] == 1
    assert distribution["source"] == {SOURCE_A11Y: 6}


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
