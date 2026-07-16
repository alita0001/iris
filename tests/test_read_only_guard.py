from revact.policies import ReadOnlyPolicyGuard, read_only_action_error
from revact.train.validators import (actions_match, answer_text,
                                     iris_tag_errors, parse_action)


VIEW = {
    "url": "http://shop.local/customer/account/",
    "axtree_txt": (
        "RootWebArea 'Account'\n"
        "  [1] link 'My Orders'\n"
        "  [2] link 'Add to Wish List'\n"
        "  [3] button 'Search'\n"
        "  [4] searchbox 'Search products'\n"
        "  [5] button 'Save address'\n"
        "  [6] link 'Reorder'\n"
    ),
}


def test_read_only_guard_allows_navigation_and_search_only():
    for action in (
            "click('1')", "click('3')", "fill('4', 'mouth guard')",
            "goto('http://shop.local/sales/order/history/')", "go_back()"):
        assert read_only_action_error(action, VIEW) is None


def test_read_only_guard_rejects_mutations_and_ambiguous_primitives():
    expected = {
        "click('2')": "mutation_like_control",
        "click('5')": "mutation_like_control",
        "click('6')": "mutation_like_control",
        "goto('http://other.local/')": "cross_origin_navigation",
        "select_option('4', 'x')": "primitive_not_read_only:select_option",
        "press('ENTER')": "primitive_not_read_only:press",
    }
    assert {action: read_only_action_error(action, VIEW)
            for action in expected} == expected


def test_wrapper_turns_rejected_action_into_terminal_non_mutating_action():
    class Policy:
        last_raw_response = "I will save it\nclick('5')"
        last_finish_reason = "stop"

        def reset(self):
            pass

        def act(self, *_args, **_kwargs):
            return "click('5')"

    guard = ReadOnlyPolicyGuard(Policy())
    action = guard.act(VIEW, goal="read", history=[])
    assert action == "report_infeasible('read_only_guard:mutation_like_control')"
    assert guard.last_finish_reason == "read_only_guard"
    assert guard.last_proposed_action == "click('5')"
    assert guard.last_proposed_completion == "I will save it\nclick('5')"
    assert guard.last_executed_action == action
    assert guard.last_executed_completion == guard.last_raw_response
    assert iris_tag_errors(guard.last_executed_completion) == []
    assert actions_match(
        parse_action(answer_text(guard.last_executed_completion)), action)
    assert guard.guard_rejections == [{
        "action": "click('5')",
        "proposed_action": "click('5')",
        "proposed_completion": "I will save it\nclick('5')",
        "reason": "mutation_like_control",
        "url": VIEW["url"],
        "executed_action": action,
        "executed_completion": guard.last_executed_completion,
    }]
    provenance = guard.execution_provenance()
    assert provenance["last_proposed_action"] == "click('5')"
    assert provenance["last_executed_action"] == action
    assert len(provenance["last_proposed_completion_sha256"]) == 64
    assert len(provenance["last_executed_completion_sha256"]) == 64


def test_wrapper_preserves_allowed_proposal_as_executed_completion():
    class Policy:
        last_raw_response = "<answer> click('1')"
        last_finish_reason = "stop"

        def reset(self):
            pass

        def act(self, *_args, **_kwargs):
            return "click('1')"

    guard = ReadOnlyPolicyGuard(Policy())
    action = guard.act(VIEW, goal="read", history=[])
    assert action == "click('1')"
    assert guard.guard_rejections == []
    assert guard.last_proposed_action == guard.last_executed_action == action
    assert guard.last_proposed_completion == guard.last_executed_completion == \
        guard.last_raw_response == "<answer> click('1')"
