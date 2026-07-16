"""parse_action: reasoning-model-safe extraction (last line / action block)."""
from revact.policies import is_terminal_action, parse_action


def test_last_line_action():
    assert parse_action("I will click.\nclick('12')") == "click('12')"


def test_mid_prose_action_is_ignored():
    # the old implementation returned click('12') from mid-prose text
    assert parse_action("I could click('12') but let's think more.\nDone thinking.") is None


def test_action_block_wins():
    text = "reasoning click('1')\n```action\nclick('42')\n```\ntrailing prose"
    assert parse_action(text) == "click('42')"


def test_last_line_with_multiple_mentions():
    text = "maybe click('1') or click('2')\nfill('5', 'hello')"
    assert parse_action(text) == "fill('5', 'hello')"


def test_none_on_empty():
    assert parse_action("") is None
    assert parse_action("no action here") is None


def test_rejects_trailing_payload_or_second_answer_tag():
    assert parse_action("scroll(0, 300)<answer> scroll(0, 300)") is None
    assert parse_action("click('12') and click('13')") is None


def test_terminal_detection():
    assert is_terminal_action("send_msg_to_user('done')")
    assert not is_terminal_action("click('3')")
