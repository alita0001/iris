"""P0: single source of truth for the agent-facing prompt format.

Training samples (data/assemble, data/multiturn), the rollout policy
(policies.LLMActionPolicy), and deployment of the trained model all render the
SAME user format:

    <goal>
    {goal}

    <history>
    - {action} -> {obs summary}      (or "(none)")

    <observation>
    {axtree}

Before P0, training samples had no <history> section while the rollout loop
had one -> train/deploy distribution mismatch. Any change to the format must
happen HERE, followed by re-materialization (assemble + split).

In MULTI-TURN samples only the first user turn carries <goal>+<history>;
subsequent user turns carry just the new <observation> (the conversation
itself is the history -- see data/multiturn.py).
"""
from __future__ import annotations

from typing import Optional

# --------------------------------------------------------------------------- #
# System prompts
# --------------------------------------------------------------------------- #
# For the TRAINED agent (single-step + multi-turn samples, and deployment).
# Routine steps answer directly; state-changing steps use the full think block.
SYSTEM = (
    "You are a safe web agent operating a website through an accessibility tree. "
    "You receive the goal, a short history of your previous actions, and the "
    "current page; earlier turns of this conversation are also your history. "
    "Act with exactly ONE action per turn, chosen from: click('bid'), "
    "fill('bid', 'text'), select_option('bid', 'value'), scroll(0, 300), "
    "go_back(), goto('url'), send_msg_to_user('text'), "
    "report_infeasible('reason'), noop(). "
    "For routine navigation output `<answer> action` directly. Whenever the "
    "action you are considering could change account or website state, FIRST "
    "state the current situation, predict the effect of that action, judge "
    "whether it is REVERSIBLE in this environment, and decide EXECUTE / VERIFY "
    "/ CONFIRM / AVOID. Output exactly: <think><observation>..<reasoning>.."
    "<prediction>..<reversibility>..<decision>..</think><answer>.."
)

# For the trajectory-collection LLM (e.g. DeepSeek): plain ReAct-ish prompt,
# no think schema — collector output never becomes a training label, only the
# executed actions and observations do.
SYSTEM_COLLECTOR = (
    "You are a web agent operating a shopping website through an accessibility "
    "tree. At each step you receive the goal, a short history, and the current "
    "page's interactive elements as lines like `[bid] role 'name'`. Respond with "
    "your brief reasoning followed by EXACTLY ONE action on its own final line, "
    "using one of: click('bid'), fill('bid', 'text'), select_option('bid', "
    "'value'), scroll(0, 300), go_back(), goto('url'), "
    "send_msg_to_user('answer'), report_infeasible('reason'), noop(). "
    "Use only bids that appear in the current page. Respect any 'do not' "
    "constraints in the goal."
)


# --------------------------------------------------------------------------- #
# User-message rendering / parsing
# --------------------------------------------------------------------------- #
def history_block(history: Optional[list]) -> str:
    """history entries are {'action': str, 'obs': str} dicts (collect format)."""
    lines = [f"- {h.get('action', '')} -> {h.get('obs', '')}" for h in (history or [])]
    return "\n".join(lines) if lines else "(none)"


def render_user(goal: str, obs_txt: str, history: Optional[list] = None) -> str:
    """First-turn user message: goal + history + current observation."""
    return (f"<goal>\n{goal}\n\n"
            f"<history>\n{history_block(history)}\n\n"
            f"<observation>\n{obs_txt}\n")


def render_followup(obs_txt: str) -> str:
    """Subsequent user turns in a multi-turn sample: only the new observation."""
    return f"<observation>\n{obs_txt}\n"


def parse_user(user: str) -> dict:
    """Split a user message into goal / history / obs.

    Handles both the current 3-section format and the pre-P0 2-section format
    (old JSONL rows and workbench test fixtures stay loadable).
    """
    import re

    m = re.search(r"<goal>\s*\n(.*?)\n\s*\n<history>\s*\n(.*?)\n\s*\n"
                  r"<observation>\s*\n(.*)", user, re.DOTALL)
    if m:
        hist = m.group(2).strip()
        return {"goal": m.group(1).strip(),
                "history": "" if hist == "(none)" else hist,
                "obs": m.group(3).strip()}
    m = re.search(r"<goal>\s*\n(.*?)\n\s*\n<observation>\s*\n(.*)", user, re.DOTALL)
    if m:
        return {"goal": m.group(1).strip(), "history": "", "obs": m.group(2).strip()}
    return {"goal": user.strip(), "history": "", "obs": ""}


# --------------------------------------------------------------------------- #
# History synthesis for reach-constructed states
# --------------------------------------------------------------------------- #
def history_from_plan(plan: Optional[list]) -> list[dict]:
    """Render a semantic reach plan (S3/scale) as agent-loop history entries.

    Plans replay by text, so the entries use the click target text where the
    live loop would have a bid — the closest honest rendering available
    without re-executing the plan."""
    entries: list[dict] = []
    for step in plan or []:
        kind = step[0]
        if kind == "goto":
            entries.append({"action": f"goto('{step[1]}')", "obs": step[1]})
        elif kind in ("click_text", "click_text_optional"):
            entries.append({"action": f"click('{step[1]}')",
                            "obs": f"clicked '{step[1]}'"})
        elif kind == "go_back":
            entries.append({"action": "go_back()", "obs": "previous page"})
    return entries


def canonical_history(action_type: str, url: str) -> list[dict]:
    """Fallback for records that predate plan recording (scale.py wrote
    reach_plan=[] before P0): the canonical navigation for that action type."""
    if action_type == "place_order":
        return [{"action": "click('add to cart')", "obs": "item added to cart"},
                {"action": f"goto('{url}')", "obs": url},
                {"action": "click('next')", "obs": "payment step"}]
    if url:
        return [{"action": f"goto('{url}')", "obs": url}]
    return []


def state_history(state: dict) -> tuple[list[dict], str]:
    """(history entries, provenance) for a reached-state record.

    provenance: 'plan' (recorded reach plan), 'canonical' (synthesized from
    action type + url), or 'none'."""
    entries = history_from_plan(state.get("reach_plan"))
    if entries:
        return entries, "plan"
    entries = canonical_history(state.get("action_type", ""), state.get("url", ""))
    return entries, ("canonical" if entries else "none")
