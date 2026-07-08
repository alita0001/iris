"""S2 policies that drive the env loop.

* ScriptedShoppingPolicy - deterministic action plans for the 3 pilot flows;
  works against MockShoppingEnv for offline pipeline tests.
* LLMActionPolicy        - an OpenAI-compatible chat policy (DeepSeek / vLLM /
  OpenAI). Used to collect expert trajectories with a strong model.

SECURITY: LLMActionPolicy NEVER takes a raw API key as an argument or hardcodes
one. It reads the key from an environment variable (name configurable, default
``DEEPSEEK_API_KEY``). base_url / model come from args or env with defaults.
Only stdlib (urllib) is used, so no extra dependency.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Optional

from . import prompts
from .config import MAX_AXTREE_CHARS_POLICY
from .envs.obs_utils import prune_axtree_txt

# Actions BrowserGym's high-level WebArena action set understands (common subset).
_ACTION_VERBS = (
    "click",
    "fill",
    "select_option",
    "hover",
    "press",
    "scroll",
    "goto",
    "go_back",
    "go_forward",
    "new_tab",
    "tab_close",
    "send_msg_to_user",
    "report_infeasible",
    "noop",
)
_ACTION_RE = re.compile(
    r"\b(" + "|".join(_ACTION_VERBS) + r")\s*\([^\n]*\)", re.IGNORECASE
)


# Actions that deliver the agent's final answer / give up. These MUST be
# executed in the env (so WebArena's validator can score the chat message);
# the collect loop ends the trajectory *after* executing them.
TERMINAL_VERBS = ("send_msg_to_user", "report_infeasible")


def action_verb(action: Optional[str]) -> str:
    if not action:
        return ""
    return action.split("(", 1)[0].strip()


def is_terminal_action(action: Optional[str]) -> bool:
    return action_verb(action) in TERMINAL_VERBS


_ACTION_BLOCK_RE = re.compile(r"```(?:action)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def parse_action(text: str) -> Optional[str]:
    """Extract THE action from a model response.

    Reasoning models mention actions mid-prose ("I could click('12') but...");
    taking the last regex match anywhere in the text picks those up. Instead:
      1. if fenced ```action``` blocks exist, use the last block;
      2. otherwise only the LAST non-empty line counts (the system prompt
         requires the action on its own final line).
    Returns None when neither contains a well-formed action call.
    """
    if not text:
        return None
    blocks = _ACTION_BLOCK_RE.findall(text)
    for blk in reversed(blocks):
        m = _ACTION_RE.search(blk)
        if m:
            return m.group(0).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        m = _ACTION_RE.search(lines[-1])
        if m:
            return m.group(0).strip()
    return None


# --------------------------------------------------------------------------- #
# Scripted policy (offline / pilot)
# --------------------------------------------------------------------------- #
_FLOWS = {
    # reach the "add to cart" affordance (product page)
    "add_to_cart": ["click('11')", "click('20')"],
    # reach the "place order" affordance (checkout page)
    "place_order": ["click('11')", "click('20')", "click('30')", "click('40')"],
    # reach the "delete address" affordance (address book)
    "delete_address": ["click('12')"],
}


class ScriptedShoppingPolicy:
    """Deterministic plan for MockShoppingEnv; stops (returns None) when done."""

    def __init__(self, flow: str):
        if flow not in _FLOWS:
            raise ValueError(f"unknown flow {flow!r}; choose from {list(_FLOWS)}")
        self.flow = flow
        self._plan = list(_FLOWS[flow])
        self._i = 0

    def reset(self):
        self._i = 0

    def act(self, obs_view: dict, goal: str = "", history: Optional[list] = None) -> Optional[str]:
        if self._i >= len(self._plan):
            return None
        a = self._plan[self._i]
        self._i += 1
        return a


# --------------------------------------------------------------------------- #
# LLM policy (OpenAI-compatible: DeepSeek / vLLM / OpenAI)
# --------------------------------------------------------------------------- #
# P0: prompt text lives in revact/prompts.py so training samples and the
# rollout loop share ONE user format (goal + history + observation).
_SYSTEM_PROMPT = prompts.SYSTEM_COLLECTOR


class LLMActionPolicy:
    """OpenAI-compatible chat completion policy.

    Parameters
    ----------
    model : str | None      -> defaults to env REVACT_LLM_MODEL or 'deepseek-v4-pro'
    base_url : str | None    -> defaults to env REVACT_LLM_BASE_URL or DeepSeek
    api_key_env : str        -> NAME of the env var holding the key
                                (default 'DEEPSEEK_API_KEY'). The value is read
                                lazily and never stored in logs.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_env: str = "DEEPSEEK_API_KEY",
        temperature: float = 0.0,
        max_tokens: int = 8192,
        max_history: int = 6,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.model = model or os.environ.get("REVACT_LLM_MODEL", "deepseek-v4-pro")
        self.base_url = (
            base_url
            or os.environ.get("REVACT_LLM_BASE_URL", "https://api.deepseek.com/v1")
        ).rstrip("/")
        self.api_key_env = api_key_env
        # Workbench/env override (same pattern as REVACT_LLM_MAX_TOKENS below).
        self.temperature = float(os.environ.get("REVACT_LLM_TEMPERATURE", temperature))
        self.top_p = os.environ.get("REVACT_LLM_TOP_P", "")
        # Reasoning models (e.g. deepseek-v4-pro) spend tokens on hidden
        # reasoning; 512 was too small and left `content` empty. Allow override.
        self.max_tokens = int(os.environ.get("REVACT_LLM_MAX_TOKENS", max_tokens))
        self.max_history = max_history
        self.timeout = timeout
        self.max_retries = max_retries
        self.system_prompt = _SYSTEM_PROMPT
        self.last_raw_response: str = ""
        self.last_finish_reason: str = ""

    def _api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise RuntimeError(
                f"API key not found. Set it via:  export {self.api_key_env}=sk-..."
                f"  (base_url={self.base_url}, model={self.model})"
            )
        return key

    def reset(self):
        self.last_raw_response = ""

    def _build_messages(self, obs_view: dict, goal: str, history: list) -> list:
        obs_txt = prune_axtree_txt(obs_view.get("axtree_txt", ""),
                                   max_chars=MAX_AXTREE_CHARS_POLICY)
        user = prompts.render_user(goal, obs_txt,
                                   (history or [])[-self.max_history:])
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]

    def _post(self, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key()}",
            },
            method="POST",
        )
        last_err = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM API call failed after {self.max_retries} tries: {last_err}")

    def act(self, obs_view: dict, goal: str = "", history: Optional[list] = None) -> Optional[str]:
        payload = {
            "model": self.model,
            "messages": self._build_messages(obs_view, goal, history or []),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.top_p:
            payload["top_p"] = float(self.top_p)
        resp = self._post(payload)
        try:
            choice = resp["choices"][0]
            msg = choice.get("message", {}) or {}
            finish = choice.get("finish_reason", "") or ""
        except (KeyError, IndexError, TypeError):
            msg, finish = {}, ""
        # Reasoning models may leave `content` empty and put text in
        # `reasoning_content`; fall back to it so we can still parse an action.
        content = (msg.get("content") or "").strip()
        if not content:
            content = (msg.get("reasoning_content") or "").strip()
        self.last_raw_response = content
        self.last_finish_reason = finish
        # Return the action as-is (including terminal send_msg/report_infeasible)
        # so the env executes it and WebArena can score the answer. The collect
        # loop is responsible for ending the trajectory after a terminal action.
        return self._extract_action(content)

    def _extract_action(self, content: str) -> Optional[str]:
        return parse_action(content)


class IrisPolicy(LLMActionPolicy):
    """Deploys the TRAINED IRIS model (served via an OpenAI-compatible
    endpoint, e.g. `vllm serve`): the training system prompt and user format
    (prompts.SYSTEM + goal/history/observation), with the action taken from
    the <answer> field of the structured output.

    `last_fields` carries the parsed <reversibility>/<decision> of the latest
    response so rollout evaluation can score world-model claims, not just
    behavior."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_prompt = prompts.SYSTEM
        self.last_fields: dict = {}

    def _extract_action(self, content: str) -> Optional[str]:
        self.last_fields = {
            tag: (m.group(1).strip() if (m := re.search(
                rf"<{tag}>\s*([^\n<]+)", content)) else "")
            for tag in ("reversibility", "decision")
        }
        m = re.search(r"<answer>\s*(.+)", content, re.DOTALL)
        if m:
            first = next((ln.strip() for ln in m.group(1).splitlines()
                          if ln.strip()), "")
            a = _ACTION_RE.search(first)
            if a:
                return a.group(0).strip()
        return parse_action(content)



