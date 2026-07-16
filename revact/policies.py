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

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Optional
from urllib.parse import urlparse

from . import prompts
from .config import POLICY_HISTORY_STEPS
from .envs.obs_utils import extract_interactive_bids
from .train.validators import parse_action as parse_browsergym_action

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


def _literal_action_line(line: str) -> Optional[str]:
    """Accept one complete literal BrowserGym call and no trailing payload.

    Regex search used to accept strings such as
    ``scroll(0, 300)<answer> scroll(0, 300)``.  Browser execution must be
    stricter than output-format scoring: the whole selected line is parsed as
    one literal AST call and its primitive must be in the supported action
    vocabulary.
    """
    candidate = str(line or "").strip()
    if not candidate or "<answer>" in candidate.lower():
        return None
    parsed = parse_browsergym_action(candidate)
    if parsed is None or parsed.name.lower() not in _ACTION_VERBS:
        return None
    return candidate


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
    # Structured IRIS completions are common call-trace inputs.  Accept one
    # answer block, but reject duplicate tags and require its first non-empty
    # line to be exactly one literal action.
    answer_count = text.lower().count("<answer>")
    if answer_count:
        if answer_count != 1:
            return None
        prefix, answer = re.split(
            r"<answer>", text, maxsplit=1, flags=re.IGNORECASE)
        if prefix.strip() and "</think>" not in prefix.lower():
            return None
        first = next((line.strip() for line in answer.splitlines()
                      if line.strip()), "")
        return _literal_action_line(first)
    blocks = _ACTION_BLOCK_RE.findall(text)
    for blk in reversed(blocks):
        lines = [line.strip() for line in blk.splitlines() if line.strip()]
        for line in reversed(lines):
            if action := _literal_action_line(line):
                return action
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return _literal_action_line(lines[-1])
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
# P2: read through the registry at construction time so workbench prompt
# overrides apply to freshly-built policies without a restart.


class LLMActionPolicy:
    """OpenAI-compatible chat completion policy.

    Parameters
    ----------
    model : str | None      -> defaults to env REVACT_LLM_MODEL or 'deepseek-v4-pro'
    base_url : str | None    -> defaults to env REVACT_LLM_BASE_URL or DeepSeek
    api_key_env : str        -> NAME of the env var holding the key
                                (default 'DEEPSEEK_API_KEY'). The value is read
                                lazily and never stored in logs.
    provider : str | None     -> provenance label only; endpoint behavior remains
                                OpenAI-compatible and user-configurable.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_env: str = "DEEPSEEK_API_KEY",
        provider: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        max_history: Optional[int] = None,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.model = model or os.environ.get("REVACT_LLM_MODEL", "deepseek-v4-pro")
        self.base_url = (
            base_url
            or os.environ.get("REVACT_LLM_BASE_URL", "https://api.deepseek.com/v1")
        ).rstrip("/")
        self.provider = (
            provider or os.environ.get("REVACT_LLM_PROVIDER", "") or
            ("openrouter" if "openrouter.ai" in self.base_url else
             "deepseek" if "deepseek.com" in self.base_url else
             "openai" if "api.openai.com" in self.base_url else "custom")
        )
        self.api_key_env = api_key_env
        # Workbench/env override (same pattern as REVACT_LLM_MAX_TOKENS below).
        self.temperature = float(os.environ.get("REVACT_LLM_TEMPERATURE", temperature))
        self.top_p = os.environ.get("REVACT_LLM_TOP_P", "")
        # Reasoning models (e.g. deepseek-v4-pro) spend tokens on hidden
        # reasoning; 512 was too small and left `content` empty. Allow override.
        self.max_tokens = int(os.environ.get("REVACT_LLM_MAX_TOKENS", max_tokens))
        # One config value drives collection, IrisPolicy deployment, rollout,
        # and trajectory-conditioned dataset materialization.
        self.max_history = (POLICY_HISTORY_STEPS if max_history is None
                            else int(max_history))
        self.timeout = timeout
        self.max_retries = max_retries
        self.system_prompt = prompts.get("collector_system")
        self.last_raw_response: str = ""
        self.last_finish_reason: str = ""
        # Immutable-by-convention trace of the latest OpenAI-compatible call.
        # This deliberately records the *name* of the credential environment
        # variable, never its value.  Formal rollout copies these fields so an
        # observed model error can be audited before it is admitted to DPO.
        self.last_request_messages: list[dict] = []
        self.last_request_sha256: str = ""
        self.last_response_id: str = ""
        self.last_response_model: str = ""
        self.last_response_created: int | str | None = None
        self.last_response_usage: dict = {}

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
        self.last_finish_reason = ""
        self.last_request_messages = []
        self.last_request_sha256 = ""
        self.last_response_id = ""
        self.last_response_model = ""
        self.last_response_created = None
        self.last_response_usage = {}

    def execution_provenance(self) -> dict:
        """Return serialisable model/decode identity without a secret value."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "credential_value_stored": False,
            "decode": {
                "temperature": self.temperature,
                "top_p": float(self.top_p) if self.top_p else None,
                "max_tokens": self.max_tokens,
                "max_history": self.max_history,
            },
            "response_id": self.last_response_id,
            "response_model": self.last_response_model,
            "response_created": self.last_response_created,
            "finish_reason": self.last_finish_reason,
            "usage": dict(self.last_response_usage),
        }

    def _build_messages(self, obs_view: dict, goal: str, history: list) -> list:
        return prompts.build_policy_messages(
            goal,
            obs_view.get("axtree_txt", ""),
            history or [],
            system_prompt=self.system_prompt,
            max_history=self.max_history,
        )

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

    def _act_messages(self, messages: list[dict]) -> Optional[str]:
        """Issue one call for an already-rendered, exact policy prompt."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.top_p:
            payload["top_p"] = float(self.top_p)
        # JSON round-trip gives the trace its own object graph: callers cannot
        # mutate it through a reference retained by a custom HTTP client.
        self.last_request_messages = json.loads(json.dumps(
            messages, ensure_ascii=False))
        self.last_request_sha256 = hashlib.sha256(json.dumps(
            self.last_request_messages, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode("utf-8")).hexdigest()
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
        self.last_response_id = str(resp.get("id") or "")
        self.last_response_model = str(resp.get("model") or "")
        self.last_response_created = resp.get("created")
        usage = resp.get("usage")
        self.last_response_usage = dict(usage) if isinstance(usage, dict) else {}
        # Return the action as-is (including terminal send_msg/report_infeasible)
        # so the env executes it and WebArena can score the answer. The collect
        # loop is responsible for ending the trajectory after a terminal action.
        return self._extract_action(content)

    def act_messages(self, messages: list[dict]) -> Optional[str]:
        """Call the model on an immutable source prompt without rerendering it.

        Formal point capture uses this only when it intentionally evaluates the
        exact serialized SFT prompt.  Ordinary deployment continues through
        :meth:`act` and the shared builder.
        """
        roles = [message.get("role") for message in messages
                 if isinstance(message, dict)]
        if roles != ["system", "user"] or len(messages) != 2:
            raise ValueError(
                "exact policy prompt must contain one system and one user message")
        return self._act_messages(messages)

    def act(self, obs_view: dict, goal: str = "", history: Optional[list] = None) -> Optional[str]:
        messages = self._build_messages(obs_view, goal, history or [])
        return self._act_messages(messages)

    def _extract_action(self, content: str) -> Optional[str]:
        return parse_action(content)


_READ_ONLY_MUTATION_TERMS = re.compile(
    r"\b(?:add\s+to\s+(?:cart|wish\s*list|compare)|subscribe|unsubscribe|"
    r"upvote|downvote|retract\s+(?:upvote|downvote)|reorder|place\s+order|"
    r"checkout|delete|remove|refund|pay|purchase|publish|create|submit|"
    r"save|update|edit\s+address|cancel\s+order|confirm\s+order)\b",
    re.IGNORECASE,
)
_READ_ONLY_BUTTONS = re.compile(
    r"\b(?:search|next|previous|view|details?|filter|sort|go|back|close)\b",
    re.IGNORECASE,
)
_READ_ONLY_LINE = re.compile(r"\[[^\]]+\]\s+([A-Za-z_][\w-]*)\s*(.*)")


def read_only_action_error(action: str, obs_view: dict) -> str | None:
    """Return why a proposed live action is not demonstrably read-only.

    This is an execution safety boundary for diagnostic collection, not a
    safety/recoverability labeler.  Ambiguous controls are rejected; no result
    from this function may enter the grounding label chain.
    """
    parsed = parse_browsergym_action(action)
    if parsed is None:
        return "unparseable_action"
    if parsed.name in {
            "send_msg_to_user", "report_infeasible", "noop", "go_back",
            "go_forward", "scroll", "hover", "new_tab", "tab_close"}:
        return None
    if parsed.name == "goto":
        if len(parsed.args) != 1 or not isinstance(parsed.args[0], str):
            return "goto_requires_literal_url"
        current = urlparse(str(obs_view.get("url") or ""))
        target = urlparse(parsed.args[0])
        if not current.scheme or not current.netloc:
            return "current_origin_unknown"
        if (target.scheme, target.netloc) != (current.scheme, current.netloc):
            return "cross_origin_navigation"
        return None
    if parsed.name not in {"click", "fill"}:
        return f"primitive_not_read_only:{parsed.name}"
    if not parsed.bid:
        return f"{parsed.name}_missing_bid"
    matches = [item for item in extract_interactive_bids(
        str(obs_view.get("axtree_txt") or "")) if item["bid"] == parsed.bid]
    if len(matches) != 1:
        return f"{parsed.name}_bid_not_uniquely_visible"
    line = matches[0]["line"]
    match = _READ_ONLY_LINE.search(line)
    if not match:
        return "interactive_role_unparseable"
    role, label = match.group(1).lower(), match.group(2)
    if _READ_ONLY_MUTATION_TERMS.search(label):
        return "mutation_like_control"
    if parsed.name == "fill":
        if role not in {"searchbox", "textbox"} or not re.search(
                r"\bsearch\b", label, re.IGNORECASE):
            return "fill_not_search_field"
        return None
    if role in {"link", "tab", "menuitem", "treeitem"}:
        return None
    if role == "button" and _READ_ONLY_BUTTONS.search(label):
        return None
    return f"click_role_or_label_not_read_only:{role}"


class ReadOnlyPolicyGuard:
    """Fail-closed wrapper for live information-retrieval smoke runs."""

    def __init__(self, policy):
        self.policy = policy
        self.guard_rejections: list[dict[str, str]] = []
        self.last_raw_response = ""
        self.last_finish_reason = ""
        self.last_request_messages: list[dict] = []
        self.last_request_sha256 = ""
        self.last_proposed_action = ""
        self.last_proposed_completion = ""
        self.last_executed_action = ""
        self.last_executed_completion = ""

    def reset(self):
        self.guard_rejections = []
        self.last_raw_response = ""
        self.last_finish_reason = ""
        self.last_request_messages = []
        self.last_request_sha256 = ""
        self.last_proposed_action = ""
        self.last_proposed_completion = ""
        self.last_executed_action = ""
        self.last_executed_completion = ""
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    @staticmethod
    def _guard_completion(action: str, reason: str) -> str:
        """Return a deployment-shaped completion for the action actually run.

        The inner model completion remains proposal evidence, but it cannot be
        recorded as the assistant completion for a different guard-substituted
        action.  This generated block is deliberately factual and mechanical:
        the guard, not a model, made the AVOID decision and no site mutation was
        attempted.
        """
        return (
            "<think>\n"
            "<observation>The proposed action was blocked by the read-only "
            f"execution guard ({reason}).\n"
            "<reasoning>The live run permits only demonstrably read-only "
            "actions, so execution must stop without attempting the proposal.\n"
            "<prediction>The site state will not be mutated by the substituted "
            "terminal action.\n"
            "<rev_check>No undo is required because the proposed site action "
            "was not executed.\n"
            "<reversibility>UNKNOWN\n"
            "<undo>none; proposed action not executed\n"
            "<decision>AVOID risk=1.0\n"
            "</think>\n"
            f"<answer> {action}"
        )

    def act(self, obs_view: dict, goal: str = "",
            history: Optional[list] = None) -> Optional[str]:
        action = self.policy.act(obs_view, goal=goal, history=history)
        proposed_completion = str(
            getattr(self.policy, "last_raw_response", "") or "")
        self.last_proposed_action = str(action or "")
        self.last_proposed_completion = proposed_completion
        self.last_raw_response = proposed_completion
        self.last_finish_reason = getattr(self.policy, "last_finish_reason", "")
        self.last_request_messages = list(getattr(
            self.policy, "last_request_messages", []) or [])
        self.last_request_sha256 = str(getattr(
            self.policy, "last_request_sha256", "") or "")
        if not action:
            self.last_executed_action = ""
            self.last_executed_completion = ""
            return action
        reason = read_only_action_error(action, obs_view)
        if reason is None:
            self.last_executed_action = action
            self.last_executed_completion = proposed_completion
            return action
        executed_action = \
            f"report_infeasible('read_only_guard:{reason}')"
        executed_completion = self._guard_completion(executed_action, reason)
        event = {
            "action": action,
            "proposed_action": action,
            "proposed_completion": proposed_completion,
            "reason": reason,
            "url": str(obs_view.get("url") or ""),
            "executed_action": executed_action,
            "executed_completion": executed_completion,
        }
        self.guard_rejections.append(event)
        self.last_executed_action = executed_action
        self.last_executed_completion = executed_completion
        # Downstream raw collection intentionally reads ``last_raw_response`` as
        # the completion corresponding to the action returned from ``act``.
        # Preserve the model proposal separately above instead of creating an
        # impossible completion/action pair.
        self.last_raw_response = executed_completion
        self.last_finish_reason = "read_only_guard"
        return executed_action

    def execution_provenance(self) -> dict:
        inner = getattr(self.policy, "execution_provenance", None)
        provenance = dict(inner() if callable(inner) else {})
        provenance["execution_wrapper"] = "read_only_policy_guard"
        provenance["guard_rejections"] = len(self.guard_rejections)
        provenance["last_proposed_action"] = self.last_proposed_action
        provenance["last_proposed_completion_sha256"] = (
            hashlib.sha256(self.last_proposed_completion.encode("utf-8")).hexdigest()
            if self.last_proposed_completion else "")
        provenance["last_executed_action"] = self.last_executed_action
        provenance["last_executed_completion_sha256"] = (
            hashlib.sha256(self.last_executed_completion.encode("utf-8")).hexdigest()
            if self.last_executed_completion else "")
        return provenance


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
        self.system_prompt = prompts.get("agent_system")
        self.last_fields: dict = {}

    def _extract_action(self, content: str) -> Optional[str]:
        self.last_fields = {
            tag: (m.group(1).strip() if (m := re.search(
                rf"<{tag}>\s*([^\n<]+)", content)) else "")
            for tag in ("reversibility", "undo", "decision")
        }
        if "<answer>" in content.lower():
            # A present but malformed answer is never rescued from prose or a
            # second injected tag.  The policy fails closed for this step.
            return parse_action(content)
        return parse_action(content)
