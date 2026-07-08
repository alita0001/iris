"""Configure WebArena's LLM-based reward judge.

WebArena computes reward via `task.validate()`. Some tasks fall back to an LLM
judge (`llm_fuzzy_match` / `llm_ua_match`), which:
  * builds an OpenAI client from OPENAI_API_KEY, and
  * calls a HARDCODED model `gpt-4-1106-preview`.

That is the only reason the collector demanded an OpenAI key. We support three
modes (chosen by `configure_reward_judge`, driven by env var REVACT_WA_JUDGE):

  * "off"      -> short-circuit the judge to 0.0 (no key needed; reward is a
                  placeholder on the ~34/187 fuzzy shopping tasks). Default when
                  no OPENAI_API_KEY is present.
  * "deepseek" -> route the judge to DeepSeek (your key), keeping WebArena's own
                  judge prompts/logic. Reuses DEEPSEEK_API_KEY.
  * "openai"   -> leave WebArena as-is (needs a real OPENAI_API_KEY / GPT-4).

You do NOT need reward for trajectory collection (S2); "deepseek"/"openai" only
matter when you want real success labels (e.g. filtering expert trajectories).
"""
from __future__ import annotations

import os

_STATE = {"mode": None}

# The raw `webarena` package reads UNPREFIXED site env vars (SHOPPING, REDDIT,
# ...) and asserts them non-empty at import time. BrowserGym only sets the
# WA_-prefixed ones (and not until env.reset). Since we import webarena's eval
# modules early (to patch the judge), copy WA_X -> X first, or the import
# crashes with "Please setup the URLs to each site".
_WA_SITES = ["SHOPPING", "SHOPPING_ADMIN", "REDDIT", "GITLAB",
             "WIKIPEDIA", "MAP", "HOMEPAGE"]


def _ensure_webarena_site_env() -> None:
    for s in _WA_SITES:
        if not os.environ.get(s) and os.environ.get("WA_" + s):
            os.environ[s] = os.environ["WA_" + s]


# --------------------------------------------------------------------------- #
# Mode: off (short-circuit)
# --------------------------------------------------------------------------- #
def _zero(pred=None, reference=None, question=None, *args, **kwargs) -> float:
    return 0.0


def disable_llm_reward(verbose: bool = True) -> bool:
    if _STATE.get("mode") == "off":
        return True
    _ensure_webarena_site_env()
    patched = []
    for modname in (
        "webarena.evaluation_harness.helper_functions",
        "webarena.evaluation_harness.evaluators",
    ):
        try:
            mod = __import__(modname, fromlist=["*"])
        except Exception:
            continue
        for fn in ("llm_fuzzy_match", "llm_ua_match"):
            if hasattr(mod, fn):
                setattr(mod, fn, _zero)
                patched.append(f"{modname}.{fn}")
    if verbose and patched:
        print("[webarena_patch] LLM reward judge OFF (no key needed; placeholder "
              "reward on fuzzy tasks).")
    _STATE["mode"] = "off"
    return bool(patched)


# --------------------------------------------------------------------------- #
# Mode: route to an OpenAI-compatible endpoint (DeepSeek / vLLM / ...)
# --------------------------------------------------------------------------- #
def route_llm_reward(
    base_url: str | None = None,
    api_key_env: str = "DEEPSEEK_API_KEY",
    model: str | None = None,
    verbose: bool = True,
) -> bool:
    """Point WebArena's judge client at an OpenAI-compatible endpoint and force
    the judge model. WebArena's own judge prompts/parsing are left intact."""
    if _STATE.get("mode") == "route":
        return True
    base_url = (
        base_url
        or os.environ.get("REVACT_WA_JUDGE_BASE_URL")
        or "https://api.deepseek.com/v1"
    ).rstrip("/")
    model = (
        model
        or os.environ.get("REVACT_WA_JUDGE_MODEL")
        or os.environ.get("REVACT_LLM_MODEL")
        or "deepseek-chat"
    )
    key = os.environ.get(api_key_env, "").strip()
    if not key:
        raise RuntimeError(
            f"WebArena judge routing needs a key. Set: export {api_key_env}=sk-..."
        )

    _ensure_webarena_site_env()
    try:
        import openai  # provided by webarena deps
        import webarena.llms.providers.openai_utils as ou
        import webarena.evaluation_harness.helper_functions as hf
    except Exception as e:  # e.g. missing site URLs -> don't hard-crash collection
        if verbose:
            print(f"[webarena_patch] could not route judge ({type(e).__name__}: {e}); "
                  "falling back to OFF.")
        return disable_llm_reward(verbose=verbose)

    client = openai.OpenAI(api_key=key, base_url=base_url)
    # Replace the cached client + factory so get_openai_client() returns ours.
    ou._client = client
    ou.get_openai_client = lambda: client

    # Force the judge model (WebArena hardcodes gpt-4-1106-preview).
    orig_gen = hf.generate_from_openai_chat_completion

    def _gen(*args, **kwargs):
        kwargs["model"] = model
        return orig_gen(*args, **kwargs)

    hf.generate_from_openai_chat_completion = _gen
    if verbose:
        print(f"[webarena_patch] LLM reward judge -> {base_url} model={model} "
              f"(key from ${api_key_env})")
    _STATE["mode"] = "route"
    return True


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
def configure_reward_judge(verbose: bool = True) -> str:
    """Decide judge behavior from env. Returns the chosen mode."""
    mode = os.environ.get("REVACT_WA_JUDGE", "").strip().lower()

    if not mode:
        # auto: keep real reward if a real OpenAI key is present, else off.
        mode = "openai" if os.environ.get("OPENAI_API_KEY") else "off"

    if mode in ("off", "disable", "none"):
        disable_llm_reward(verbose=verbose)
        return "off"
    if mode in ("deepseek", "route", "vllm", "openai_compatible"):
        route_llm_reward(verbose=verbose)
        return "route"
    # mode == "openai": leave WebArena untouched (needs OPENAI_API_KEY).
    if verbose:
        print("[webarena_patch] using WebArena's native OpenAI judge "
              "(requires OPENAI_API_KEY + gpt-4-1106-preview access).")
    _STATE["mode"] = "openai"
    return "openai"


# Backward-compatible name used by earlier make_env wiring.
def maybe_disable_llm_reward(verbose: bool = True) -> bool:
    return configure_reward_judge(verbose=verbose) == "off"
