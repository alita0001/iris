"""Central configuration: paths, WebArena env, site paths, tolerances.

Values resolve in this order (later wins):
  1. built-in defaults below;
  2. ``configs/default.yaml`` (optional; ignored when PyYAML is unavailable so
     the package stays stdlib-importable);
  3. environment variables (``REVACT_DATA_ROOT``, ``WA_SHOPPING``, ...).

Safe to import anywhere; no heavy deps.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"

# Version stamp written into grounded-data manifests: bump when a signal or
# undo controller changes semantics, so old labels stay attributable.
# p2: added reddit (Postmill) probes; reddit vote_score reads the ACTIVE-state
#     up-control ('Retract upvote') so the score is not misread off a comment
#     widget (see docs/findings-multisite.md). Earlier reddit_vote rows stamped
#     p1 are pre-fix artifacts.  A latest-non-UNKNOWN rule exists only in the
#     frozen legacy class-smoke display loader and is forbidden for formal joins.
# p3 separates action effect from budget-relative recovery and forbids new
# mathematical IRREVERSIBLE claims from a bounded controller failure.
CONTROLLER_VERSION = "2026.07-p3"


def _load_yaml_config() -> dict:
    path = CONFIG_DIR / "default.yaml"
    if not path.exists():
        return {}
    try:
        import yaml  # optional dependency
    except ImportError:
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_YAML = _load_yaml_config()


def _cfg(*keys, default=None):
    node = _YAML
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


# --------------------------------------------------------------------------- #
# Paths (data layout: raw/ grounded/ train/)
# --------------------------------------------------------------------------- #
DATA_ROOT = Path(os.environ.get("REVACT_DATA_ROOT",
                                _cfg("paths", "data_root",
                                     default=str(PROJECT_ROOT / "data"))))

RAW_DIR = DATA_ROOT / "raw"
RAW_TRAJ_DIR = RAW_DIR / "trajectories"
STATE_BANK_DIR = RAW_DIR / "state_bank"
SCREENSHOT_DIR = RAW_DIR / "screenshots"
TASK_LIST_PATH = RAW_DIR / "shopping_task_ids.json"
PILOT_TASKS_PATH = RAW_DIR / "pilot_task_ids.json"
PRODUCT_URLS_PATH = RAW_DIR / "product_urls.json"

GROUNDED_DIR = DATA_ROOT / "grounded"
# ``reversibility.jsonl`` and ``MANIFEST.jsonl`` are frozen legacy/class-level
# smoke-probe assets.  Formal supervision is admitted only through the
# point-level body + 1:1 manifest below.
REVERSIBILITY_PATH = GROUNDED_DIR / "reversibility.jsonl"
POINT_GROUNDING_PATH = GROUNDED_DIR / "probe_points.jsonl"
POINT_MANIFEST_PATH = GROUNDED_DIR / "POINT_MANIFEST.jsonl"
GROUNDING_QUARANTINE_DIR = GROUNDED_DIR / "quarantine"

EVAL_DATA_DIR = DATA_ROOT / "eval"
EVALUATION_TRUTH_PATH = EVAL_DATA_DIR / "truth.jsonl"
EVALUATION_TRUTH_MANIFEST_PATH = EVAL_DATA_DIR / "TRUTH_MANIFEST.jsonl"

# Opinion baselines are deliberately outside formal grounding/evaluation
# truth.  V2 is keyed by point × evaluation-case/goal × rater; legacy v1 rows
# cannot identify the goal behind normative risk and remain audit-only.
OPINION_DIR = DATA_ROOT / "opinions"
OPINION_LABEL_PATH = OPINION_DIR / "opinion_labels.v2.jsonl"
OPINION_MANIFEST_PATH = OPINION_DIR / "OPINION_LABEL_MANIFEST.v2.jsonl"
OPINION_INPUT_DIR = OPINION_DIR / "inputs"
FORMAL_OPINION_INPUT_PATH = (
    OPINION_INPUT_DIR / "formal_request_constraint_inputs.v2.jsonl")
FORMAL_OPINION_INPUT_MANIFEST_PATH = (
    OPINION_INPUT_DIR / "FORMAL_REQUEST_CONSTRAINT_INPUT_MANIFEST.v2.jsonl")

TRAIN_DIR = DATA_ROOT / "train"
# Frozen pilot artifacts remain at ``sft/``, ``dpo/`` and ``splits/``.  They
# are intentionally *not* the defaults used by any trainer.  Formal trainers
# consume only the point-grounded split namespace below.
FORMAL_TRAIN_DIR = TRAIN_DIR / "formal"
FORMAL_SPLITS_DIR = FORMAL_TRAIN_DIR / "splits"
FORMAL_SFT_PATH = FORMAL_TRAIN_DIR / "iris_sft_transition_v3.jsonl"
FORMAL_MULTITURN_SFT_PATH = (
    FORMAL_TRAIN_DIR / "iris_sft_multiturn_transition_v3.jsonl")
FORMAL_DISTILLED_SFT_PATH = (
    FORMAL_TRAIN_DIR / "iris_sft_distilled_transition_v5.jsonl")
FORMAL_MULTITURN_DISTILLED_SFT_PATH = (
    FORMAL_TRAIN_DIR / "iris_sft_multiturn_distilled_transition_v5.jsonl")
# Candidate/on-policy DPO main sets are intentionally distinct from the
# synthetic-flip ablation written by the assemblers.
FORMAL_DPO_PATH = FORMAL_TRAIN_DIR / "iris_dpo_transition_v3.jsonl"
FORMAL_MULTITURN_DPO_PATH = (
    FORMAL_TRAIN_DIR / "iris_dpo_multiturn_transition_v3.jsonl")
# Only a manifest-pinned supplement whose basename matches the active release
# namespace may join the active split.  Older on-policy smoke assets remain
# inspectable but cannot silently contaminate a later formal release.
FORMAL_DPO_SUPPLEMENT_GLOB = \
    "iris_dpo_on_policy_transition_v3_strict_*.jsonl"
FORMAL_SFT_TRAIN_PATH = FORMAL_SPLITS_DIR / "sft_train.jsonl"
FORMAL_DPO_TRAIN_PATH = FORMAL_SPLITS_DIR / "dpo_train.jsonl"
SFT_PATH = TRAIN_DIR / "sft" / "revact_sft.jsonl"
DPO_PATH = TRAIN_DIR / "dpo" / "revact_dpo.jsonl"
SPLITS_DIR = TRAIN_DIR / "splits"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# --------------------------------------------------------------------------- #
# WebArena / BrowserGym
# --------------------------------------------------------------------------- #
WEBARENA_GYM_PREFIX = "browsergym/webarena."
SITE = "shopping"
WA_SHOPPING = os.environ.get("WA_SHOPPING", _cfg("webarena", "shopping", default=""))
WA_SHOPPING_ADMIN = os.environ.get("WA_SHOPPING_ADMIN",
                                   _cfg("webarena", "shopping_admin", default=""))
WA_REDDIT = os.environ.get("WA_REDDIT", _cfg("webarena", "reddit", default=""))
WA_GITLAB = os.environ.get("WA_GITLAB", _cfg("webarena", "gitlab", default=""))
# Site-specific task ids bootstrap the correct authenticated browser storage.
SESSION_TASK_ID = _cfg("webarena", "session_task", default="webarena.21")
SHOPPING_ADMIN_SESSION_TASK_ID = _cfg(
    "webarena", "shopping_admin_session_task", default="webarena.0")
# A reddit task id bootstraps a logged-in Postmill session for reddit probes.
REDDIT_SESSION_TASK_ID = _cfg("webarena", "reddit_session_task", default="webarena.27")

# Stable Magento paths (appended to the base url).
SHOPPING_PATHS = {
    "cart": "/checkout/cart/",
    "checkout": "/checkout/",
    "address": "/customer/address/",
    "address_new": "/customer/address/new/",
    "order_history": "/sales/order/history/",
    "wishlist": "/wishlist/",
    "compare": "/catalog/product_compare/index/",
    "newsletter": "/newsletter/manage/",
}

# Stable Postmill (reddit) paths (appended to the base url).
REDDIT_PATHS = {
    "forums": "/forums",
    "submit": "/submit",
    "front": "/",
}

# --------------------------------------------------------------------------- #
# Site registry: the single source of truth for "which sites this workbench
# knows how to ground". Each site names the env var holding its base URL, a
# session-bootstrap task id, and the paths dict its probes navigate with. Adding
# a mirror (gitlab, wikipedia, ...) is: deploy it, add a row here, add a probe
# module under grounding/probes/. Everything downstream (CLI --site, workbench
# site facet, cross-site split) reads this registry rather than hard-coding
# "shopping".
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SiteSpec:
    name: str                 # canonical site key (matches ProbeSpec.site)
    label: str                # human label for the UI
    base_env: str             # env var holding the base url
    session_task: str         # a task id that bootstraps a logged-in session
    paths_name: str           # which *_PATHS dict this site uses
    environment_family: str   # underlying application/technology family
    requires_login: bool = True


SITES: dict[str, SiteSpec] = {
    "shopping": SiteSpec("shopping", "Shopping (Magento)", "WA_SHOPPING",
                         str(SESSION_TASK_ID), "SHOPPING_PATHS", "magento"),
    "shopping_admin": SiteSpec("shopping_admin", "Shopping Admin (Magento)",
                               "WA_SHOPPING_ADMIN",
                               str(SHOPPING_ADMIN_SESSION_TASK_ID),
                               "SHOPPING_PATHS", "magento"),
    "reddit": SiteSpec("reddit", "Reddit (Postmill)", "WA_REDDIT",
                       str(REDDIT_SESSION_TASK_ID), "REDDIT_PATHS", "postmill"),
}

_PATHS_BY_NAME = {"SHOPPING_PATHS": SHOPPING_PATHS, "REDDIT_PATHS": REDDIT_PATHS}


def site_base(site: str) -> str:
    """Resolve a site's base URL from its registered env var ('' if unset)."""
    spec = SITES.get(site)
    if spec is None:
        return ""
    return os.environ.get(spec.base_env, "") or globals().get(spec.base_env, "") or ""


def site_paths(site: str) -> dict:
    spec = SITES.get(site)
    return _PATHS_BY_NAME.get(spec.paths_name, {}) if spec else {}


def site_environment_family(site: str) -> str:
    """Return the application family; concrete sites remain separate axes."""
    spec = SITES.get(site)
    return spec.environment_family if spec else ""

# --------------------------------------------------------------------------- #
# Key-state detection during collection (S2). Keyword matching is intentionally
# shallow here; grounding refines it with probes.
# --------------------------------------------------------------------------- #
PILOT_ACTION_KEYWORDS: dict[str, list[str]] = {
    "add_to_cart": ["add to cart", "add to bag"],
    "place_order": ["place order", "proceed to checkout", "checkout", "confirm order"],
    "delete_address": ["delete", "remove address", "delete address"],
}
ADDRESS_URL_HINTS = ["address", "/customer/address", "account"]

# --------------------------------------------------------------------------- #
# Observation serialization limits
# --------------------------------------------------------------------------- #
MAX_AXTREE_CHARS_SNAPSHOT = int(_cfg("obs", "max_axtree_chars_snapshot", default=6000))
MAX_AXTREE_CHARS_POLICY = int(_cfg("obs", "max_axtree_chars_policy", default=12000))
POLICY_HISTORY_STEPS = int(_cfg("obs", "policy_history_steps", default=9))


# --------------------------------------------------------------------------- #
# Fingerprint tolerances
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FingerprintTolerance:
    text_jaccard: float = 0.05  # allowed visible-text drift when checking restore


DEFAULT_TOL = FingerprintTolerance(
    text_jaccard=float(_cfg("fingerprint", "text_jaccard", default=0.05)))


# --------------------------------------------------------------------------- #
# Training (used by revact.train.sft / revact.eval.decisions)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainConfig:
    model_path: str = _cfg("train", "model_path",
                           default="/workspace/models/Qwen2.5-3B-Instruct")
    max_len: int = int(_cfg("train", "max_len", default=4096))
    lora_r: int = int(_cfg("train", "lora_r", default=16))
    lora_alpha: int = int(_cfg("train", "lora_alpha", default=32))
    lora_dropout: float = float(_cfg("train", "lora_dropout", default=0.05))
    epochs: float = float(_cfg("train", "epochs", default=3))
    lr: float = float(_cfg("train", "lr", default=1e-4))
    batch_size: int = int(_cfg("train", "batch_size", default=2))
    grad_accum: int = int(_cfg("train", "grad_accum", default=4))
    output_dir: str = _cfg("train", "output_dir",
                           default=str(OUTPUTS_DIR / "sft_lora"))


TRAIN = TrainConfig()

# --------------------------------------------------------------------------- #
# Teacher distillation (S7)
# --------------------------------------------------------------------------- #
DISTILL_MODEL = os.environ.get(
    "REVACT_DISTILL_MODEL", _cfg("distill", "model", default="deepseek-chat"))
DISTILL_BASE_URL = os.environ.get(
    "REVACT_DISTILL_BASE_URL",
    _cfg("distill", "base_url", default="https://api.deepseek.com/v1"))
DISTILL_KEY_ENV = os.environ.get(
    "REVACT_DISTILL_KEY_ENV",
    _cfg("distill", "api_key_env", default="DEEPSEEK_API_KEY"))
DISTILL_MAX_RETRIES = int(_cfg("distill", "max_retries", default=3))


def ensure_dirs() -> None:
    for d in (RAW_TRAJ_DIR, STATE_BANK_DIR, SCREENSHOT_DIR, GROUNDED_DIR,
              SFT_PATH.parent, DPO_PATH.parent, SPLITS_DIR):
        d.mkdir(parents=True, exist_ok=True)
