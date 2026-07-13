"""Content-addressed, immutable prompt and generation provenance bundles.

``prompts_fp`` is useful only if the corresponding full text remains
recoverable.  Bundles live below ``data/manifests/prompts`` by default and are
created with exclusive semantics: an existing fingerprint can never be
silently replaced with different content.

Prompt text and generation configuration intentionally have separate
identities.  Identical text keeps its backwards-compatible ``prompts_fp``;
changing the producer, model, or decode settings creates a distinct
``prompt_generation_fp`` under ``manifests/prompt_generations``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config


SCHEMA_VERSION = "iris.prompts.v2"
LEGACY_SCHEMA_VERSION = "iris.prompts.v1"
GENERATION_SCHEMA_VERSION = "iris.prompt-generation.v1"
_FP_RE = re.compile(r"^[0-9a-f]{12}$")
_GENERATION_FP_RE = re.compile(r"^[0-9a-f]{16}$")


def canonical_blob(effective_prompts: dict[str, Any]) -> str:
    # Byte-for-byte compatible with prompts_fp values materialized before the
    # generation-provenance split.
    return json.dumps(effective_prompts, ensure_ascii=False, sort_keys=True)


def content_fingerprint(effective_prompts: dict[str, Any]) -> str:
    return hashlib.sha1(
        canonical_blob(effective_prompts).encode("utf-8")).hexdigest()[:12]


def bundle_dir(root: Path | None = None) -> Path:
    override = os.environ.get("REVACT_PROMPTS_STORE", "")
    if override:
        return Path(override)
    data_root = Path(root) if root is not None else config.DATA_ROOT
    return data_root / "manifests" / "prompts"


def generation_bundle_dir(root: Path | None = None) -> Path:
    override = os.environ.get("REVACT_PROMPT_GENERATIONS_STORE", "")
    if override:
        return Path(override)
    data_root = Path(root) if root is not None else config.DATA_ROOT
    return data_root / "manifests" / "prompt_generations"


def store_bundle(effective_prompts: dict[str, Any], *, root: Path | None = None,
                 parent_fp: str = "", author: str = "pipeline",
                 model_config: dict | None = None) -> Path:
    """Persist prompt text once, independently of generation configuration.

    ``model_config`` used to be stored in a file addressed only by prompt
    text, which silently lost later configurations.  Refuse that ambiguous
    call and direct callers to :func:`store_generation_bundle`.
    """
    if model_config is not None:
        raise ValueError(
            "model_config is generation provenance; use "
            "store_generation_bundle() so it participates in the fingerprint")
    fp = content_fingerprint(effective_prompts)
    directory = bundle_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{fp}.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fp,
        "parent_fp": parent_fp,
        "author": author,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompts": effective_prompts,
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fp \
                or canonical_blob(existing.get("prompts") or {}) != \
                canonical_blob(effective_prompts):
            raise RuntimeError(f"immutable prompt bundle collision at {path}")
        return path
    try:
        with path.open("x", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
    except FileExistsError:  # another worker won the race; verify it
        return store_bundle(effective_prompts, root=root, parent_fp=parent_fp,
                            author=author)
    return path


def load_bundle(fp: str, *, root: Path | None = None) -> dict:
    if not _FP_RE.fullmatch(fp or ""):
        raise ValueError("prompt fingerprint must be 12 lowercase hex characters")
    path = bundle_dir(root) / f"{fp}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in {
            SCHEMA_VERSION, LEGACY_SCHEMA_VERSION}:
        raise ValueError(f"unsupported prompt bundle schema: {path}")
    actual = content_fingerprint(payload.get("prompts") or {})
    if payload.get("fingerprint") != fp or actual != fp:
        raise ValueError(f"prompt bundle failed integrity check: {path}")
    return payload


def diff_bundles(left_fp: str, right_fp: str, *, root: Path | None = None) -> dict:
    """Exact prompt-id diff for workbench/A-B audit output."""
    left = load_bundle(left_fp, root=root).get("prompts") or {}
    right = load_bundle(right_fp, root=root).get("prompts") or {}
    changed = {}
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            changed[key] = {"before": left.get(key), "after": right.get(key)}
    return {"left_fp": left_fp, "right_fp": right_fp,
            "n_changed": len(changed), "changed": changed}


def _generation_identity(*, prompts_fp: str, producer: str,
                         model: str | dict, decode_config: dict) -> dict:
    if not _FP_RE.fullmatch(prompts_fp or ""):
        raise ValueError("prompts_fp must be 12 lowercase hex characters")
    if not isinstance(producer, str) or not producer.strip():
        raise ValueError("producer must be a non-empty string")
    if not ((isinstance(model, str) and model.strip()) or
            (isinstance(model, dict) and model)):
        raise ValueError("model must be a non-empty string or object")
    if not isinstance(decode_config, dict):
        raise ValueError("decode_config must be an object")
    identity = {
        "prompts_fp": prompts_fp,
        "producer": producer,
        "model": model,
        "decode_config": decode_config,
    }
    # Fail before hashing values that are not portable JSON (including NaN).
    json.dumps(identity, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return identity


def generation_fingerprint(*, prompts_fp: str, producer: str,
                           model: str | dict, decode_config: dict) -> str:
    """Hash every generation setting that can change an artifact."""
    identity = _generation_identity(
        prompts_fp=prompts_fp, producer=producer, model=model,
        decode_config=decode_config)
    blob = json.dumps(identity, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def store_generation_bundle(*, prompts_fp: str, producer: str,
                            model: str | dict, decode_config: dict,
                            root: Path | None = None) -> Path:
    """Persist immutable generation provenance linked to a prompt bundle."""
    # Prohibit orphan generation records.
    load_bundle(prompts_fp, root=root)
    identity = _generation_identity(
        prompts_fp=prompts_fp, producer=producer, model=model,
        decode_config=decode_config)
    fp = generation_fingerprint(**identity)
    directory = generation_bundle_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{fp}.json"
    payload = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "fingerprint": fp,
        **identity,
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"immutable generation bundle collision at {path}")
        return path
    encoded = json.dumps(payload, ensure_ascii=False, indent=2,
                         sort_keys=True, allow_nan=False) + "\n"
    try:
        with path.open("x", encoding="utf-8") as fh:
            fh.write(encoded)
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"immutable generation bundle collision at {path}")
    return path


def load_generation_bundle(fp: str, *, root: Path | None = None) -> dict:
    """Load and verify generation config plus its full prompt bundle."""
    if not _GENERATION_FP_RE.fullmatch(fp or ""):
        raise ValueError(
            "prompt generation fingerprint must be 16 lowercase hex characters")
    path = generation_bundle_dir(root) / f"{fp}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != GENERATION_SCHEMA_VERSION:
        raise ValueError(f"unsupported prompt generation schema: {path}")
    identity = _generation_identity(
        prompts_fp=payload.get("prompts_fp", ""),
        producer=payload.get("producer", ""),
        model=payload.get("model"),
        decode_config=payload.get("decode_config"),
    )
    actual = generation_fingerprint(**identity)
    if payload.get("fingerprint") != fp or actual != fp:
        raise ValueError(f"prompt generation bundle failed integrity check: {path}")
    return {**payload,
            "prompt_bundle": load_bundle(identity["prompts_fp"], root=root)}
