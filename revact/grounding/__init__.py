"""Grounding layer: execute-then-undo reversibility measurement.

Importing :mod:`revact.grounding.probes` registers all site probes.
"""
from .base import (DESTRUCTIVE, NON_DESTRUCTIVE, SELF_RECOVERING,  # noqa: F401
                   ProbeContext, ProbeSpec, ReversibilityResult,
                   destructive_allowed, get_probe, list_probes,
                   load_reversibility, mk_result, run_probe, save_results)
