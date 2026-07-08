"""RevAct/IRIS: grounded-reversibility data pipeline for safe web agents.

Layout:
  envs/       env harness, fingerprint, obs utils, mock env, webarena patch
  grounding/  execute-then-undo probes, signals, undo controllers, registry
  data/       collect, reach, scale, assemble, splits
  train/      LoRA SFT, teacher conditional distillation
  eval/       decision accuracy (calibration/rollouts to come)
  cli.py      unified entry point:  python -m revact.cli <command>

Core modules are stdlib-importable; heavy deps (browsergym, torch) load
lazily inside the code paths that need them.
"""

__version__ = "0.2.0"
__all__ = ["config", "policies", "envs", "grounding", "data", "train", "eval", "cli"]
