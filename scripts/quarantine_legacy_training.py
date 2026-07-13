#!/usr/bin/env python3
"""Index legacy training rows into a reversible, source-preserving quarantine."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from revact.data.quarantine import write_quarantine_index  # noqa: E402  # script bootstraps repository import path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(json.dumps(write_quarantine_index(args.data_root), ensure_ascii=False,
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
