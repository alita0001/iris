"""``python -m revact.server`` — start the dataset workbench."""
from __future__ import annotations

import argparse
import sys

from .app import serve


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="revact.server",
                                description="IRIS dataset workbench server")
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address (default local-only)")
    p.add_argument("--port", type=int, default=7788)
    args = p.parse_args(argv)
    return serve(host=args.host, port=args.port)


if __name__ == "__main__":
    sys.exit(main())
