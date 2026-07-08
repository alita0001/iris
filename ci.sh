#!/usr/bin/env bash
# Minimal CI: lint + offline test suite. No env, no key, no GPU needed.
set -e
cd "$(dirname "$0")"
echo "== ruff =="
python3 -m ruff check revact/ tests/
echo "== pytest (offline) =="
python3 -m pytest
echo "CI OK"
