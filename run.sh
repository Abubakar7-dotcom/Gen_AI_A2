#!/usr/bin/env bash
# One-command start: creates the virtual environment on first run, then launches the app and
# opens http://127.0.0.1:7860 in the browser. Extra arguments go to the app, e.g. ./run.sh --no-draft
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "First start: creating .venv and installing requirements (a few minutes)..."
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
  else
    PY=$(command -v python3.11 || command -v python3.12 || command -v python3.13 || true)
    if [ -z "$PY" ]; then echo "Python 3.11+ is required (brew install python@3.11, or install uv)."; exit 1; fi
    "$PY" -m venv .venv && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -r requirements.txt
  fi
fi
exec .venv/bin/python -m app.app "$@"
