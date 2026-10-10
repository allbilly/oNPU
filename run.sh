#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo 'Install first: uv sync --python ~/tinygrad/.venv/bin/python'
  exit 1
fi
exec .venv/bin/python -m onpu "$@"
