#!/bin/sh
set -eu
# Run from repository root. Explicit opt-in per stack; model downloads are separate.
stack=${1:-core}
case "$stack" in core) env=.venv;; audio|moshi|laya) env=.venv-$stack;; *) echo 'usage: sh scripts/setup.sh [core|audio|moshi|laya]' >&2; exit 2;; esac
uv venv --python 3.12 "$env"
uv pip sync --python "$env/bin/python" "requirements/$stack.txt"
if [ "$stack" = core ]; then uv pip install --python "$env/bin/python" --no-deps -e .; fi
