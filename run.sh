#!/usr/bin/env bash
# tiny-alexa — one-command run.
#
# Creates a virtualenv (once), installs the minimal runtime dependencies
# (NO torch, NO training deps) and starts the browser voice-control server.
#
#   ./run.sh                 # start on http://localhost:8321
#   ./run.sh --debug         # forward any flag to src/server.py
#
# First run takes ~30-60 s (dependency install); later runs start instantly.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
VENV=".venv"

echo "▸ tiny-alexa — one-command run"

# 1. virtualenv (created once, reused afterwards)
if [ ! -d "$VENV" ]; then
  echo "▸ creating $VENV"
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# 2. minimal runtime dependencies (browser UI + ONNX inference + wake word)
echo "▸ installing runtime dependencies"
pip install --quiet --upgrade pip
pip install --quiet \
  "numpy" "soundfile" "onnxruntime" "openwakeword==0.4.0" "aiohttp>=3.9"

# 3. launch the server, forwarding any extra flags
echo "▸ starting tiny-alexa → http://localhost:8321   (Ctrl-C to stop)"
exec python src/server.py "$@"
