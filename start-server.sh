#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")"
exec ./.venv/bin/python -m mlx_audio.server