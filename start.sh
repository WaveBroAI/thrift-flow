#!/usr/bin/env bash
# Start thrift-flow proxy
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f ".venv/bin/python" ]]; then
    echo "[thrift-flow] Creating virtualenv..."
    python3 -m venv .venv
    .venv/bin/pip install -q -r requirements.txt
fi

if [[ ! -f ".env" && -f ".env.example" ]]; then
    echo "[thrift-flow] WARNING: .env not found. Copy .env.example to .env and add your API keys."
    exit 1
fi

exec .venv/bin/python main.py "$@"
