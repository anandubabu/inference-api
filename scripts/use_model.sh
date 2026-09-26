#!/usr/bin/env bash
# Drop in your own ONNX file and (re)start the API against it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:-}"
if [[ -z "$SRC" ]]; then
  echo "Usage: ./scripts/use_model.sh /path/to/your_model.onnx" >&2
  exit 1
fi
if [[ ! -f "$SRC" ]]; then
  echo "File not found: $SRC" >&2
  exit 1
fi
mkdir -p "$ROOT/models"
cp -f "$SRC" "$ROOT/models/model.onnx"
echo "Installed $SRC -> models/model.onnx"
echo "Start with: make run   or   docker compose up --build"
