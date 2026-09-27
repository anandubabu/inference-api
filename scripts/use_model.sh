#!/usr/bin/env bash
# Drop in a compatible ONNX anomaly model (float32 [N,F], score_samples output).
# Also update models/threshold.json / model_meta.json to match.
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
echo "Ensure threshold.json / model_meta.json match this model, then: make run"
