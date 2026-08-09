#!/usr/bin/env bash
set -euo pipefail

# Run VLMEvalKit-backed TGIF benchmarks such as HallusionBench and OCRBench.
#
# Required:
#   MODEL_PATH=/path/or/hf-id/to/tgif-checkpoint
#
# Optional:
#   VLMEVALKIT_ROOT=/path/to/VLMEvalKit
#   MODEL_NAME=tgif
#   DATASETS="HallusionBench OCRBench"
#   WORK_DIR=./outputs/vlmeval
#   CUDA_VISIBLE_DEVICES=0
#
# This script assumes the current TGIF repository is importable as `llava`.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:-${1:-}}"
if [ -z "$MODEL_PATH" ]; then
  echo "Usage: MODEL_PATH=<checkpoint> bash scripts/eval_vlmevalkit.sh" >&2
  echo "   or: bash scripts/eval_vlmevalkit.sh <checkpoint>" >&2
  exit 2
fi

VLMEVALKIT_ROOT="${VLMEVALKIT_ROOT:-../VLMEvalKit}"
MODEL_NAME="${MODEL_NAME:-tgif}"
DATASETS="${DATASETS:-HallusionBench OCRBench}"
WORK_DIR="${WORK_DIR:-./outputs/vlmeval}"
CONFIG_PATH="${CONFIG_PATH:-/tmp/tgif_vlmevalkit_config.json}"

if [ ! -f "${VLMEVALKIT_ROOT}/run.py" ]; then
  echo "Missing VLMEvalKit checkout: ${VLMEVALKIT_ROOT}" >&2
  echo "Set VLMEVALKIT_ROOT=/path/to/VLMEvalKit." >&2
  exit 2
fi

python - "$CONFIG_PATH" "$MODEL_NAME" "$MODEL_PATH" $DATASETS <<'PY'
import json
import sys

config_path, model_name, model_path, *datasets = sys.argv[1:]

dataset_classes = {
    "HallusionBench": "ImageYORNDataset",
    "OCRBench": "OCRBench",
    "TextVQA_VAL": "ImageVQADataset",
    "MMBench_DEV_EN": "ImageMCQDataset",
    "ScienceQA_VAL": "ImageMCQDataset",
    "GQA_TestDev_Balanced": "ImageVQADataset",
}

unknown = [name for name in datasets if name not in dataset_classes]
if unknown:
    raise SystemExit(f"Unsupported DATASETS entries: {unknown}")

cfg = {
    "model": {
        model_name: {
            "class": "LLaVA",
            "model_path": model_path,
        }
    },
    "data": {
        name: {
            "class": dataset_classes[name],
            "dataset": name,
        }
        for name in datasets
    },
}

with open(config_path, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
print(config_path)
PY

PYTHONPATH="${REPO_ROOT}:${VLMEVALKIT_ROOT}:${PYTHONPATH:-}" \
  python "${VLMEVALKIT_ROOT}/run.py" \
    --config "$CONFIG_PATH" \
    --work-dir "$WORK_DIR" \
    --verbose
