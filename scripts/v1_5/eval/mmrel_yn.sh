#!/usr/bin/env bash
set -euo pipefail

# Evaluate models on MMRel Discriminative_YN_QA.
#
# Required:
#   DATA_ROOT=/path/to/mmrel
#
# Optional:
#   CKPT_ROOT=/path/to/checkpoints
#   ANSWERS_ROOT=/path/to/answers
#   CUDA_VISIBLE_DEVICES=0
#   MODEL_KEYS="baseline ours"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${DATA_ROOT:-./playground/data/eval/mmrel}"
QA_DIR="${QA_DIR:-${DATA_ROOT}/json/Evaluation/Discriminative_YN_QA}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${DATA_ROOT}}"
ANSWERS_ROOT="${ANSWERS_ROOT:-${DATA_ROOT}/answers}"
CKPT_ROOT="${CKPT_ROOT:-./checkpoints}"

SUBSETS=(
  dall-e_action_all
  dall-e_spatial_all
  spec_comparative_all
  spec_spatial_all
  vg_action_all
  vg_comparative_all
  vg_spatial_all
)

declare -A MODELS
MODELS[baseline]="${BASELINE_MODEL:-liuhaotian/llava-v1.5-7b}"
MODELS[ours]="${TGIF_MODEL:-${CKPT_ROOT}/llava-v1.5-7b-layer_selector-none}"
MODELS[baseline_13b]="${BASELINE_13B_MODEL:-liuhaotian/llava-v1.5-13b}"
MODELS[ours_13b]="${TGIF_13B_MODEL:-${CKPT_ROOT}/llava-v1.5-13b-layer_selector-none}"
MODELS[baseline_siglip]="${BASELINE_SIGLIP_MODEL:-${CKPT_ROOT}/llava-v1.5-7b-mlp2x_gelu-none-google-siglip-large-patch16-384}"
MODELS[ours_siglip]="${TGIF_SIGLIP_MODEL:-${CKPT_ROOT}/llava-v1.5-7b-layer_selector-none-google-siglip-large-patch16-384}"

MODEL_KEYS="${MODEL_KEYS:-baseline ours}"

for MODEL_KEY in ${MODEL_KEYS}; do
  MODEL_PATH="${MODELS[$MODEL_KEY]:-}"
  if [ -z "$MODEL_PATH" ]; then
    echo "Unknown MODEL_KEY: $MODEL_KEY" >&2
    exit 2
  fi

  ANSWERS_DIR="${ANSWERS_ROOT}/${MODEL_KEY}"
  mkdir -p "$ANSWERS_DIR"

  echo "=========================================="
  echo "Model: ${MODEL_KEY} (${MODEL_PATH})"
  echo "=========================================="

  for SUBSET in "${SUBSETS[@]}"; do
    QUESTION_FILE="${QA_DIR}/${SUBSET}.jsonl"
    ANSWERS_FILE="${ANSWERS_DIR}/${SUBSET}.jsonl"

    if [ ! -f "$QUESTION_FILE" ]; then
      echo "Missing question file: $QUESTION_FILE" >&2
      exit 2
    fi

    if [ -f "$ANSWERS_FILE" ]; then
      echo "  --> ${SUBSET} (cached)"
      continue
    fi

    echo "  --> ${SUBSET}"
    python -m llava.eval.model_vqa_mmrel \
      --model-path "$MODEL_PATH" \
      --image-folder "$IMAGE_FOLDER" \
      --question-file "$QUESTION_FILE" \
      --answers-file "$ANSWERS_FILE" \
      --conv-mode "${CONV_MODE:-vicuna_v1}" \
      --temperature "${TEMPERATURE:-0}"
  done

  echo ""
  echo "Results for ${MODEL_KEY}:"
  python -m llava.eval.eval_mmrel_yn --results-dir "$ANSWERS_DIR"
  echo ""
done
