#!/usr/bin/env bash
#
# finetune_from_hf.sh - download a pretrained projector from HF and run a RESUMABLE finetune.
#
# Usage:
#   bash scripts/exp/finetune_from_hf.sh
#   # Crashed mid-run? Just run the SAME command again; it auto-resumes from the last checkpoint
#   # (train.py:1006 resumes when output_dir already contains checkpoint-*).
#
# Common env overrides:
#   GPUS=0,1,2,3,4,5,6,7   MASTER_PORT=29500   PER_DEVICE_BS=16   GRAD_ACCUM=1
#   SAVE_STEPS=500   SAVE_TOTAL_LIMIT=2   RUN_EVAL=1   WANDB_MODE=offline
#
set -euo pipefail
set -x

PROJECTOR="layer_selector"
if [ "${1:-layer_selector}" != "layer_selector" ]; then
  echo "ERROR: this public TGIF release only supports mm_projector_type=layer_selector." >&2
  exit 2
fi
CONFIG_TAG="${CONFIG_TAG:-none}"
RUN_TAG="${RUN_TAG:-google-siglip-large-patch16-384}"
HF_REPO="${HF_REPO:-cclinn/llava-siglip-pretrain-projectors}"
BASE_MODEL="${BASE_MODEL:-lmsys/vicuna-7b-v1.5}"
VISION_TOWER="${VISION_TOWER:-google/siglip-large-patch16-384}"

GPUS="${GPUS:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-29500}"
PER_DEVICE_BS="${PER_DEVICE_BS:-16}"             # 16 x 4gpu x ACCUM 2 = global 128 (recipe). On 8 GPUs: BS=16 ACCUM=1.
GRAD_ACCUM="${GRAD_ACCUM:-2}"
SAVE_STEPS="${SAVE_STEPS:-500}"                  # checkpoint cadence (~25 min); resume loses at most this much
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"        # full-7B zero3 ckpt ~100GB each -> watch disk
EVAL_GPU="${EVAL_GPU:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# activate the llava env if it isn't already importable (skip with SKIP_ENV=1)
if [ -z "${SKIP_ENV:-}" ] && ! python -c "import llava" >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || echo /raid/miniconda3)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh" && conda activate "${CONDA_ENV:-llava}"
fi

PRETRAIN_NAME="llava-v1.5-7b-pretrain-${PROJECTOR}-${CONFIG_TAG}-${RUN_TAG}"
PROJECTOR_PATH="./checkpoints/${PRETRAIN_NAME}/mm_projector.bin"
OUTPUT_DIR="./checkpoints/llava-v1.5-7b-${PROJECTOR}-${CONFIG_TAG}-${RUN_TAG}"
CKPT_NAME="llava-v1.5-7b-${PROJECTOR}-${CONFIG_TAG}-${RUN_TAG}"

# --- disk sanity (each zero3 7B checkpoint ~100GB; need room for SAVE_TOTAL_LIMIT + 1 during rotation) ---
FREE_GB="$(df -P "$REPO_ROOT" | awk 'NR==2{print int($4/1024/1024)}')"
NEED_GB="$(( (SAVE_TOTAL_LIMIT + 1) * 100 ))"
echo ">>> free disk: ${FREE_GB} GB | est. checkpoint need (~100GB x $((SAVE_TOTAL_LIMIT+1))): ${NEED_GB} GB"
[ "$FREE_GB" -lt "$NEED_GB" ] && \
  echo "WARNING: low disk — free space or lower SAVE_TOTAL_LIMIT, or checkpoint saves may fail." >&2

# --- 1) fetch the pretrained projector from HF (idempotent: skips if already local) ---
if [ -f "$PROJECTOR_PATH" ]; then
  echo ">>> projector already present: $PROJECTOR_PATH (skip download)"
else
  echo ">>> downloading ${PRETRAIN_NAME}/mm_projector.bin from ${HF_REPO}"
  python - "$HF_REPO" "$PRETRAIN_NAME" <<'PY'
import sys
from huggingface_hub import hf_hub_download
repo, name = sys.argv[1], sys.argv[2]
p = hf_hub_download(repo_id=repo, filename=f"{name}/mm_projector.bin", local_dir="./checkpoints")
print("downloaded ->", p)
PY
fi
[ -f "$PROJECTOR_PATH" ] || { echo "ERROR: $PROJECTOR_PATH missing after download"; exit 1; }

# --- 2) resumable finetune (re-run this script to resume) ---
if compgen -G "${OUTPUT_DIR}/checkpoint-*" >/dev/null 2>&1; then
  echo ">>> found existing checkpoints in ${OUTPUT_DIR} -> RESUMING"
fi

deepspeed --include "localhost:${GPUS}" --master_port "${MASTER_PORT}" llava/train/train_mem.py \
  --deepspeed ./scripts/zero3.json \
  --model_name_or_path "${BASE_MODEL}" \
  --version v1 \
  --data_path ./playground/data/llava_v1_5_mix665k.json \
  --image_folder ./playground/data \
  --vision_tower "${VISION_TOWER}" \
  --pretrain_mm_mlp_adapter "${PROJECTOR_PATH}" \
  --mm_projector_type "${PROJECTOR}" \
  --layer_selector_alpha 0.0 \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size "${PER_DEVICE_BS}" \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to "${REPORT_TO:-wandb}"

# --- 3) optional POPE eval (set RUN_EVAL=1) ---
if [ -n "${RUN_EVAL:-}" ]; then
  echo ">>> POPE eval for ${PROJECTOR}"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" bash scripts/exp/eval/pope.sh "${OUTPUT_DIR}" "${CKPT_NAME}"
fi
echo ">>> done."
