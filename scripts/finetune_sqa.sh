#!/bin/bash

CACHE_ROOT="${OCEAN_CACHE_ROOT:-/ocean/projects/cis250047p/clin14}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/.triton/autotune}"
export TMPDIR="${TMPDIR:-$CACHE_ROOT/tmp}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$CACHE_ROOT/.cache/torch_extensions}"
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR" "$HF_HOME" "$TORCH_EXTENSIONS_DIR"

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-13b-v1.3 \
    --version $PROMPT_VERSION \
    --data_path /Data/ScienceQA/data/scienceqa/llava_train_QCM-LEA.json \
    --image_folder /Data/ScienceQA/data/scienceqa/images/train \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./checkpoints/huggingface/liuhaotian/llava-pretrain-vicuna-13b-v1.3/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-vicuna-13b-v1.3-pretrain_lcs558k_plain-ScienceQA_QCM_LEA-12e \
    --num_train_epochs 12 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
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
    --report_to wandb
