#!/bin/bash

MODEL_PATH="${MODEL_PATH:-${1:-./checkpoints/llava-v1.5-7b-layer_selector-none}}"
CKPT_NAME="${CKPT_NAME:-${2:-llava-v1.5-7b-layer_selector-none}}"

python -m llava.eval.model_vqa_science \
    --model-path "$MODEL_PATH" \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file "./playground/data/eval/scienceqa/answers/${CKPT_NAME}.jsonl" \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file "./playground/data/eval/scienceqa/answers/${CKPT_NAME}.jsonl" \
    --output-file "./playground/data/eval/scienceqa/answers/${CKPT_NAME}_output.jsonl" \
    --output-result "./playground/data/eval/scienceqa/answers/${CKPT_NAME}_result.json"
