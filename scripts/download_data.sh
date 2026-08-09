#!/usr/bin/env bash
#
# download_data.sh - fetch & prepare LLaVA-1.5 data (pretrain / finetune / POPE eval).
#
# Usage:
#   bash scripts/download_data.sh [all|pretrain|finetune|eval]   # default: all
#   DATA_ROOT=/path/to/data bash scripts/download_data.sh all    # default DATA_ROOT=./playground/data
#
# Notes:
#   * Idempotent: re-running skips datasets that are already extracted.
#   * Uses aria2c (multi-stream) when available, else falls back to wget.
#   * Per-host connection limits are baked in (downloads.cs.stanford.edu / GQA
#     refuses >1-2 parallel connections with HTTP 503, so it is fetched single-stream).
#
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./playground/data}"
WHAT="${1:-all}"

have(){ command -v "$1" >/dev/null 2>&1; }
log(){ printf '\n==> %s\n' "$*"; }
die(){ printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# present DIR MIN -> true if DIR already holds >= MIN files (used to skip finished datasets)
present(){ [ -d "$1" ] && [ "$(find "$1" -type f 2>/dev/null | head -n "$2" | wc -l)" -ge "$2" ]; }

# fetch URL OUTFILE [CONN] -> resumable download (aria2c multi-stream, else wget single)
fetch(){
  local url="$1" out="$2" conn="${3:-16}"
  mkdir -p "$(dirname "$out")"
  if have aria2c; then
    aria2c -c -x"$conn" -s"$conn" -k1M --max-tries=10 --retry-wait=5 \
           --auto-file-renaming=false --allow-overwrite=true \
           -d "$(dirname "$out")" -o "$(basename "$out")" "$url"
  else
    wget -c --tries=10 --timeout=120 -O "$out" "$url"
  fi
}

# unpack ARCHIVE DESTDIR -> verify integrity, extract, delete archive (zip or tar)
unpack(){
  local arc="$1" dest="$2"
  mkdir -p "$dest"
  case "$arc" in
    *.zip)
      unzip -tq "$arc" >/dev/null || die "corrupt archive: $arc (delete and re-run)"
      unzip -q -o "$arc" -d "$dest" ;;
    *.tar|*.tar.*)
      tar -tf "$arc" >/dev/null || die "corrupt archive: $arc (delete and re-run)"
      tar -xf "$arc" -C "$dest" ;;
    *) die "unknown archive type: $arc" ;;
  esac
  rm -f "$arc"
}

preflight(){
  for t in curl unzip tar find; do have "$t" || die "missing required tool: $t"; done
  have aria2c || have wget || die "need either aria2c or wget"
  have aria2c || log "aria2c not found -> using wget (slower, single-stream)."
}

# ---------------------------------------------------------------------------

pretrain(){
  log "PRETRAIN (LLaVA-Pretrain, 558K)"
  local d="$DATA_ROOT/LLaVA-Pretrain"
  local base="https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main"
  mkdir -p "$d"
  [ -f "$d/blip_laion_cc_sbu_558k.json" ] || fetch "$base/blip_laion_cc_sbu_558k.json" "$d/blip_laion_cc_sbu_558k.json" 16
  if present "$d/images" 550000; then log "pretrain images present, skip"; else
    fetch "$base/images.zip" "$d/images.zip" 16
    unpack "$d/images.zip" "$d"          # zip contains images/<shard>/... -> $d/images
  fi
}

finetune(){
  log "FINETUNE (llava_v1_5_mix665k + images)"
  mkdir -p "$DATA_ROOT"
  [ -f "$DATA_ROOT/llava_v1_5_mix665k.json" ] || \
    fetch "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json" \
          "$DATA_ROOT/llava_v1_5_mix665k.json" 16

  # COCO train2017 (zip contains train2017/)
  present "$DATA_ROOT/coco/train2017" 110000 || {
    fetch "http://images.cocodataset.org/zips/train2017.zip" "$DATA_ROOT/coco_train2017.zip" 16
    unpack "$DATA_ROOT/coco_train2017.zip" "$DATA_ROOT/coco"; }

  # GQA (zip contains images/) -- THIS HOST REFUSES PARALLEL CONNS (503): single-stream only
  present "$DATA_ROOT/gqa/images" 140000 || {
    fetch "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip" "$DATA_ROOT/gqa_images.zip" 1
    unpack "$DATA_ROOT/gqa_images.zip" "$DATA_ROOT/gqa"; }

  # OCR-VQA (HF tar contains ocr_vqa/images/) -- legacy amazon patch image URL is dead; not needed
  present "$DATA_ROOT/ocr_vqa/images" 200000 || {
    fetch "https://huggingface.co/datasets/ej2/llava-ocr-vqa/resolve/main/ocr_vqa.tar?download=true" "$DATA_ROOT/ocr_vqa.tar" 16
    unpack "$DATA_ROOT/ocr_vqa.tar" "$DATA_ROOT"; }

  # TextVQA (zip contains train_images/)
  present "$DATA_ROOT/textvqa/train_images" 24000 || {
    fetch "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip" "$DATA_ROOT/textvqa_train_val.zip" 16
    unpack "$DATA_ROOT/textvqa_train_val.zip" "$DATA_ROOT/textvqa"; }

  # Visual Genome (cs.stanford.edu tolerates a few conns; zips contain VG_100K / VG_100K_2)
  present "$DATA_ROOT/vg/VG_100K" 60000 || {
    fetch "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip" "$DATA_ROOT/vg_images.zip" 8
    unpack "$DATA_ROOT/vg_images.zip" "$DATA_ROOT/vg"; }
  present "$DATA_ROOT/vg/VG_100K_2" 40000 || {
    fetch "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip" "$DATA_ROOT/vg_images2.zip" 8
    unpack "$DATA_ROOT/vg_images2.zip" "$DATA_ROOT/vg"; }
}

eval_pope(){
  log "EVAL (POPE)"
  local e="$DATA_ROOT/eval"
  if [ ! -d "$e/pope" ]; then
    have gdown || python -m pip install gdown || die "could not install gdown (needed for the Google Drive eval.zip)"
    gdown "1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy" -O "$DATA_ROOT/eval.zip"   # bare ID (this gdown lacks --fuzzy)
    unpack "$DATA_ROOT/eval.zip" "$e"                                    # zip has dataset folders at top -> eval/pope, ...
  else
    log "eval/ present, skip eval.zip"
  fi
  # POPE coco label files are NOT in eval.zip -> fetch from the official POPE repo
  mkdir -p "$e/pope/coco"
  for c in adversarial popular random; do
    [ -f "$e/pope/coco/coco_pope_$c.json" ] || \
      curl -fsSL "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_$c.json" \
           -o "$e/pope/coco/coco_pope_$c.json"
  done
  # COCO val2014 images (zip contains val2014/)
  present "$e/pope/val2014" 40000 || {
    fetch "http://images.cocodataset.org/zips/val2014.zip" "$e/pope/val2014.zip" 16
    unpack "$e/pope/val2014.zip" "$e/pope"; }
}

verify(){
  log "VERIFY (image counts)"
  for d in LLaVA-Pretrain/images coco/train2017 gqa/images ocr_vqa/images \
           textvqa/train_images vg/VG_100K vg/VG_100K_2 eval/pope/val2014; do
    if [ -d "$DATA_ROOT/$d" ]; then
      printf "  %-26s %s\n" "$d" "$(find "$DATA_ROOT/$d" -type f \( -iname '*.jpg' -o -iname '*.png' \) | wc -l)"
    fi
  done
}

# ---------------------------------------------------------------------------

preflight
log "DATA_ROOT=$DATA_ROOT   target=$WHAT"
case "$WHAT" in
  pretrain) pretrain ;;
  finetune) finetune ;;
  eval)     eval_pope ;;
  all)      pretrain; finetune; eval_pope ;;
  *) die "usage: $0 [all|pretrain|finetune|eval]" ;;
esac
verify
log "done."
