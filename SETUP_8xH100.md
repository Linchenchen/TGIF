# TGIF Setup (8xH100)

Minimal setup to finetune from a pretrained projector (hosted on HF) + run POPE eval on an
8×H100 (80 GB) node. Pretraining is skipped — the projector is downloaded automatically.

## Requirements
- 8× H100 (80 GB), NVIDIA driver supporting CUDA 12.x
- conda
- ~400 GB free disk: finetune + eval images (~100 GB) plus ZeRO-3 checkpoints (~100 GB each)

## 1. Environment
```bash
git clone https://github.com/Linchenchen/TGIF.git
cd TGIF
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip
pip install -e ".[train]"
pip install peft==0.10.0 transformers==4.37.2     # peft pin avoids a clear_device_cache import crash
```

## 2. Data
Finetuning skips pretrain (the projector comes from HF), so you only need the finetune + eval data:
```bash
bash scripts/download_data.sh finetune     # llava_v1_5_mix665k.json + coco/gqa/ocr_vqa/textvqa/vg images
bash scripts/download_data.sh eval         # POPE: questions, labels, val2014 images
```
Produces under `./playground/data/`: `coco/train2017`, `gqa/images`, `ocr_vqa/images`,
`textvqa/train_images`, `vg/VG_100K(+_2)`, `eval/pope/`.
(Run `bash scripts/download_data.sh all` instead only if you also intend to pretrain from scratch.)

## 3. Finetune + eval (resumable)
`scripts/exp/finetune_from_hf.sh` downloads the pretrained projector from HF, then runs a
**resumable** finetune — it checkpoints periodically and **auto-resumes on re-run**
(`train.py` resumes when `output_dir` already contains `checkpoint-*`). On 8×H100:

```bash
GPUS=0,1,2,3,4,5,6,7 GRAD_ACCUM=1 RUN_EVAL=1 WANDB_MODE=offline \
  bash scripts/exp/finetune_from_hf.sh layer_selector      # or: mlp2x_gelu
```

- **Resume:** interrupted (crash / preemption)? Run the **exact same line again** — it continues from the last checkpoint. No flags needed.
- **Batch:** `GRAD_ACCUM=1` → global batch 16 × 8 × 1 = 128 (the recipe). The script auto-detects the repo root and activates the `llava` env, so no path editing is required.
- **wandb:** `WANDB_MODE=offline` skips the login prompt; drop it (and run `wandb login`) for live logging. `RUN_EVAL=1` runs POPE after finetune.
- **Disk:** each full-7B ZeRO-3 checkpoint is ~100 GB; `SAVE_TOTAL_LIMIT` (default 2) sets how many are kept — lower it or free space if tight. `SAVE_STEPS` (default 500) sets cadence (≤ ~25 min lost on resume).
- **Edge case:** if a crash lands *exactly during* a save, the newest `checkpoint-NNNN` may be partial and resume will fail to load it — `rm -rf` that newest checkpoint and re-run (it falls back to the previous one).
