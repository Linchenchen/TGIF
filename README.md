# TGIF

Official implementation for **TGIF: Text-Guided Layer Fusion Mitigates Hallucination in Multimodal LLMs**.

TGIF builds on LLaVA and keeps the original `llava` Python package layout for compatibility. The core method adds a text-guided layer-fusion projector (`mm_projector_type=layer_selector`) that routes over vision encoder layers using the input text representation, then projects the fused visual features into the language model.

This public release includes the canonical LayerSelector implementation used for TGIF. Earlier experimental projector variants are intentionally omitted to keep the code path focused and reproducible.

## News

- 2026-08: Initial public code release.

## Installation

This codebase follows the LLaVA environment. Linux with CUDA is recommended.

```bash
git clone https://github.com/Linchenchen/TGIF.git
cd TGIF
conda create -n tgif python=3.10 -y
conda activate tgif
pip install --upgrade pip
pip install -e .
```

For training:

```bash
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Artifacts

Weights and datasets are not stored in this repository.

- Base LLaVA checkpoints: use the public LLaVA/Vicuna checkpoints supported by the upstream LLaVA code.
- Released TGIF 7B checkpoint: [`cclinn/TGIF-LLaVA-v1.5-7B`](https://huggingface.co/cclinn/TGIF-LLaVA-v1.5-7B).
- TGIF finetuned checkpoints: pass a local checkpoint path or Hugging Face model id with `MODEL_PATH` / `CKPT_ROOT` in the evaluation scripts.
- Datasets: place LLaVA training and evaluation data under `playground/data` by default, or set `DATA_ROOT` for scripts that support an external data directory.

Example:

```bash
bash scripts/exp/eval/pope.sh cclinn/TGIF-LLaVA-v1.5-7B tgif-7b
```

## Training

The TGIF training path is `layer_selector`.

```bash
GPUS=0,1,2,3,4,5,6,7 GRAD_ACCUM=1 WANDB_MODE=offline \
  bash scripts/exp/finetune_from_hf.sh layer_selector
```

Common overrides:

```bash
HF_REPO=<hf_repo_with_projectors>
BASE_MODEL=lmsys/vicuna-7b-v1.5
VISION_TOWER=google/siglip-large-patch16-384
SAVE_STEPS=500
SAVE_TOTAL_LIMIT=2
REPORT_TO=wandb
```

The script is resumable: rerunning the same command resumes from an existing `checkpoint-*` in the output directory.

## Evaluation

The paper evaluation coverage is summarized in [docs/PAPER_EVALS.md](docs/PAPER_EVALS.md). In short, POPE, TextVQA, MMRel, MMBench, ScienceQA, and GQA are available through native LLaVA/TGIF scripts; HallusionBench and OCRBench are run through the VLMEvalKit wrapper in `scripts/eval_vlmevalkit.sh`.

### POPE

```bash
bash scripts/exp/eval/pope.sh <model_path> <checkpoint_name>
```

The script expects POPE files under `playground/data/eval/pope` unless you adapt the paths in the standard LLaVA data layout.

### MMRel Discriminative Yes/No QA

```bash
DATA_ROOT=/path/to/mmrel \
CKPT_ROOT=/path/to/checkpoints \
bash scripts/v1_5/eval/mmrel_yn.sh
```

Expected MMRel layout:

```text
$DATA_ROOT/
  json/Evaluation/Discriminative_YN_QA/*.jsonl
  <image files referenced by the jsonl questions>
```

Score a directory of answer files:

```bash
python -m llava.eval.eval_mmrel_yn --results-dir <answers_dir>
```

Score one answer file:

```bash
python -m llava.eval.eval_mmrel_yn --answers-file <answers.jsonl>
```

### HallusionBench and OCRBench

```bash
MODEL_PATH=/path/or/hf-id/to/tgif-checkpoint \
VLMEVALKIT_ROOT=/path/to/VLMEvalKit \
DATASETS="HallusionBench OCRBench" \
bash scripts/eval_vlmevalkit.sh
```

## Acknowledgements

This repository is based on [LLaVA](https://github.com/haotian-liu/LLaVA). We keep the original package structure and Apache-2.0 license notices for compatibility with LLaVA training, serving, and evaluation scripts.

## Citation

Please cite the TGIF paper if you use this code. BibTeX metadata will be added after the paper record is finalized.

```bibtex
@article{tgif2026,
  title={TGIF: Text-Guided Layer Fusion Mitigates Hallucination in Multimodal LLMs},
  year={2026}
}
```
