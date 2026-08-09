# Paper Evaluation Coverage

This repository includes TGIF-native evaluation scripts for the LLaVA-format benchmarks and a VLMEvalKit wrapper for benchmarks that are already maintained there.

| Paper benchmark | Included runner | Notes |
| --- | --- | --- |
| HallusionBench | `scripts/eval_vlmevalkit.sh` | Uses VLMEvalKit `HallusionBench` / `ImageYORNDataset`. |
| POPE | `scripts/exp/eval/pope.sh` or `scripts/v1_5/eval/pope.sh` | Native LLaVA-format runner and scorer. |
| TextVQA | `scripts/exp/eval/textvqa.sh` or `scripts/v1_5/eval/textvqa.sh` | Native LLaVA-format runner and scorer. |
| OCRBench | `scripts/eval_vlmevalkit.sh` | Uses VLMEvalKit `OCRBench`. |
| MMRel | `scripts/v1_5/eval/mmrel_yn.sh` | TGIF-specific yes/no relation-grounding runner and scorer. |
| MMBench | `scripts/exp/eval/mmbench.sh`, `scripts/v1_5/eval/mmbench.sh`, or `scripts/eval_vlmevalkit.sh` | Native runner produces upload files; VLMEvalKit can run `MMBench_DEV_EN`. |
| ScienceQA | `scripts/exp/eval/sqa.sh`, `scripts/v1_5/eval/sqa.sh`, or `scripts/eval_vlmevalkit.sh` | Native runner and VLMEvalKit `ScienceQA_VAL` are both supported. |
| GQA | `scripts/exp/eval/gqa.sh`, `scripts/v1_5/eval/gqa.sh`, or `scripts/eval_vlmevalkit.sh` | Native runner and VLMEvalKit `GQA_TestDev_Balanced` are both supported. |

## VLMEvalKit Usage

Install TGIF and VLMEvalKit in the same Python environment so `import llava` resolves to this repository:

```bash
pip install -e .
cd /path/to/VLMEvalKit
pip install -e .
```

Run the hallucination/OCR benchmarks used in the paper:

```bash
cd /path/to/TGIF
MODEL_PATH=/path/or/hf-id/to/tgif-checkpoint \
VLMEVALKIT_ROOT=/path/to/VLMEvalKit \
DATASETS="HallusionBench OCRBench" \
bash scripts/eval_vlmevalkit.sh
```

Run all supported VLMEvalKit-backed paper benchmarks:

```bash
MODEL_PATH=/path/or/hf-id/to/tgif-checkpoint \
VLMEVALKIT_ROOT=/path/to/VLMEvalKit \
DATASETS="HallusionBench OCRBench TextVQA_VAL MMBench_DEV_EN ScienceQA_VAL GQA_TestDev_Balanced" \
bash scripts/eval_vlmevalkit.sh
```

POPE and MMRel should still be run with the native TGIF/LLaVA scripts because they use repository-specific data layout and scoring code.
