"""Score MMRel Discriminative_YN_QA predictions.

Usage:
    python eval_mmrel_yn.py --results-dir <dir_with_jsonl_files>
    python eval_mmrel_yn.py --answers-file <single_file.jsonl>
"""
import argparse
import json
from pathlib import Path


def score_file(path):
    correct, total = 0, 0
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            if item["pred"] == item["label"]:
                correct += 1
            total += 1
    return correct, total


def main(args):
    if args.answers_file:
        files = [Path(args.answers_file)]
    else:
        files = sorted(Path(args.results_dir).glob("*.jsonl"))

    if not files:
        print("No .jsonl files found.")
        return

    total_correct, total_count = 0, 0
    col_w = 32

    print(f"\n{'Subset':<{col_w}} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * (col_w + 30))

    for f in files:
        correct, total = score_file(f)
        acc = correct / total if total > 0 else 0.0
        total_correct += correct
        total_count += total
        print(f"{f.stem:<{col_w}} {correct:>8} {total:>8} {acc:>9.1%}")

    if len(files) > 1:
        overall = total_correct / total_count if total_count > 0 else 0.0
        print("-" * (col_w + 30))
        print(f"{'OVERALL':<{col_w}} {total_correct:>8} {total_count:>8} {overall:>9.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results-dir", type=str)
    group.add_argument("--answers-file", type=str)
    args = parser.parse_args()
    main(args)
