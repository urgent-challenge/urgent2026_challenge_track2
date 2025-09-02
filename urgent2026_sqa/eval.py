import argparse
import json
from pathlib import Path

from urgent2026_sqa.utils import calculate_metrics

"""
Example jsonl line for ref and pred:
{"sample_id": "sys0001-utt0001", "system_id": "sys0001", "metrics": {"mos": 3.5, "lps": 0.8}}
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, type=Path, help="path to ref jsonl file")
    parser.add_argument("--pred", required=True, type=Path, help="path to pred jsonl file")
    parser.add_argument("--ref-metric", type=str, default="mos", help="metric key in ref file")
    parser.add_argument("--pred-metric", type=str, default="mos", help="metric key in pred file")

    args = parser.parse_args()
    return args


def load_jsonl(path: Path, metric: str) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            items.append(
                {
                    "sample_id": item["sample_id"],
                    "system_id": item["system_id"],
                    "value": item["metrics"][metric],
                }
            )
    return items


if __name__ == "__main__":

    args = get_args()
    refs, preds = [], []
    preds = load_jsonl(args.pred, args.pred_metric)
    refs = load_jsonl(args.ref, args.ref_metric)
    results = calculate_metrics(preds, refs)
    print(json.dumps(results, indent=4))
