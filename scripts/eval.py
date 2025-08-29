import argparse
import json
from utils import compute_metrics
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, type=Path, help="path to ref jsonl file")
    parser.add_argument("--pred", required=True, type=Path, help="path to pred jsonl file")
    parser.add_argument("--metric", default="mos", type=str, help="metric to evaluate")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()
    with open

    pass
