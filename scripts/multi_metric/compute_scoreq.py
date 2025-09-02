import argparse
from pathlib import Path

import numpy as np
import scoreq  # pip install scoreq
import torch
from tqdm import tqdm


def scoreq_metric(model, audio_path):
    with torch.no_grad():
        pred_mos = model.predict(test_path=audio_path, ref_path=None)
    return pred_mos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-scp", type=Path, required=True, help="Path to wav.scp")
    parser.add_argument("--out-scp", type=Path, required=True, help="Path to metric.scp")
    args = parser.parse_args()
    args.out_scp.parent.mkdir(parents=True, exist_ok=True)

    model = scoreq.Scoreq(data_domain="natural", mode="nr")

    scores = []
    num_lines = sum(1 for _ in open(args.wav_scp, "r"))
    with open(args.wav_scp, "r") as wav_scp:
        for line in tqdm(wav_scp, total=num_lines, desc="Computing ScoreQ scores"):
            uid, audio_path = line.strip().split()
            score = scoreq_metric(model, audio_path)
            scores.append((uid, score))

    with open(args.out_scp, "w") as metric_scp:
        for uid, score in scores:
            metric_scp.write(f"{uid} {score}\n")
