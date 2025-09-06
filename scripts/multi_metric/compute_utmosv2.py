import argparse
from pathlib import Path

import torch
import utmosv2
from tqdm import tqdm


def utmosv2_metric(model, audio_path):
    mos = model.predict(input_path=audio_path)
    return mos


if __name__ == "__main__":
    raise NotImplementedError("UTMOSv2 is too slow for this implementation. ")

    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-scp", type=Path, required=True, help="Path to wav.scp")
    parser.add_argument("--out-scp", type=Path, required=True, help="Path to metric.scp")

    args = parser.parse_args()

    args.out_scp.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    utmos_model = utmosv2.create_model(pretrained=True, device=device)

    scores = []
    num_lines = sum(1 for _ in open(args.wav_scp, "r"))
    with open(args.wav_scp, "r") as wav_scp:
        for line in tqdm(wav_scp, total=num_lines, desc="Computing UTMOSv2 scores"):
            uid, audio_path = line.strip().split()
            score = utmosv2_metric(utmos_model, audio_path)
            scores.append((uid, score))

    with open(args.out_scp, "w") as metric_scp:
        for uid, score in scores:
            metric_scp.write(f"{uid} {score}\n")
