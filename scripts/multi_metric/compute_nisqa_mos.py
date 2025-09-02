import argparse
import warnings
from pathlib import Path

import torch
from nisqa_utils import load_nisqa_model, predict_nisqa
from tqdm import tqdm

nisqa_model_path = Path("local/build_dataset/NISQA/weights/nisqa.tar")


def nisqa_metric(model, audio_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        nisqa_score = predict_nisqa(model, audio_path)
    return float(nisqa_score["mos_pred"])


def main(args):
    data_pairs = []
    with open(args.inf_scp, "r") as f:
        for line in f:
            uid, audio_path = line.strip().split()
            data_pairs.append((uid, audio_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-scp", type=Path, required=True, help="Path to wav.scp")
    parser.add_argument("--out-scp", type=Path, required=True, help="Path to metric.scp")
    args = parser.parse_args()
    args.out_scp.parent.mkdir(parents=True, exist_ok=True)

    assert (
        nisqa_model_path.exists()
    ), f"The NISQA model '{nisqa_model_path}' doesn't exist. run git submodule update --init --recursive"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_nisqa_model(nisqa_model_path.as_posix(), device=device)

    scores = []
    num_lines = sum(1 for _ in open(args.wav_scp, "r"))
    with open(args.wav_scp, "r") as wav_scp:
        for line in tqdm(wav_scp, total=num_lines, desc="Computing NISQA scores"):
            uid, audio_path = line.strip().split()
            score = nisqa_metric(model, audio_path)
            scores.append((uid, score))

    with open(args.out_scp, "w") as metric_scp:
        for uid, score in scores:
            metric_scp.write(f"{uid} {score}\n")
