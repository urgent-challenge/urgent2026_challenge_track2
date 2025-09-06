import argparse
from pathlib import Path

import distillmos
import torch
import torchaudio
from tqdm import tqdm

TARGET_FS = 16000


@torch.inference_mode()
def distill_mos_metric(model, audio_path, device):
    x, sr = torchaudio.load(audio_path)
    x = x[0, None, :]
    # resample to 16kHz if needed
    if sr != TARGET_FS:
        x = torchaudio.transforms.Resample(sr, TARGET_FS)(x)

    mos = model(x.to(device=device))
    return mos.cpu().item()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-scp", type=Path, required=True, help="Path to wav.scp")
    parser.add_argument("--out-scp", type=Path, required=True, help="Path to metric.scp")

    args = parser.parse_args()

    args.out_scp.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sqa_model = distillmos.ConvTransformerSQAModel().to(device=device)
    sqa_model.eval()

    scores = []
    num_lines = sum(1 for _ in open(args.wav_scp, "r"))
    with open(args.wav_scp, "r") as wav_scp:
        for line in tqdm(wav_scp, total=num_lines, desc="Computing DistillMos scores"):
            uid, audio_path = line.strip().split()
            score = distill_mos_metric(sqa_model, audio_path, device)
            scores.append((uid, score))

    with open(args.out_scp, "w") as metric_scp:
        for uid, score in scores:
            metric_scp.write(f"{uid} {score}\n")
