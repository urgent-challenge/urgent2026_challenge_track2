import argparse
from pathlib import Path

import librosa
import torch
from tqdm import tqdm


def utmos_metric(model, audio_path):
    wave, sr = librosa.load(audio_path, sr=None, mono=True)
    wave = torch.from_numpy(wave).unsqueeze(0).to(device=model.device)
    utmos_score = model(wave, sr)
    return float(utmos_score.cpu().item())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-scp", type=Path, required=True, help="Path to wav.scp")
    parser.add_argument("--out-scp", type=Path, required=True, help="Path to metric.scp")

    args = parser.parse_args()

    args.out_scp.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    utmos_model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device=device)
    utmos_model.device = device

    scores = []
    num_lines = sum(1 for _ in open(args.wav_scp, "r"))
    with open(args.wav_scp, "r") as wav_scp:
        for line in tqdm(wav_scp, total=num_lines, desc="Computing UTMOS scores"):
            uid, audio_path = line.strip().split()
            score = utmos_metric(utmos_model, audio_path)
            scores.append((uid, score))

    with open(args.out_scp, "w") as metric_scp:
        for uid, score in scores:
            metric_scp.write(f"{uid} {score}\n")
