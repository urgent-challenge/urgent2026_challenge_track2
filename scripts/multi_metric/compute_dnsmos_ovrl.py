import argparse
from pathlib import Path

import requests
import soundfile as sf
import soxr
import torch
from espnet2.enh.layers.dnsmos import DNSMOS_local
from tqdm import tqdm

TARGET_FS = 16000
primary_model_path = Path("local/build_dataset/DNSMOS/sig_bak_ovr.onnx")
p808_model_path = Path("local/build_dataset/DNSMOS/model_v8.onnx")


def download_file(url: str, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(save_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)


def dnsmos_metric(model, audio_path):
    audio, fs = sf.read(audio_path, dtype="float32")
    assert audio.ndim == 1, audio.shape
    if fs != TARGET_FS:
        audio = soxr.resample(audio, fs, TARGET_FS)
        fs = TARGET_FS
    with torch.no_grad():
        dnsmos_score = model(audio, fs)
    return float(dnsmos_score["OVRL"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-scp", type=Path, required=True, help="Path to wav.scp")
    parser.add_argument("--out-scp", type=Path, required=True, help="Path to metric.scp")
    args = parser.parse_args()
    args.out_scp.parent.mkdir(parents=True, exist_ok=True)

    model = DNSMOS_local(
        primary_model_path.as_posix(),
        p808_model_path.as_posix(),
        use_gpu=torch.cuda.is_available(),
        convert_to_torch=False,
    )

    scores = []
    num_lines = sum(1 for _ in open(args.wav_scp, "r"))
    with open(args.wav_scp, "r") as wav_scp:
        for line in tqdm(wav_scp, total=num_lines, desc="Computing DNSMOS scores"):
            uid, audio_path = line.strip().split()
            score = dnsmos_metric(model, audio_path)
            scores.append((uid, score))

    with open(args.out_scp, "w") as metric_scp:
        for uid, score in scores:
            metric_scp.write(f"{uid} {score}\n")
