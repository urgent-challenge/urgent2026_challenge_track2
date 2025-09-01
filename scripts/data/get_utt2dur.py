#!/usr/bin/env python3
import argparse
from pathlib import Path

import torchaudio
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-scp", type=Path, required=True, help="Path to wav.scp")
    parser.add_argument("--out-scp", type=Path, required=True, help="Path to metric.scp")
    args = parser.parse_args()

    args.out_scp.parent.mkdir(parents=True, exist_ok=True)
    num_lines = sum(1 for _ in open(args.wav_scp, "r"))
    with open(args.wav_scp, "r") as wav_scp, open(args.out_scp, "w") as f:
        for line in tqdm(wav_scp, total=num_lines, desc="Computing utt2dur"):
            uid, audio_path = line.strip().split()
            info = torchaudio.info(audio_path)
            sr = info.sample_rate
            duration = info.num_frames / sr
            f.write(f"{uid} {duration:.3f}\n")
