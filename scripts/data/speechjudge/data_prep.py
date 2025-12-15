#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import torchaudio
from datasets import load_dataset
from tqdm import tqdm


def prepare_data(data: Path, split: str):
    items = []
    for sample in tqdm(load_dataset("RMSnow/SpeechJudge-Data", split=split)):
        breakpoint()
        wav_file_a = data / f"{sample['index']}_a.wav"
        wav_file_b = data / f"{sample['index']}_b.wav"
        if not wav_file_a.exists():
            samples = sample["audioA"].get_all_samples()
            torchaudio.save(wav_file_a, samples.data, samples.sample_rate)
        if not wav_file_b.exists():
            samples = sample["audioB"].get_all_samples()
            torchaudio.save(wav_file_b, samples.data, samples.sample_rate)

        del sample["audioA"]
        del sample["audioB"]
        preference = sample["naturalness_label"]
        assert preference in ["A", "B"], f"Unexpected preference: {preference}"
        cmos = 1 if preference == "A" else -1

        sample = {
            "audios": [wav_file_a.absolute().as_posix(), wav_file_b.absolute().as_posix()],
            "cmos": cmos,
        }
        breakpoint()
        items.append(sample)
    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "validation", "test"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.data.mkdir(parents=True, exist_ok=True)
    items = prepare_data(args.data, args.split)
    df = pd.DataFrame(items)
    df.to_csv(args.out, index=False)
