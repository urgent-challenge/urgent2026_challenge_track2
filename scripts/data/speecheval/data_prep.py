#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import torchaudio
from datasets import load_dataset
from tqdm import tqdm


def prepare_data(data: Path, split: str):
    items = []
    for sample in tqdm(load_dataset(data / "rawdata", split=split)):
        breakpoint()
        audio_path_1 = sample["path"]
        audio_path_2 = sample["path_B"]
        wav_file = data / split / submission_id / f"{fileid}.flac"
        wav_file.parent.mkdir(parents=True, exist_ok=True)
        if not wav_file.exists():
            samples = sample["audio"].get_all_samples()
            torchaudio.save(wav_file, samples.data, samples.sample_rate)
        del sample["audio"]
        sample["wav_path"] = wav_file.absolute().as_posix()
        sample = {k: v for k, v in sample.items() if v is not None}
        items.append(sample)
    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "validation", "test"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    items = prepare_data(args.data, args.split)
    df = pd.DataFrame(items)
    df.to_csv(args.out, index=False)
