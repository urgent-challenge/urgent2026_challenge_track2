#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import torchaudio
from datasets import load_dataset
from tqdm import tqdm


def prepare_data(data: Path, split: str):
    if split == "train":
        # NOTE: The blind test set has both MOS-labeled and unlabeled samples
        # enhanced from the same set of noisy base recordings. Using unlabeled
        # samples for training is optional: they don't overlap with labeled
        # test data but share the same source, so opinions on fairness may differ.
        phases = ["validation", "nonblind_test"]  # + ["blind_test"]
    elif split == "test":
        phases = ["blind_test_mos"]

    items = []
    for phase in phases:
        for sample in tqdm(load_dataset("urgent-challenge/urgent2024-sqa", split=phase)):
            submission_id = sample["system_id"].rsplit("_", 1)[1]
            fileid = sample["sample_id"].rsplit("_", 1)[1]
            wav_file = data / phase / submission_id / f"{fileid}.flac"
            wav_file.parent.mkdir(parents=True, exist_ok=True)
            if not wav_file.exists():
                samples = sample["audio"].get_all_samples()
                torchaudio.save(wav_file, samples.data, samples.sample_rate)
            del sample["audio"]
            sample["wav_path"] = wav_file.absolute().as_posix()
            if "raw_ratings" in sample and sample["raw_ratings"] is not None:
                raw_ratings = sample["raw_ratings"]
                del sample["raw_ratings"]
                for score in raw_ratings:
                    sample_ = {k: v for k, v in sample.items() if v is not None}
                    sample_["listener_id"] = None
                    sample_["score"] = score
                    items.append(sample_)
            else:
                items.append(sample)
    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    items = prepare_data(args.data, args.split)
    df = pd.DataFrame(items)
    df.to_csv(args.out, index=False)
