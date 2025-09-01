#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
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
        for parquet_file in (data / phase).glob("*.parquet"):
            table = pq.read_table(parquet_file)
            for row in tqdm(table.to_pylist(), desc=f"Processing {parquet_file}"):
                submission_id, _, fileid = row["sample_id"].split("_")[-3:]
                wav_file = data / phase / submission_id / f"{fileid}.flac"
                wav_file.parent.mkdir(parents=True, exist_ok=True)

                if not wav_file.exists():
                    with open(wav_file, "wb") as f:
                        f.write(row["audio"])
                del row["audio"]
                row["wav_path"] = wav_file.absolute().as_posix()
                items.append(row)
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
