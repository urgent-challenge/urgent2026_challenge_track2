#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-path",
        required=True,
        type=Path,
        help=("original csv file path."),
    )
    parser.add_argument(
        "--wavdir",
        required=True,
        type=Path,
        help=(
            "directory of the waveform files. This is needed because wav paths in BVCC metadata files do not contain the wav directory."
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help=("output csv file path."),
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    num_lines = sum(1 for _ in open(args.original_path)) - 1
    with open(args.original_path, newline="") as infile, open(args.out, "w", newline="") as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=["wav_path", "system_id", "sample_id", "score", "listener_id"])
        writer.writeheader()

        for row in tqdm(reader, total=num_lines):
            if row["condition"].lower() == "ref":
                continue  # Skip ref rows

            wav_name = row["sample"]
            sample_id = wav_name.rsplit(".", 1)[0]
            system_id = row["condition"]
            wav_path = (args.wavdir / system_id / wav_name).resolve().as_posix()

            score = row["MOS"]
            writer.writerow(
                {
                    "wav_path": wav_path,
                    "system_id": system_id,
                    "sample_id": sample_id,
                    "score": float(score),
                    "listener_id": "",
                }
            )
