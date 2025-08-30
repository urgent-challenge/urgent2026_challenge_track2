#!/usr/bin/env python3

from pathlib import Path

import pandas as pd
from tqdm import tqdm


def csv2scps(csv_path, output_dir: Path):
    df = pd.read_csv(csv_path)
    if "score" in df.columns:
        df["mos"] = df.groupby("wav_path")["score"].transform("mean")
    unique_audios = set()
    utt2sys, utt2audio_path, utt2mos = {}, {}, {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating scp files"):
        audio_path = row["wav_path"]
        if audio_path in unique_audios:
            continue
        unique_audios.add(audio_path)
        assert row["sample_id"] not in utt2sys, f"duplicate sample_id: {row['sample_id']}"
        utt2sys[row["sample_id"]] = row["system_id"]
        utt2audio_path[row["sample_id"]] = row["wav_path"]

        if "mos" in row:
            utt2mos[row["sample_id"]] = round(float(row["mos"]), 4)

    uids = sorted(list(utt2audio_path.keys()))
    with open(output_dir / "utt2sys", "w") as utt2sys_scp, open(output_dir / "wav.scp", "w") as wav_scp:
        for uid in uids:
            utt2sys_scp.write(f"{uid} {utt2sys[uid]}\n")
            wav_scp.write(f"{uid} {utt2audio_path[uid]}\n")

    if len(utt2mos) > 0:
        with open(output_dir / "mos.scp", "w") as mos_scp:
            for uid in uids:
                mos_scp.write(f"{uid} {utt2mos[uid]:.4f}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mkae utt2sys, wav.scp, mos.scp from csv")
    parser.add_argument("csv_path", type=Path, help="Path to the input CSV file")
    parser.add_argument("output_dir", type=Path, help="Path to the output dir")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv2scps(args.csv_path, args.output_dir)
    print(f"Wrote scps to {args.output_dir}")
