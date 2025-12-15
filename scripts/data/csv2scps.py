#!/usr/bin/env python3

from pathlib import Path

import pandas as pd
from tqdm import tqdm

METRICS = [
    "distill_mos",
    "dnsmos_ovrl",
    "estoi",
    "lps",
    "lsd",
    "mcd",
    "mos",
    "nisqa_mos",
    "pesqc2",
    "pesq",
    "sbert",
    "scoreq",
    "sdr",
    "sigmos_col",
    "sigmos_disc",
    "sigmos_loud",
    "sigmos_noise",
    "sigmos_ovrl",
    "sigmos_reverb",
    "sigmos_sig",
    "spksim",
    "utmos",
]

EXTRA_FIELDS = [
    "raw_ratings",
    "listeners",
]


def csv2scps(csv_path, output_dir: Path):
    df = pd.read_csv(csv_path, dtype=str)
    if "score" in df.columns:
        # convert score to float
        df["score"] = df["score"].astype(float)
        df["mos"] = df.groupby("wav_path")["score"].transform("mean")
        # make a raw_ratings column as list of raw ratings
        if "listener_id" in df.columns:
            df["raw_ratings"] = df.groupby("wav_path")["score"].transform(lambda x: ";".join(str(_) for _ in x))
            df["listeners"] = df.groupby("wav_path")["listener_id"].transform(lambda x: ";".join(str(_) for _ in x))
            del df["score"]
            del df["listener_id"]
    unique_audios = set()
    utt2system, utt2sample, utt2audio_path, field_to_utt2value = {}, {}, {}, {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating scp files"):
        audio_path = row["wav_path"]
        if audio_path in unique_audios:
            continue
        unique_audios.add(audio_path)
        system_id, sample_id = row["system_id"], row["sample_id"]
        uttid = f"{system_id}:{sample_id}"
        assert uttid not in utt2system, f"duplicate sample_id: {uttid}"
        utt2system[uttid] = system_id
        utt2sample[uttid] = sample_id
        utt2audio_path[uttid] = row["wav_path"]

        for key in METRICS + EXTRA_FIELDS:
            if key not in row:
                continue
            if key not in field_to_utt2value:
                field_to_utt2value[key] = {}
            if isinstance(row[key], (float, int)):
                field_to_utt2value[key][uttid] = round(float(row[key]), 4)
            else:
                field_to_utt2value[key][uttid] = row[key]

    uids = sorted(list(utt2audio_path.keys()))
    with (
        open(output_dir / "utt2system", "w") as utt2sys_scp,
        open(output_dir / "utt2sample", "w") as utt2sample_scp,
        open(output_dir / "wav.scp", "w") as wav_scp,
    ):
        for uid in uids:
            utt2sys_scp.write(f"{uid} {utt2system[uid]}\n")
            wav_scp.write(f"{uid} {utt2audio_path[uid]}\n")
            utt2sample_scp.write(f"{uid} {utt2sample[uid]}\n")

    for field, utt2value in field_to_utt2value.items():
        with open(output_dir / f"{field}.scp", "w") as field_scp:
            for uid in uids:
                if isinstance(utt2value[uid], str):
                    field_scp.write(f"{uid} {utt2value[uid]}\n")
                else:
                    field_scp.write(f"{uid} {utt2value[uid]:.4f}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Make utt2system, utt2sample, wav.scp, mos.scp from csv")
    parser.add_argument("csv_path", type=Path, help="Path to the input CSV file")
    parser.add_argument("output_dir", type=Path, help="Path to the output dir")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv2scps(args.csv_path, args.output_dir)
    print(f"Wrote scps to {args.output_dir}")
