#!/usr/bin/env python3

import json

import pandas as pd
import torchaudio
from tqdm import tqdm


def csv2jsonl(csv_path, jsonl_path):
    df = pd.read_csv(csv_path)
    if "score" in df.columns:
        df["mos"] = df.groupby("wav_path")["score"].transform("mean")
    unique_audios = set()
    items = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting to JSONL"):
        audio_path = row["wav_path"]
        if audio_path in unique_audios:
            continue
        unique_audios.add(audio_path)
        info = torchaudio.info(audio_path)
        sr = info.sample_rate
        duration = info.num_frames / sr
        item = {
            "audio_path": row["wav_path"],
            "system_id": row["system_id"],
            "sample_id": row["sample_id"],
            "duration": duration,
            "sr": sr,
        }
        if "mos" in row:
            item["metrics"] = {"mos": round(float(row["mos"]), 4)}
        items.append(item)

    with open(jsonl_path, "w") as jsonl_file:
        for item in items:
            jsonl_file.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert CSV to JSONL")
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file")
    parser.add_argument("jsonl_path", type=str, help="Path to the output JSONL file")
    args = parser.parse_args()
    csv2jsonl(args.csv_path, args.jsonl_path)
    print(f"Converted {args.csv_path} to {args.jsonl_path}")
