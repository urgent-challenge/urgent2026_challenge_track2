#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=str, help="Path to the input Excel file.")
    parser.add_argument("--wavdir", type=str, help="Sheet name in the Excel file.")
    parser.add_argument("--output-csv", type=str, default="converted.csv", help="Path to the output CSV file.")
    args = parser.parse_args()

    df = pd.read_excel(args.xlsx, sheet_name="Subjective Test Scores")

    df = df[["Filename", "ConditionID", "sample MOS"]].copy()

    df["wav_path"] = df["Filename"].apply(lambda x: os.path.join(args.wavdir, x))
    df["system_id"] = df["ConditionID"].astype(str).str.zfill(3)
    df["sample_id"] = df["Filename"].apply(lambda x: os.path.splitext(x)[0])

    df["score"] = df["sample MOS"]

    listener_id = ""
    df["listener_id"] = listener_id
    df["listener_idx"] = ""

    df_out = df[["wav_path", "system_id", "sample_id", "score", "listener_id", "listener_idx"]]
    df_out.to_csv(args.output_csv, index=False)
