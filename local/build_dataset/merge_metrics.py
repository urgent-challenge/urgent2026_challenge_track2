#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Callable, Optional


def read_scp(path: Path, key_type: Optional[Callable] = None, value_type: Optional[Callable] = None) -> dict[str, str]:
    result = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            uid, value = line.strip().split(maxsplit=1)
            if key_type is not None:
                uid = key_type(uid)
            if value_type is not None:
                value = value_type(value)
            result[uid] = value
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert CSV to JSONL")
    parser.add_argument("data", type=Path, help="Path to data dir")
    parser.add_argument("jsonl_path", type=Path, help="Path to output JSONL file")

    args = parser.parse_args()

    metric_scps = [path for path in args.data.glob("*.scp") if path.name != "wav.scp"]

    utt2audio_path = read_scp(args.data / "wav.scp")
    utt2sys = read_scp(args.data / "utt2sys")
    utt2dur = read_scp(args.data / "utt2dur", value_type=float)

    metric_to_utt2val = {}
    for metric_scp in metric_scps:
        metric = metric_scp.stem
        metric_to_utt2val[metric] = read_scp(metric_scp, value_type=lambda x: round(float(x), 4))

    with open(args.jsonl_path, "w") as jsonl_file:
        for uid in sorted(utt2audio_path.keys()):
            audio_path = utt2audio_path[uid]
            item = {
                "audio_path": audio_path,
                "system_id": utt2sys[uid],
                "sample_id": uid,
                "duration": utt2dur[uid],
                "metrics": {},
            }
            for metric, utt2val in metric_to_utt2val.items():
                if uid in utt2val:
                    if "metrics" not in item:
                        item["metrics"] = {}
                    item["metrics"][metric] = utt2val[uid]
            jsonl_file.write(json.dumps(item) + "\n")
