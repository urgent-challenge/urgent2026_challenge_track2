#!/usr/bin/env python3

import json
import logging
from pathlib import Path
from typing import Callable, Optional

from csv2scps import METRICS


def read_scp(path: Path, key_type: Optional[Callable] = None, value_type: Optional[Callable] = None) -> dict[str, str]:
    if not path.exists():
        logging.warning(f"SCP file {path} does not exist, this can be normal for listeners.scp or moslist.scp")
        return {}
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

    metric_scps = [path for path in args.data.glob("*.scp") if path.stem in METRICS]

    utt2audio_path = read_scp(args.data / "wav.scp")
    utt2system = read_scp(args.data / "utt2system")
    utt2sample = read_scp(args.data / "utt2sample")
    utt2dur = read_scp(args.data / "utt2dur", value_type=float)

    utt2listeners = read_scp(args.data / "listeners.scp")
    utt2raw_ratings = read_scp(args.data / "raw_ratings.scp")

    metric_to_utt2score = {}
    for metric_scp in metric_scps:
        metric = metric_scp.stem
        metric_to_utt2score[metric] = read_scp(metric_scp, value_type=lambda x: round(float(x), 4))

    with open(args.jsonl_path, "w") as jsonl_file:
        for uid in sorted(utt2audio_path.keys()):
            audio_path = utt2audio_path[uid]
            item = {
                "audio_path": audio_path,
                "uid": uid,
                "system_id": utt2system[uid],
                "sample_id": utt2sample[uid],
                "duration": utt2dur[uid],
                "metrics": {},
            }
            if uid in utt2listeners and uid in utt2raw_ratings:
                if not (utt2listeners[uid][0] == "nan" and len(utt2listeners[uid]) <= 1):
                    item["listener2mos"] = [
                        (listener, float(mos))
                        for listener, mos in zip(utt2listeners[uid].split(";"), utt2raw_ratings[uid].split(";"))
                    ]
            for metric, utt2score in metric_to_utt2score.items():
                if uid in utt2score:
                    if "metrics" not in item:
                        item["metrics"] = {}
                    item["metrics"][metric] = utt2score[uid]
            jsonl_file.write(json.dumps(item) + "\n")
