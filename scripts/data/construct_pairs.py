import argparse
import itertools
import json
from pathlib import Path

from tqdm import tqdm

# python construct_pairs.py \
#     --input data/somos/dev/data.jsonl \
#     --output data_pairs/somos/dev.jsonl \
#     --loose


def construct_pairs(utt2system, utt2sample, utt2mos, utt2audio_path, loose):
    if not loose:
        sample2system_and_utts = {}
        for utt, system in utt2system.items():
            sample = utt2sample[utt]
            if sample not in sample2system_and_utts:
                sample2system_and_utts[sample] = []
            sample2system_and_utts[sample].append((system, utt))

        for sample, items in tqdm(sample2system_and_utts.items()):
            for _, utt1 in items:
                for _, utt2 in items:
                    if utt1 == utt2:
                        continue
                    pair = {
                        "audios": (utt2audio_path[utt1], utt2audio_path[utt2]),
                        "cmos": round(utt2mos[utt1] - utt2mos[utt2], 3),
                    }
                    yield pair
    else:
        utts = list(utt2mos.keys())
        for i in tqdm(range(len(utts))):
            for j in range(i + 1, len(utts)):
                utt1, utt2 = utts[i], utts[j]
                if utt1 == utt2:
                    continue
                pair = {
                    "audios": (utt2audio_path[utt1], utt2audio_path[utt2]),
                    "cmos": round(utt2mos[utt1] - utt2mos[utt2], 3),
                }
                yield pair


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct pairs of items from two input files.")
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to data.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to the output pairs jsonl file",
    )
    parser.add_argument(
        "--loose",
        action="store_true",
        help="whether only same sample pairs from different systems are allowed",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100_000,
        help="maximum number of pairs to construct",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    utt2system, utt2sample, utt2mos, utt2audio_path = {}, {}, {}, {}
    with open(args.input, "r") as f:
        for line in f:
            item = json.loads(line)
            uid = item["uid"]
            utt2mos[uid] = item["metrics"]["mos"]
            utt2audio_path[uid] = item["audio_path"]
            if not args.loose:
                utt2system[uid] = item["system_id"]
                utt2sample[uid] = item["sample_id"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_pairs = 0
    with open(args.output, "w") as f:
        for pair in itertools.islice(
            construct_pairs(utt2system, utt2sample, utt2mos, utt2audio_path, args.loose), args.limit
        ):
            n_pairs += 1
            f.write(json.dumps(pair) + "\n")
    print(f"Constructed {n_pairs} pairs and saved to {args.output}.")
    if n_pairs == 0:
        args.output.unlink()
