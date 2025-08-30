#!/bin/bash
set -euo pipefail

# the data directory should contain wav.scp and utt2sys
data="$1"


if [ -z $data ]; then
    echo "Usage: $0 <data-dir>"
    exit 1
fi

if [ ! -f $data/wav.scp ] || [ ! -f $data/utt2sys ]; then
    echo "Error: $data/wav.scp does not exist"
    exit 1
fi

if [ ! -e $data/utt2dur ]; then
    python local/build_dataset/get_utt2dur.py --wav-scp $data/wav.scp  --out-scp $data/utt2dur
fi

if [ ! -e $data/nisqa.scp ]; then
    python local/build_dataset/compute_nisqa.py --wav-scp $data/wav.scp  --out-scp $data/nisqa.scp
fi

if [ ! -e $data/dnsmos.scp ]; then
    python local/build_dataset/compute_dnsmos.py --wav-scp $data/wav.scp --out-scp $data/dnsmos.scp
fi

if [ ! -e $data/scoreq.scp ]; then
    python local/build_dataset/compute_scoreq.py --wav-scp $data/wav.scp --out-scp $data/scoreq.scp
fi

if [ ! -e $data/utmos.scp ]; then
    python local/build_dataset/compute_utmos.py --wav-scp $data/wav.scp --out-scp $data/utmos.scp
fi

if [ ! -e $data/utmos.scp ]; then
    python local/build_dataset/compute_utmos.py --wav-scp $data/wav.scp --out-scp $data/utmos.scp
fi

python local/build_dataset/merge_metrics.py $data $data/data.jsonl


echo "Done. time elapsed: ${SECONDS}s"