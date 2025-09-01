#!/bin/bash

csv=$1
data=$2


if [ -z $csv ] || [ -z $data ]; then
    echo "Usage: $0 <csv-file> <data-dir>"
    exit 1
fi

scripts/data/csv2scps.py $csv $data
if [ ! -e $data/utt2dur ]; then
    scripts/data/get_utt2dur.py --wav-scp $data/wav.scp  --out-scp $data/utt2dur
fi
scripts/data/collect_metrics.py $data $data/data.jsonl