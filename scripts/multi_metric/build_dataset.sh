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
    python scripts/data/get_utt2dur.py --wav-scp $data/wav.scp  --out-scp $data/utt2dur
fi

if [ ! -e $data/nisqa_mos.scp ]; then
    python scripts/multi_metric/compute_nisqa_mos.py --wav-scp $data/wav.scp  --out-scp $data/nisqa_mos.scp
fi

if [ ! -e $data/dnsmos_ovrl.scp ]; then
    python scripts/multi_metric/compute_dnsmos_ovrl.py --wav-scp $data/wav.scp --out-scp $data/dnsmos_ovrl.scp
fi

if [ ! -e $data/scoreq.scp ]; then
    python scripts/multi_metric/compute_scoreq.py --wav-scp $data/wav.scp --out-scp $data/scoreq.scp
fi

if [ ! -e $data/utmos.scp ]; then
    python scripts/multi_metric/compute_utmos.py --wav-scp $data/wav.scp --out-scp $data/utmos.scp
fi

python scripts/multi_metric/collect_metrics.py $data $data/data.jsonl


echo "Done. time elapsed: ${SECONDS}s"