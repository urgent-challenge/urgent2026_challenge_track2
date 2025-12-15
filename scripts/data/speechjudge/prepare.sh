#!/usr/bin/env bash
set -e

db=$1

echo "===== Start preparing [speecheval] dataset ====="

mkdir -p ${db}/data data/speechjudge


if [ ! -f data/speechjudge/train/data.jsonl ]; then
    scripts/data/speechjudge/data_prep.py --data "${db}" --split "train" --out "data/speechjudge/train/data.jsonl"
fi

if [ ! -f data/speechjudge/dev/data.jsonl ]; then
    scripts/data/speechjudge/data_prep.py --data "${db}" --split "dev" --out  "data/speechjudge/dev/data.jsonl"
fi

if [ ! -f data/speechjudge/test/data.jsonl ]; then
    scripts/data/speechjudge/data_prep.py --data "${db}" --split "test" --out  "data/speechjudge/test/data.jsonl"
fi

if [ ! -f data/speechjudge/other/data.jsonl ]; then
    scripts/data/speechjudge/data_prep.py --data "${db}" --split "other" --out  "data/speechjudge/other/data.jsonl"
fi

echo "===== Finished preparing [speecheval] dataset ====="