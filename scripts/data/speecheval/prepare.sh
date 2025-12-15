#!/usr/bin/env bash
set -e

db=$1

echo "===== Start preparing [speecheval] dataset ====="

mkdir -p ${db}/data data/speecheval

# download dataset
if [ ! -e ${db}/download.done ]; then
    mkdir -p ${db}
    pushd ${db}
    # wget https://zenodo.org/records/7378801/files/somos.zip
    hf download  --local-dir rawdata --repo-type dataset Hui519/SpeechEval
    popd
    echo "Successfully finished download."
    touch ${db}/download.done
else
    echo "Already exists. Skip download."
fi


if [ ! -f data/speecheval/train/data.jsonl ]; then
    scripts/data/speecheval/data_prep.py --data "${db}" --split "train" --out "data/speecheval/train/data.jsonl"
fi

if [ ! -f data/speecheval/dev/data.jsonl ]; then
    scripts/data/speecheval/data_prep.py --data "${db}" --split "validation" --out  "data/speecheval/dev/data.jsonl"
fi

if [ ! -f data/speecheval/test/data.jsonl ]; then
    scripts/data/speecheval/data_prep.py --data "${db}" --split "test" --out  "data/speecheval/test/data.jsonl"
fi

echo "===== Finished preparing [speecheval] dataset ====="