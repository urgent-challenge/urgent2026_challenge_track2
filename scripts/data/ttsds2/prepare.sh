#!/usr/bin/env bash
set -e

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

db=$1

echo "===== Start preparing [TTSDS2] dataset ====="
# download dataset
cwd=`pwd`
if [ ! -e ${db}/download.done ]; then
    mkdir -p ${db}
    pushd ${db}
    hf download  --local-dir . --repo-type dataset urgent-challenge/urgent26_track2_sqa ttsds2.zip
    unzip ttsds2.zip
    rm ttsds2.zip
    popd
    echo "Successfully finished download."
    touch ${db}/download.done
else
    echo "Already exists. Skip download."
fi


mkdir -p ${db}/data data/ttsds2

if [ ! -e data/ttsds2/train/data.jsonl ]; then
    scripts/data/ttsds2/data_prep.py \
        --original-path "${db}/subjective_results.csv" --wavdir "${db}" --out "${db}/data/train.csv" --seed 1337
    scripts/data/csv2data.sh "${db}/data/train.csv" "data/ttsds2/train"
fi

echo "===== Finished preparing [TTSDS2] dataset ====="