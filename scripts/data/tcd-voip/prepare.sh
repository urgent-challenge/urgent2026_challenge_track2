#!/usr/bin/env bash
set -e

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

db=$1

echo "===== Start preparing [TCD-VOIP] dataset ====="

# download dataset
if [ ! -e ${db}/download.done ]; then
    mkdir -p ${db}
    pushd ${db}
    # gdown 1rHJN34vP-W8SJtjpNUnx5RIks3o5L5he
    hf download  --local-dir . --repo-type dataset urgent-challenge/urgent26_track2_sqa TCD-VOIP.zip
    unzip TCD-VOIP.zip
    rm TCP-VOIP.zip
    mkdir -p wav
    find TCD-VOIP -type f -name "*.wav" -exec mv {} wav/ \;
    mv TCD-VOIP/README.txt .
    mv TCD-VOIP/*.xlsx  mos.xlsx
    rm -r TCD-VOIP
    popd
    echo "Successfully finished download."
    touch ${db}/download.done
else
    echo "Already exists. Skip download."
fi


mkdir -p ${db}/data data/tcd-voip
if [ ! -e data/tcd-voip/train/data.jsonl ]; then
    scripts/data/tcd-voip/data_prep.py \
        --xlsx "${db}/mos.xlsx" --wavdir "${db}/wav" --out "${db}/data/train.csv"
    scripts/data/csv2data.sh "${db}/data/train.csv" "data/tcd-voip/train"
fi

echo "===== Finished preparing [TCD-VOIP] dataset ====="