#!/usr/bin/env bash
set -e

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

db=$1

echo "===== Start preparing [PSTN] dataset ====="

# download dataset
cwd=`pwd`
if [ ! -e ${db}/pstn.done ]; then
    mkdir -p ${db}
    pushd ${db}
    wget -c https://challenge.blob.core.windows.net/pstn/train.zip -O pstn.zip
    # hf download  --local-dir . --repo-type dataset urgent-challenge/urgent26_track2_sqa pstn.zip
    unzip pstn.zip
    # rm pstn.zip
    popd
    echo "Successfully finished download."
    touch ${db}/pstn.done
else
    echo "Already exists. Skip download."
fi

mkdir -p ${db}/data data/pstn

if [ ! -e data/pstn/train.jsonl ]; then
    scripts/data/pstn/data_prep.py \
        --original-path "${db}/pstn_train/pstn_train.csv" --wavdir "${db}/pstn_train" --setname "train" --out "${db}/data/pstn_train.csv" --seed 1337 --dev_ratio 0.0
    scripts/data/csv2jsonl.py "${db}/data/pstn_train.csv" "data/pstn/train.jsonl"
fi



echo "===== Finished preparing [PSTN] dataset ====="