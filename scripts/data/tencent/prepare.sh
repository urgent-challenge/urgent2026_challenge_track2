#!/usr/bin/env bash
set -e

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

db=$1

echo "===== Start preparing [Tencent] dataset ====="

# download dataset
if [ ! -e ${db}/download.done ]; then
    mkdir -p ${db}
    pushd ${db}
    # wget https://huggingface.co/datasets/urgent-challenge/urgent26_track2_sqa/resolve/main/TencentCorpus.zip
    hf download  --local-dir . --repo-type dataset urgent-challenge/urgent26_track2_sqa TencentCorpus.zip
    unzip TencentCorpus.zip
    mv TencentCorups/* .
    rmdir TencentCorups
    rm TencentCorpus.zip
    popd
    echo "Successfully finished download."
    touch ${db}/download.done
else
    echo "Already exists. Skip download."
fi


mkdir -p ${db}/data data/tencent
if [ ! -e data/tencent/train/data.jsonl ]; then
    scripts/data/tencent/data_prep.py \
        --original-path "${db}/withoutReverberationTrainDevMOS.csv" "${db}/withReverberationTrainDevMOS.csv" \
        --wavdir "${db}" --out "${db}/data/train.csv" \
        --setname "train" --seed 1337 --dev_ratio 0.0
    scripts/data/csv2data.sh "${db}/data/train.csv" "data/tencent/train"
fi

echo "===== Finished preparing [Tencent] dataset ====="