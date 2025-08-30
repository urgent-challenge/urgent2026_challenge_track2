#!/usr/bin/env bash
set -e

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

db=$1

echo "===== Start preparing [Tencent] dataset ====="

# download dataset
if [ ! -e ${db}/tencent.done ]; then
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
    touch ${db}/tencent.done
else
    echo "Already exists. Skip download."
fi


mkdir -p ${db}/data data/tencent
if [ ! -e data/tencent/train/wav.scp ]; then
    scripts/data/tencent/data_prep.py \
        --original-path "${db}/withoutReverberationTrainDevMOS.csv" "${db}/withReverberationTrainDevMOS.csv" \
        --wavdir "${db}" --out "${db}/data/tencent_train.csv" \
        --setname "train" --seed 1337
    scripts/data/csv2scps.py "${db}/data/tencent_train.csv" "data/tencent/train"
fi

if [ ! -e data/tencent/dev/wav.scp ]; then
    scripts/data/tencent/data_prep.py \
        --original-path "${db}/withoutReverberationTrainDevMOS.csv" "${db}/withReverberationTrainDevMOS.csv" \
        --wavdir "${db}" --out "${db}/data/tencent_dev.csv" \
        --setname "dev" --seed 1337
    scripts/data/csv2scps.py "${db}/data/tencent_dev.csv" "data/tencent/dev"
fi

echo "===== Finished preparing [Tencent] dataset ====="